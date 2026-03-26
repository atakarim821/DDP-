#include <cuda.h>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <chrono>

#define MAX_NODE     100000000
#define INF_COST     0x7FFFFFFF
#define WARP_SIZE    32

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

using namespace std;

// __device__ volatile unsigned long long Cx[MAX_NODE];
__device__ unsigned long long* Cx;          // device-side pointer (lives in __device__ memory)
unsigned long long* D_cx_alloc = nullptr;   // host-side handle for cudaFree



__device__ __host__ __forceinline__
unsigned long long pack(unsigned int cost, int parent) {
    return ((unsigned long long)cost << 32) | (unsigned int)parent;
}
__device__ __host__ __forceinline__
unsigned int unpackCost(unsigned long long v) { return (unsigned int)(v >> 32); }
__device__ __host__ __forceinline__
int unpackParent(unsigned long long v)        { return (int)(v & 0xFFFFFFFF); }

struct GpuTimer {
    cudaEvent_t start, stop;
    void create()  { cudaEventCreate(&start); cudaEventCreate(&stop); }
    void begin()   { cudaEventRecord(start); }
    void end()     { cudaEventRecord(stop); }
    float ms() {
        cudaEventSynchronize(stop);
        float t = 0;
        cudaEventElapsedTime(&t, start, stop);
        return t;
    }
    void destroy() { cudaEventDestroy(start); cudaEventDestroy(stop); }
};

__global__ void find_min_f(const int* __restrict__ frontier, int frontier_size, unsigned int* __restrict__ min_f) {
    unsigned int local_min = INF_COST;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < frontier_size) {
        local_min = unpackCost(Cx[frontier[tid]]);
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        local_min = min(local_min, __shfl_down_sync(0xffffffff, local_min, offset));
    }

    __shared__ unsigned int shared_min[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    if (lane == 0) shared_min[wid] = local_min;
    __syncthreads();

    if (wid == 0) {
        local_min = (lane < (blockDim.x / 32)) ? shared_min[lane] : INF_COST;
        for (int offset = 16; offset > 0; offset /= 2) {
            local_min = min(local_min, __shfl_down_sync(0xffffffff, local_min, offset));
        }
        if (lane == 0 && local_min != INF_COST) {
            atomicMin(min_f, local_min);
        }
    }
}

__global__ void expand_delta(
    const int* __restrict__ off,
    const int* __restrict__ edge,
    const unsigned int* __restrict__ W,
    const int* __restrict__ Hx,
    const int* __restrict__ frontier,
    int frontier_size,
    unsigned int threshold,
    int* __restrict__ next_queue,
    int N, int E, int flagDiff,
    const int* __restrict__ diff_off,
    const int* __restrict__ diff_edge,
    const unsigned int* __restrict__ diff_weight,
    int dE)
{
    unsigned int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    unsigned int lane_id = threadIdx.x % WARP_SIZE;

    if (warp_id >= (unsigned int)frontier_size) return;

    int node = frontier[warp_id];
    unsigned int node_f = unpackCost(Cx[node]);

    if (node_f > threshold) {
        if (lane_id == 0) next_queue[node] = 1;
        return;
    }

    unsigned int node_hx = (unsigned int)__ldg(&Hx[node]);
    unsigned int node_g  = node_f - node_hx;

    int start = __ldg(&off[node]);
    int end   = (node != N-1) ? __ldg(&off[node+1]) : E;

    for (int i = start + (int)lane_id; i < end; i += WARP_SIZE) {
        int child = __ldg(&edge[i]);
        if (child < 0) continue;

        unsigned int w = __ldg(&W[i]);
        unsigned int hx_child = (unsigned int)__ldg(&Hx[child]);
        unsigned int new_cost = node_g + w + hx_child;
        unsigned long long new_val = pack(new_cost, node);

        unsigned long long old_val = atomicMin((unsigned long long*)&Cx[child], new_val);
        if (new_val < old_val) {
            next_queue[child] = 1;
        }
    }

    if (flagDiff) {
        start = __ldg(&diff_off[node]);
        end   = (node != N-1) ? __ldg(&diff_off[node+1]) : dE;

        for (int i = start + (int)lane_id; i < end; i += WARP_SIZE) {
            int child = __ldg(&diff_edge[i]);
            if (child < 0) continue;

            unsigned int w = __ldg(&diff_weight[i]);
            unsigned int hx_child = (unsigned int)__ldg(&Hx[child]);
            unsigned int new_cost = node_g + w + hx_child;
            unsigned long long new_val = pack(new_cost, node);

            unsigned long long old_val = atomicMin((unsigned long long*)&Cx[child], new_val);
            if (new_val < old_val) {
                next_queue[child] = 1;
            }
        }
    }
}

__global__ void compact_frontier(
    int* __restrict__ in_queue,
    int* __restrict__ frontier,
    int* __restrict__ frontier_size,
    int N)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < N && in_queue[id] == 1) {
        int pos = atomicAdd(frontier_size, 1);
        frontier[pos] = id;
        in_queue[id] = 0;
    }
}

__global__ void Updating(
    const int* __restrict__ u,
    const int* __restrict__ v,
    const unsigned int* __restrict__ W,
    const int* __restrict__ Hx,
    int* __restrict__ next_queue,
    int N, int dE)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= dE) return;

    int node = u[id], child = v[id];
    unsigned int node_f = unpackCost(Cx[node]);
    if (node_f == (unsigned int)INF_COST) return;

    unsigned int node_g = node_f - (unsigned int)__ldg(&Hx[node]);
    unsigned int new_cost = node_g + W[id] + (unsigned int)__ldg(&Hx[child]);

    unsigned long long new_val = pack(new_cost, node);
    unsigned long long old_val = atomicMin((unsigned long long*)&Cx[child], new_val);
    if (new_val < old_val) {
        next_queue[child] = 1;
    }
}

__global__ void getCx(int dest, int* val) {
    if (threadIdx.x == 0 && blockIdx.x == 0)
        *val = (int)unpackCost(Cx[dest]);
}

void createDiffGraph(int N,
                     unordered_map<int, vector<pair<int,int>>>& adj,
                     int* diffOff, int* diffEdges, unsigned int* diffWeight)
{
    int cnt = 0;
    for (int i = 0; i < N; i++) {
        diffOff[i] = cnt;
        auto it = adj.find(i);
        if (it != adj.end())
            for (auto& p : it->second) {
                diffEdges[cnt]  = p.first;
                diffWeight[cnt] = (unsigned int)p.second;
                cnt++;
            }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 3) return 1;

    nvtxRangePushA("Application_Total");

    auto wall_start = chrono::high_resolution_clock::now();

    int startNode = stoi(argv[1]);
    int endNode   = stoi(argv[2]);
    unsigned int DELTA = 200; 

    nvtxRangePushA("Read_Graph_File");
    FILE* fg = fopen("graph.txt", "r");
    if (!fg) {
        nvtxRangePop();
        nvtxRangePop();
        return 1;
    }

    int N, E;
    fscanf(fg, "%d %d\n", &N, &E);

    int* H_offset = (int*)malloc(sizeof(int)*N);
    int* H_edges  = (int*)malloc(sizeof(int)*E);
    unsigned int* H_weight = (unsigned int*)malloc(sizeof(unsigned int)*E);
    int* H_hx = (int*)malloc(sizeof(int)*N);
    unsigned long long* H_cx = (unsigned long long*)malloc(sizeof(unsigned long long)*N);
    int* H_dest_cost = (int*)malloc(sizeof(int));

    for (int i = 0; i < N; i++) H_cx[i] = pack(INF_COST, -1);

    for (int i = 0; i < E; i++) fscanf(fg, "%d",  &H_edges[i]);
    for (int i = 0; i < N; i++) fscanf(fg, "%d",  &H_offset[i]);
    for (int i = 0; i < E; i++) fscanf(fg, "%u",  &H_weight[i]);
    fclose(fg);
    nvtxRangePop();

    nvtxRangePushA("Read_Hx_File");
    FILE* fhx = fopen("Hx.txt", "r");
    for (int i = 0; i < N; i++) {
        H_hx[i] = 0;
        if (fhx) { int t; if (fscanf(fhx, "%d", &t) == 1) H_hx[i] = t; }
    }
    if (fhx) fclose(fhx);
    nvtxRangePop();

    H_cx[startNode] = pack((unsigned int)H_hx[startNode], -1);

    nvtxRangePushA("Device_Malloc");
    int *D_offset, *D_edges, *D_hx;
    unsigned int *D_weight;
    int *D_dest_cost;
    int *D_frontier, *D_next_queue, *D_frontier_size;
    unsigned int *D_min_f;
    int *D_diff_edges, *D_diff_offset;
    unsigned int *D_diff_weight;

    gpuErrchk(cudaMalloc(&D_offset, sizeof(int)*N));
    gpuErrchk(cudaMalloc(&D_edges, sizeof(int)*E));
    gpuErrchk(cudaMalloc(&D_weight, sizeof(unsigned int)*E));
    gpuErrchk(cudaMalloc(&D_hx, sizeof(int)*N));
    gpuErrchk(cudaMalloc(&D_dest_cost, sizeof(int)));
    gpuErrchk(cudaMalloc(&D_frontier, sizeof(int)*N));
    gpuErrchk(cudaMalloc(&D_next_queue, sizeof(int)*N));
    gpuErrchk(cudaMalloc(&D_frontier_size, sizeof(int)));
    gpuErrchk(cudaMalloc(&D_min_f, sizeof(unsigned int)));
    gpuErrchk(cudaMalloc(&D_cx_alloc, sizeof(unsigned long long) * N));
    nvtxRangePop();
    int numThreads = 512;
    int nBlocks = (N + numThreads - 1) / numThreads;

    int H_frontier_size = 1;

    nvtxRangePushA("Host_To_Device_Init");
    gpuErrchk(cudaMemcpy(D_offset, H_offset, sizeof(int)*N, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(D_edges, H_edges, sizeof(int)*E, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(D_weight, H_weight, sizeof(unsigned int)*E, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(D_hx, H_hx, sizeof(int)*N, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(D_cx_alloc, H_cx, sizeof(unsigned long long) * N,
                         cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(D_frontier, &startNode, sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(D_frontier_size, &H_frontier_size, sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemset(D_next_queue, 0, sizeof(int)*N));
    gpuErrchk(cudaMemcpyToSymbol(Cx, &D_cx_alloc, sizeof(unsigned long long *)));
    nvtxRangePop();

    GpuTimer timer_static_kernels;
    GpuTimer timer_static_total;
    timer_static_kernels.create();
    timer_static_total.create();

    timer_static_total.begin();
    timer_static_kernels.begin();

    unsigned int H_min_f;
    unsigned int INF_VAL = INF_COST;

    nvtxRangePushA("Static_AStar_Execution");
    while (H_frontier_size > 0) {
        nvtxRangePushA("Static_AStar_Iteration");
        
        nvtxRangePushA("Find_Min_Kernel");
        gpuErrchk(cudaMemcpy(D_min_f, &INF_VAL, sizeof(unsigned int), cudaMemcpyHostToDevice));
        int minBlocks = (H_frontier_size + numThreads - 1) / numThreads;
        find_min_f<<<minBlocks, numThreads>>>(D_frontier, H_frontier_size, D_min_f);
        gpuErrchk(cudaMemcpy(&H_min_f, D_min_f, sizeof(unsigned int), cudaMemcpyDeviceToHost));
        nvtxRangePop();

        nvtxRangePushA("Check_Destination");
        getCx<<<1, 1>>>(endNode, D_dest_cost);
        gpuErrchk(cudaMemcpy(H_dest_cost, D_dest_cost, sizeof(int), cudaMemcpyDeviceToHost));
        if (H_min_f >= (unsigned int)*H_dest_cost) {
            nvtxRangePop();
            nvtxRangePop();
            break;
        }
        nvtxRangePop();

        nvtxRangePushA("Expand_Kernel");
        unsigned int threshold = H_min_f + DELTA;
        int expBlocks = (WARP_SIZE * H_frontier_size + numThreads - 1) / numThreads;
        expand_delta<<<expBlocks, numThreads>>>(
            D_offset, D_edges, D_weight, D_hx,
            D_frontier, H_frontier_size, threshold,
            D_next_queue, N, E, 0, nullptr, nullptr, nullptr, 0);
        nvtxRangePop();

        nvtxRangePushA("Compact_Kernel");
        int H_zero = 0;
        gpuErrchk(cudaMemcpy(D_frontier_size, &H_zero, sizeof(int), cudaMemcpyHostToDevice));
        compact_frontier<<<nBlocks, numThreads>>>(D_next_queue, D_frontier, D_frontier_size, N);
        gpuErrchk(cudaMemcpy(&H_frontier_size, D_frontier_size, sizeof(int), cudaMemcpyDeviceToHost));
        nvtxRangePop();

        nvtxRangePop();
    }
    nvtxRangePop();

    timer_static_kernels.end();
    getCx<<<1, 1>>>(endNode, D_dest_cost);
    gpuErrchk(cudaMemcpy(H_dest_cost, D_dest_cost, sizeof(int), cudaMemcpyDeviceToHost));
    timer_static_total.end();

    auto wall_end_static = chrono::high_resolution_clock::now();
    double wall_static_ms = chrono::duration<double, milli>(wall_end_static - wall_start).count();

    printf("\n=== STATIC A* TIMING ===\n");
    printf("  Wall-clock: %.3f ms\n", wall_static_ms);
    printf("  GPU total:  %.3f ms\n", timer_static_total.ms());
    printf("  GPU kernels only: %.3f ms\n", timer_static_kernels.ms());
    printf("  Cost on static graph: %d\n\n", *H_dest_cost);

    nvtxRangePushA("Read_Updates_File_Open");
    FILE* fdiff = fopen("Updates.txt", "r");
    if (!fdiff) {
        nvtxRangePop();
        nvtxRangePop();
        return 0;
    }
    nvtxRangePop();

    GpuTimer timer_dyn_kernels;
    GpuTimer timer_dyn_total;
    timer_dyn_kernels.create();
    timer_dyn_total.create();

    int batch = 0;
    int line;

    nvtxRangePushA("Dynamic_Updates_Loop");
    while (fscanf(fdiff, "%d\n", &line) != EOF) {
        nvtxRangePushA("Dynamic_Batch_Iteration");
        batch++;

        nvtxRangePushA("Host_Read_and_Process_Batch");
        int* H_u = (int*)malloc(sizeof(int)*line);
        int* H_v = (int*)malloc(sizeof(int)*line);
        unsigned int* H_w = (unsigned int*)malloc(sizeof(unsigned int)*line);
        int* H_diff_edges  = (int*)malloc(sizeof(int)*line);
        int* H_diff_offset = (int*)malloc(sizeof(int)*N);
        unsigned int* H_diff_weight = (unsigned int*)malloc(sizeof(unsigned int)*line);
        unordered_map<int, vector<pair<int,int>>> adj;
        int insertEdge = 0;

        for (int i = 0; i < line; i++) {
            int flag, u, v; unsigned int w;
            fscanf(fdiff, "%d %d %d %u\n", &flag, &u, &v, &w);
            if (flag == 1) insertEdge++;
            H_u[i] = u; H_v[i] = v; H_w[i] = w;
            adj[u].push_back({v, (int)w});
        }
        createDiffGraph(N, adj, H_diff_offset, H_diff_edges, H_diff_weight);
        nvtxRangePop();

        nvtxRangePushA("Device_Malloc_Diff");
        int *D_u, *D_v; unsigned int *D_w;
        gpuErrchk(cudaMalloc(&D_u, sizeof(int)*line));
        gpuErrchk(cudaMalloc(&D_v, sizeof(int)*line));
        gpuErrchk(cudaMalloc(&D_w, sizeof(unsigned int)*line));
        gpuErrchk(cudaMalloc(&D_diff_edges, sizeof(int)*line));
        gpuErrchk(cudaMalloc(&D_diff_offset, sizeof(int)*N));
        gpuErrchk(cudaMalloc(&D_diff_weight, sizeof(unsigned int)*line));
        nvtxRangePop();

        nvtxRangePushA("Host_To_Device_Diff");
        gpuErrchk(cudaMemcpy(D_u, H_u, sizeof(int)*line, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(D_v, H_v, sizeof(int)*line, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(D_w, H_w, sizeof(unsigned int)*line, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(D_diff_edges, H_diff_edges, sizeof(int)*line, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(D_diff_offset, H_diff_offset, sizeof(int)*N, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(D_diff_weight, H_diff_weight, sizeof(unsigned int)*line, cudaMemcpyHostToDevice));
        nvtxRangePop();

        timer_dyn_total.begin();
        gpuErrchk(cudaMemset(D_next_queue, 0, sizeof(int)*N));
        timer_dyn_kernels.begin();

        nvtxRangePushA("Dynamic_Updating_Kernel");
        Updating<<<(line+numThreads-1)/numThreads, numThreads>>>(
            D_u, D_v, D_w, D_hx, D_next_queue, N, line);
        nvtxRangePop();

        nvtxRangePushA("Dynamic_Compact_Init");
        int H_zero = 0;
        gpuErrchk(cudaMemcpy(D_frontier_size, &H_zero, sizeof(int), cudaMemcpyHostToDevice));
        compact_frontier<<<nBlocks, numThreads>>>(D_next_queue, D_frontier, D_frontier_size, N);
        gpuErrchk(cudaMemcpy(&H_frontier_size, D_frontier_size, sizeof(int), cudaMemcpyDeviceToHost));
        nvtxRangePop();

        nvtxRangePushA("Dynamic_AStar_Execution");
        while (H_frontier_size > 0) {
            nvtxRangePushA("Dynamic_AStar_Iteration");

            nvtxRangePushA("Find_Min_Kernel");
            gpuErrchk(cudaMemcpy(D_min_f, &INF_VAL, sizeof(unsigned int), cudaMemcpyHostToDevice));
            int minBlocks = (H_frontier_size + numThreads - 1) / numThreads;
            find_min_f<<<minBlocks, numThreads>>>(D_frontier, H_frontier_size, D_min_f);
            gpuErrchk(cudaMemcpy(&H_min_f, D_min_f, sizeof(unsigned int), cudaMemcpyDeviceToHost));
            nvtxRangePop();

            nvtxRangePushA("Check_Destination");
            getCx<<<1, 1>>>(endNode, D_dest_cost);
            gpuErrchk(cudaMemcpy(H_dest_cost, D_dest_cost, sizeof(int), cudaMemcpyDeviceToHost));
            if (H_min_f >= (unsigned int)*H_dest_cost) {
                nvtxRangePop();
                nvtxRangePop();
                break;
            }
            nvtxRangePop();

            nvtxRangePushA("Expand_Kernel");
            unsigned int threshold = H_min_f + DELTA;
            int expBlocks = (WARP_SIZE * H_frontier_size + numThreads - 1) / numThreads;
            expand_delta<<<expBlocks, numThreads>>>(
                D_offset, D_edges, D_weight, D_hx,
                D_frontier, H_frontier_size, threshold,
                D_next_queue, N, E, 1, D_diff_offset, D_diff_edges, D_diff_weight, insertEdge);
            nvtxRangePop();

            nvtxRangePushA("Compact_Kernel");
            gpuErrchk(cudaMemcpy(D_frontier_size, &H_zero, sizeof(int), cudaMemcpyHostToDevice));
            compact_frontier<<<nBlocks, numThreads>>>(D_next_queue, D_frontier, D_frontier_size, N);
            gpuErrchk(cudaMemcpy(&H_frontier_size, D_frontier_size, sizeof(int), cudaMemcpyDeviceToHost));
            nvtxRangePop();

            nvtxRangePop();
        }
        nvtxRangePop();

        timer_dyn_kernels.end();
        getCx<<<1, 1>>>(endNode, D_dest_cost);
        gpuErrchk(cudaMemcpy(H_dest_cost, D_dest_cost, sizeof(int), cudaMemcpyDeviceToHost));
        timer_dyn_total.end();

        printf("=== DYNAMIC UPDATE (batch %d, %d edges) ===\n", batch, line);
        printf("  GPU kernels only:               %.3f ms\n", timer_dyn_kernels.ms());
        printf("  GPU total (kernels + PCIe):     %.3f ms\n", timer_dyn_total.ms());
        printf("  Cost after update: %d\n\n", *H_dest_cost);

        nvtxRangePushA("Free_Batch_Memory");
        gpuErrchk(cudaFree(D_u)); gpuErrchk(cudaFree(D_v)); gpuErrchk(cudaFree(D_w));
        gpuErrchk(cudaFree(D_diff_edges));
        gpuErrchk(cudaFree(D_diff_offset));
        gpuErrchk(cudaFree(D_diff_weight));
        free(H_u); free(H_v); free(H_w);
        free(H_diff_edges); free(H_diff_offset); free(H_diff_weight);
        nvtxRangePop();

        nvtxRangePop();
    }
    nvtxRangePop();
    fclose(fdiff);

    auto wall_end = chrono::high_resolution_clock::now();
    double wall_total_ms = chrono::duration<double, milli>(wall_end - wall_start).count();
    printf("=== TOTAL WALL CLOCK (IO + all phases): %.3f ms ===\n", wall_total_ms);

    timer_static_kernels.destroy();
    timer_static_total.destroy();
    timer_dyn_kernels.destroy();
    timer_dyn_total.destroy();

    nvtxRangePop();
    return 0;
}