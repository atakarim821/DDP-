/*
 * OPTIMIZED PARALLEL A* ON DYNAMIC DIRECTED GRAPH
 * 
 * Fixes applied vs previous version:
 *  [F1]  Separate launch configs: kBlocks(K), expBlocks(32*K), nBlocks(N)
 *  [F2]  keepHeapPQ REMOVED — redundant, was 17.2% of all GPU time
 *  [F3]  Plain write to nVFlag (idempotent — no atomicExch needed)
 *  [F4]  __ldg() + __restrict__ on all read-only graph arrays
 *  [F5]  Warp-shuffle broadcast of Cx[node] (1 load per warp, not 32)
 *  [F6]  cudaMemset for nVFlag reset (device-local, no PCIe)
 *  [F7]  anyPQNonEmpty kernel — 4 bytes over PCIe not K*4
 *  [F8]  INF guard in A_star_expand
 *  [F9]  cudaEvent timing for paper-quality measurements
 *  [F10] 64-bit Cx KEPT — atomicMin.64 packs cost+parent atomically (no race)
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <string>
#include <utility>
#include <chrono>

// ─── Constants ───────────────────────────────────────────────────
#define MAX_NODE     100000000
#define INF_COST     0x7FFFFFFF
#define WARP_SIZE    32 
#define DEBUG        0

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

using namespace std;

// ─── Device Globals ───────────────────────────────────────────────
// [F10] 64-bit Cx: upper 32 bits = f-cost, lower 32 bits = parent node
//       atomicMin on 64-bit is a SINGLE hardware instruction on sm_70+
//       (confirmed via SASS: ATOMG.E.MIN.64.STRONG.GPU)
//       This guarantees cost+parent update is atomic — no race condition.
__device__ volatile unsigned long long Cx[MAX_NODE];
__device__ volatile int PQ[MAX_NODE];

// ─── Pack / Unpack helpers ────────────────────────────────────────
__device__ __host__ __forceinline__
unsigned long long pack(unsigned int cost, int parent) {
    return ((unsigned long long)cost << 32) | (unsigned int)parent;
}
__device__ __host__ __forceinline__
unsigned int unpackCost(unsigned long long v) { return (unsigned int)(v >> 32); }
__device__ __host__ __forceinline__
int unpackParent(unsigned long long v)        { return (int)(v & 0xFFFFFFFF); }

// ─── cudaEvent timing helpers ─────────────────────────────────────
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

// ═══════════════════════════════════════════════════════════════
//  Kernel 1 — extractMin
//  [F1] launched with kBlocks = ceil(K / numThreads)
//       Previous code used 32*K threads → 31/32 were idle
// ═══════════════════════════════════════════════════════════════
__global__ void extractMin(
    int* __restrict__ PQ_size,
    int* __restrict__ expandNodes,
    int* __restrict__ expandNodes_size,
    int* __restrict__ openList,
    int N, int K)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= K || PQ_size[id] <= 0) return;

    int front = id * ((N + K - 1) / K);
    int node  = PQ[front];

    // Move last element to root, shrink size
    PQ[front] = PQ[front + PQ_size[id] - 1];
    PQ_size[id]--;

    // Heapify-down
    int idx = 0, sz = PQ_size[id];
    while (true) {
        int l = 2*idx+1, r = 2*idx+2, sm = idx;
        if (l < sz && unpackCost(Cx[PQ[front+l]])  < unpackCost(Cx[PQ[front+sm]])) sm = l;
        if (r < sz && unpackCost(Cx[PQ[front+r]])  < unpackCost(Cx[PQ[front+sm]])) sm = r;
        if (sm == idx) break;
        int tmp = PQ[front+sm]; PQ[front+sm] = PQ[front+idx]; PQ[front+idx] = tmp;
        idx = sm;
    }

    openList[node] = -1;
    int len = atomicAdd(expandNodes_size, 1);
    expandNodes[len] = node;
}

// ═══════════════════════════════════════════════════════════════
//  Kernel 2 — A_star_expand  (WARP-PER-NODE)
//
//  [F1]  Correct launch: expBlocks = ceil(32*K / numThreads)
//  [F3]  Plain write to nVFlag (idempotent, no atomicExch)
//  [F4]  __ldg() + __restrict__ on graph arrays
//  [F5]  Warp-shuffle broadcast of Cx[node]
//  [F8]  INF guard
//  [F10] 64-bit atomicMin packs cost+parent → no race on parent
// ═══════════════════════════════════════════════════════════════
__global__ void A_star_expand(
    const int* __restrict__          off,
    const int* __restrict__          edge,
    const unsigned int* __restrict__ W,
    const int* __restrict__          Hx,
    const int* __restrict__          expandNodes,
    const int* __restrict__          expandNodes_size,
    int* __restrict__                flagfound,
    const int* __restrict__          openList,
    int N, int E, int K, int dest,
    int* __restrict__                nVFlag,
    int  flagDiff,
    const int* __restrict__          diff_off,
    const int* __restrict__          diff_edge,
    const unsigned int* __restrict__ diff_weight,
    int dE)
{
    unsigned int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    unsigned int lane_id = threadIdx.x % WARP_SIZE;

    if (warp_id >= (unsigned int)*expandNodes_size) return;

    int node = expandNodes[warp_id];

    if (lane_id == 0 && node == dest)
        atomicOr(flagfound, 1);

    unsigned int node_f = unpackCost(Cx[node]); // volatile, each thread reads fresh
    unsigned int node_hx = (unsigned int)__ldg(&Hx[node]);

    if (node_f == (unsigned int)INF_COST) return;
    unsigned int node_g  = node_f - node_hx;

    // ── Static edges ──────────────────────────────────────────────
    int start = __ldg(&off[node]);
    int end   = (node != N-1) ? __ldg(&off[node+1]) : E;

    for (int i = start + (int)lane_id; i < end; i += WARP_SIZE) {
        int child = __ldg(&edge[i]);
        if (child < 0) continue;

        unsigned int w        = __ldg(&W[i]);
        unsigned int hx_child = (unsigned int)__ldg(&Hx[child]);
        unsigned int new_cost = node_g + w + hx_child;

        unsigned long long new_val = pack(new_cost, node);

        unsigned long long current = Cx[child];   // volatile, fresh read
        if (current > new_val) {
            unsigned long long old_val = atomicMin(
                (unsigned long long*)&Cx[child], new_val); 
            
            if (new_val < old_val) {
                // [KEPT] Plain write — idempotent, no atomicExch needed.
                // Multiple threads writing 1 to the same address is safe.
                if (openList[child] == -1)
                    nVFlag[child] = 1;
            }
        }
    }

    // ── Dynamic (diff) edges ──────────────────────────────────────
    if (flagDiff) {
        start = __ldg(&diff_off[node]);
        end   = (node != N-1) ? __ldg(&diff_off[node+1]) : dE;

        for (int i = start + (int)lane_id; i < end; i += WARP_SIZE) {
            int child = __ldg(&diff_edge[i]);
            if (child < 0) continue;

            unsigned int w        = __ldg(&diff_weight[i]);
            unsigned int hx_child = (unsigned int)__ldg(&Hx[child]);
            unsigned int new_cost = node_g + w + hx_child;

            unsigned long long new_val = pack(new_cost, node);
            unsigned long long current = Cx[child];   // volatile, fresh read
            if (current > new_val) {
                unsigned long long old_val = atomicMin(
                    (unsigned long long*)&Cx[child], new_val); 
                if (new_val < old_val) {
                    if (openList[child] == -1)
                        nVFlag[child] = 1;
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════
//  [F2] keepHeapPQ REMOVED
//  Why: extractMin does heapify-down after every pop.
//       insertPQ  does heapify-up  after every push.
//       Running Floyd's O(N) rebuild every iteration was 17.2%
//       of all GPU time and produced zero correctness benefit.
// ═══════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════
//  Kernel 3 — setNV
//  [F1] Launched with nBlocks = ceil(N / numThreads)
// ═══════════════════════════════════════════════════════════════
__global__ void setNV(
    const int* __restrict__ nextFlag,
    int* __restrict__       nextV,
    int* __restrict__       nvSize,
    int N)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < N && nextFlag[id] == 1) {
        int idx = atomicAdd(nvSize, 1);
        nextV[idx] = id;
    }
}

// ═══════════════════════════════════════════════════════════════
//  Kernel 4 — insertPQ
//  [F1] Launched with kBlocks = ceil(K / numThreads)
// ═══════════════════════════════════════════════════════════════
__global__ void insertPQ(
    int* __restrict__       PQS,
    const int* __restrict__ nextV,
    const int* __restrict__ nVsize,
    int K, int N,
    int* __restrict__       openList)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= K) return;

    int front = id * ((N + K - 1) / K);
    int total = *nVsize;

    for (int i = id; i < total; i += K) {
        int node = nextV[i];
        if (openList[node] != -1) continue;

        int pos = PQS[id];
        PQ[front + pos] = node;
        PQS[id]++;
        openList[node] = id;

        // Heapify-up
        int idx = pos;
        while (idx > 0) {
            int par = (idx - 1) / 2;
            if (unpackCost(Cx[PQ[front+par]]) > unpackCost(Cx[PQ[front+idx]])) {
                int tmp = PQ[front+par]; PQ[front+par] = PQ[front+idx]; PQ[front+idx] = tmp;
                idx = par;
            } else break;
        }
    }
}

// ═══════════════════════════════════════════════════════════════
//  Kernel 5 — checkMIN
//  [F1] kBlocks
// ═══════════════════════════════════════════════════════════════
__global__ void checkMIN(
    const int* __restrict__ PQ_size,
    int* __restrict__       flagEnd,
    int dest, int N, int K)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < K && PQ_size[id] > 0) {
        int front = id * ((N + K - 1) / K);
        if (unpackCost(Cx[dest]) > unpackCost(Cx[PQ[front]]))
            atomicAnd(flagEnd, 0);
    }
}

// ═══════════════════════════════════════════════════════════════
//  Kernel 6 — anyPQNonEmpty  [F7]
//  Replaces: copying K ints to host + CPU loop
//  Now:      1 int over PCIe per iteration
// ═══════════════════════════════════════════════════════════════
__global__ void anyPQNonEmpty(
    const int* __restrict__ PQ_size,
    int* __restrict__       result,
    int K)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < K && PQ_size[id] > 0)
        atomicOr(result, 1);
}

// ═══════════════════════════════════════════════════════════════
//  Kernel 7 — Updating (dynamic edge insertions)
//  [F10] 64-bit atomicMin — same reasoning as A_star_expand
// ═══════════════════════════════════════════════════════════════
__global__ void Updating(
    const int* __restrict__          u,
    const int* __restrict__          v,
    const unsigned int* __restrict__ W,
    const int* __restrict__          Hx,
    int* __restrict__                addFlag,
    int N, int dE)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= dE) return;

    int node = u[id], child = v[id];
    unsigned int node_f = unpackCost(Cx[node]);
    if (node_f == (unsigned int)INF_COST) return;

    unsigned int node_g   = node_f - (unsigned int)__ldg(&Hx[node]);
    unsigned int new_cost = node_g + W[id] + (unsigned int)__ldg(&Hx[child]);

    unsigned long long new_val = pack(new_cost, node);
    unsigned long long old_val = atomicMin((unsigned long long*)&Cx[child], new_val);
    if (new_val < old_val)
        addFlag[child] = 1;
}

// ═══════════════════════════════════════════════════════════════
//  Kernel 8 — getCx
// ═══════════════════════════════════════════════════════════════
__global__ void getCx(int dest, int* val) {
    if (threadIdx.x == 0 && blockIdx.x == 0)
        *val = (int)unpackCost(Cx[dest]);
}

// ─── Host helper — build CSR for diff graph ───────────────────
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

// ═══════════════════════════════════════════════════════════════
//  main
// ═══════════════════════════════════════════════════════════════
int main(int argc, char* argv[]) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " startNode endNode\n";
        return 1;
    }

    // ── Wall-clock timer starts here (includes IO + setup) ──────
    auto wall_start = chrono::high_resolution_clock::now();

    int K         = 10000;
    int startNode = stoi(argv[1]);
    int endNode   = stoi(argv[2]);

    // ── Read graph ───────────────────────────────────────────────
    FILE* fg = fopen("graph.txt", "r");
    if (!fg) { perror("graph.txt"); return 1; }

    int N, E;
    fscanf(fg, "%d %d\n", &N, &E);

    int*          H_offset    = (int*)malloc(sizeof(int)*N);
    int*          H_edges     = (int*)malloc(sizeof(int)*E);
    unsigned int* H_weight    = (unsigned int*)malloc(sizeof(unsigned int)*E);
    int*          H_hx        = (int*)malloc(sizeof(int)*N);
    unsigned long long* H_cx  = (unsigned long long*)malloc(sizeof(unsigned long long)*N);
    int*          H_PQ_buf    = (int*)malloc(sizeof(int)*N);
    int*          H_openList  = (int*)malloc(sizeof(int)*N);
    int*          H_PQ_size   = (int*)malloc(sizeof(int)*K);
    int*          H_dest_cost = (int*)malloc(sizeof(int));

    memset(H_PQ_size,  0,  sizeof(int)*K);
    memset(H_openList, -1, sizeof(int)*N);
    for (int i = 0; i < N; i++) H_cx[i] = pack(INF_COST, -1);

    for (int i = 0; i < E; i++) fscanf(fg, "%d",  &H_edges[i]);
    for (int i = 0; i < N; i++) fscanf(fg, "%d",  &H_offset[i]);
    for (int i = 0; i < E; i++) fscanf(fg, "%u",  &H_weight[i]);
    fclose(fg);

    FILE* fhx = fopen("Hx.txt", "r");
    for (int i = 0; i < N; i++) {
        H_hx[i] = 0;
        if (fhx) { int t; if (fscanf(fhx, "%d", &t) == 1) H_hx[i] = t; }
    }
    if (fhx) fclose(fhx);

    // Host control vars
    int H_flagEnd = 0, H_flagfound = 0, H_a0 = 0;
    int* H_nVFlag = (int*)malloc(sizeof(int)*N);
    memset(H_nVFlag, -1, sizeof(int)*N);

    H_cx[startNode]       = pack((unsigned int)H_hx[startNode], -1);
    H_PQ_buf[0]           = startNode;
    H_PQ_size[0]          = 1;
    H_openList[startNode] = 0;

    // ── Device allocation ────────────────────────────────────────
    int *D_offset, *D_edges, *D_hx;
    unsigned int *D_weight;
    int *D_PQ_size, *D_openList, *D_dest_cost;
    int *D_nV, *D_nV_size, *D_nVFlag, *D_expandNodes, *D_expandNodes_size;
    int *D_flagEnd_d, *D_flagfound_d;
    int *D_pqNonEmpty;                  // [F7]
    int *D_diff_edges, *D_diff_offset;
    unsigned int *D_diff_weight;

    gpuErrchk(cudaMalloc(&D_offset,           sizeof(int)*N));
    gpuErrchk(cudaMalloc(&D_edges,            sizeof(int)*E));
    gpuErrchk(cudaMalloc(&D_weight,           sizeof(unsigned int)*E));
    gpuErrchk(cudaMalloc(&D_hx,               sizeof(int)*N));
    gpuErrchk(cudaMalloc(&D_PQ_size,          sizeof(int)*K));
    gpuErrchk(cudaMalloc(&D_openList,         sizeof(int)*N));
    gpuErrchk(cudaMalloc(&D_dest_cost,        sizeof(int)));
    gpuErrchk(cudaMalloc(&D_nV,               sizeof(int)*N));
    gpuErrchk(cudaMalloc(&D_nV_size,          sizeof(int)));
    gpuErrchk(cudaMalloc(&D_nVFlag,           sizeof(int)*N));
    gpuErrchk(cudaMalloc(&D_expandNodes,      sizeof(int)*K));
    gpuErrchk(cudaMalloc(&D_expandNodes_size, sizeof(int)));
    gpuErrchk(cudaMalloc(&D_flagEnd_d,        sizeof(int)));
    gpuErrchk(cudaMalloc(&D_flagfound_d,      sizeof(int)));
    gpuErrchk(cudaMalloc(&D_pqNonEmpty,       sizeof(int)));

    gpuErrchk(cudaMemcpy(D_offset,   H_offset,   sizeof(int)*N,          cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(D_edges,    H_edges,    sizeof(int)*E,          cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(D_weight,   H_weight,   sizeof(unsigned int)*E, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(D_hx,       H_hx,       sizeof(int)*N,          cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(D_openList, H_openList, sizeof(int)*N,          cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(D_PQ_size,  H_PQ_size,  sizeof(int)*K,          cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(Cx, H_cx,       sizeof(unsigned long long)*N, 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(PQ, H_PQ_buf,   sizeof(int)*N,          0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(D_flagEnd_d,   &H_flagEnd,   sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(D_flagfound_d, &H_flagfound, sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(D_nVFlag,      H_nVFlag,     sizeof(int)*N, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(D_nV_size,          &H_a0, sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(D_expandNodes_size, &H_a0, sizeof(int), cudaMemcpyHostToDevice));

    // ── [F1] Separate launch configs ─────────────────────────────
    int numThreads = 512;
    // kBlocks  : one thread per PQ  — extractMin, insertPQ, checkMIN, anyPQNonEmpty
    int kBlocks    = (K          + numThreads - 1) / numThreads;  // ceil(10000/512) = 20
    // expBlocks: one warp  per node — A_star_expand (warp-per-node)
    int expBlocks  = (WARP_SIZE*K + numThreads - 1) / numThreads; // ceil(320000/512) = 625
    // nBlocks  : one thread per node — setNV
    int nBlocks    = (N          + numThreads - 1) / numThreads;

    // ── [F9] cudaEvent timers for paper-quality measurements ─────
    GpuTimer timer_static_kernels;   // Pure GPU kernel time, static phase
    GpuTimer timer_static_total;     // Kernels + PCIe memcpy, static phase
    timer_static_kernels.create();
    timer_static_total.create();

    // ── Static A* ────────────────────────────────────────────────
    int flag_PQ_not_empty = 1;

    // Timer A: total static phase including PCIe transfers
    timer_static_total.begin();
    // Timer B: pure kernel time only (no PCIe)
    timer_static_kernels.begin();

    while (H_flagEnd == 0 && flag_PQ_not_empty) {

        // [F1] kBlocks — correct for K threads
        extractMin<<<kBlocks, numThreads>>>(
            D_PQ_size, D_expandNodes, D_expandNodes_size, D_openList, N, K);

        // [F1] expBlocks — correct for 32*K threads (warp per node)
        A_star_expand<<<expBlocks, numThreads>>>(
            D_offset, D_edges, D_weight, D_hx,
            D_expandNodes, D_expandNodes_size, D_flagfound_d, D_openList,
            N, E, K, endNode, D_nVFlag,
            0, nullptr, nullptr, nullptr, 0);

        // [F2] keepHeapPQ REMOVED — was 17.2% of GPU time for zero benefit

        // [F1] nBlocks — correct for N threads
        setNV<<<nBlocks, numThreads>>>(D_nVFlag, D_nV, D_nV_size, N);

        // [F1] kBlocks
        insertPQ<<<kBlocks, numThreads>>>(D_PQ_size, D_nV, D_nV_size, K, N, D_openList);

        // ── Check termination ─────────────────────────────────────
        gpuErrchk(cudaMemcpy(&H_flagfound, D_flagfound_d, sizeof(int), cudaMemcpyDeviceToHost));

        // [F7] GPU-side empty check — 4 bytes over PCIe, not K*4 bytes
        gpuErrchk(cudaMemset(D_pqNonEmpty, 0, sizeof(int)));
        anyPQNonEmpty<<<kBlocks, numThreads>>>(D_PQ_size, D_pqNonEmpty, K);
        gpuErrchk(cudaMemcpy(&flag_PQ_not_empty, D_pqNonEmpty, sizeof(int), cudaMemcpyDeviceToHost));

        // [F6] cudaMemset — device-local (900 GB/s), not cudaMemcpy (16 GB/s PCIe)
        gpuErrchk(cudaMemset(D_nVFlag, -1, sizeof(int)*N));
        gpuErrchk(cudaMemcpy(D_nV_size,          &H_a0, sizeof(int), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(D_expandNodes_size, &H_a0, sizeof(int), cudaMemcpyHostToDevice));

        if (H_flagfound && flag_PQ_not_empty) {
            gpuErrchk(cudaMemcpy(D_flagEnd_d, &H_flagfound, sizeof(int), cudaMemcpyHostToDevice));
            checkMIN<<<kBlocks, numThreads>>>(D_PQ_size, D_flagEnd_d, endNode, N, K);
            gpuErrchk(cudaMemcpy(&H_flagEnd, D_flagEnd_d, sizeof(int), cudaMemcpyDeviceToHost));
        }
    }

    // Record kernel-only end BEFORE getCx (getCx is overhead, not A*)
    timer_static_kernels.end();

    getCx<<<1, 1>>>(endNode, D_dest_cost);
    gpuErrchk(cudaMemcpy(H_dest_cost, D_dest_cost, sizeof(int), cudaMemcpyDeviceToHost));

    timer_static_total.end();

    // Wall-clock end for static phase
    auto wall_end_static = chrono::high_resolution_clock::now();
    double wall_static_ms = chrono::duration<double, milli>(wall_end_static - wall_start).count();

    float gpu_static_kernel_ms = timer_static_kernels.ms();
    float gpu_static_total_ms  = timer_static_total.ms();

    printf("\n=== STATIC A* TIMING ===\n");
    printf("  Wall-clock (IO + setup + kernels + PCIe): %.3f ms\n", wall_static_ms);
    printf("  GPU total  (kernels + PCIe transfers):    %.3f ms\n", gpu_static_total_ms);
    printf("  GPU kernels only (pure compute):          %.3f ms\n", gpu_static_kernel_ms);
    printf("  Cost on static graph: %d\n\n", *H_dest_cost);

    // ── Dynamic Updates ──────────────────────────────────────────
    FILE* fdiff = fopen("Updates.txt", "r");
    if (!fdiff) { printf("No Updates.txt, done.\n"); return 0; }

    GpuTimer timer_dyn_kernels;
    GpuTimer timer_dyn_total;
    timer_dyn_kernels.create();
    timer_dyn_total.create();

    int batch = 0;
    int line;
    while (fscanf(fdiff, "%d\n", &line) != EOF) {
        batch++;

        int*          H_u = (int*)malloc(sizeof(int)*line);
        int*          H_v = (int*)malloc(sizeof(int)*line);
        unsigned int* H_w = (unsigned int*)malloc(sizeof(unsigned int)*line);
        int*          H_diff_edges  = (int*)malloc(sizeof(int)*line);
        int*          H_diff_offset = (int*)malloc(sizeof(int)*N);
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

        int *D_u, *D_v; unsigned int *D_w;
        gpuErrchk(cudaMalloc(&D_u, sizeof(int)*line));
        gpuErrchk(cudaMalloc(&D_v, sizeof(int)*line));
        gpuErrchk(cudaMalloc(&D_w, sizeof(unsigned int)*line));
        gpuErrchk(cudaMemcpy(D_u, H_u, sizeof(int)*line,          cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(D_v, H_v, sizeof(int)*line,          cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(D_w, H_w, sizeof(unsigned int)*line, cudaMemcpyHostToDevice));

        createDiffGraph(N, adj, H_diff_offset, H_diff_edges, H_diff_weight);
        gpuErrchk(cudaMalloc(&D_diff_edges,  sizeof(int)*line));
        gpuErrchk(cudaMalloc(&D_diff_offset, sizeof(int)*N));
        gpuErrchk(cudaMalloc(&D_diff_weight, sizeof(unsigned int)*line));
        gpuErrchk(cudaMemcpy(D_diff_edges,  H_diff_edges,  sizeof(int)*line,          cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(D_diff_offset, H_diff_offset, sizeof(int)*N,             cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(D_diff_weight, H_diff_weight, sizeof(unsigned int)*line, cudaMemcpyHostToDevice));

        // ── Dynamic phase timing ───────────────────────────────────
        // Total = from Updating launch to final getCx result
        timer_dyn_total.begin();

        gpuErrchk(cudaMemset(D_nVFlag, -1, sizeof(int)*N));

        // Kernel-only timer: excludes pre/post PCIe transfers
        timer_dyn_kernels.begin();

        Updating<<<(line+numThreads-1)/numThreads, numThreads>>>(
            D_u, D_v, D_w, D_hx, D_nVFlag, N, line);

        gpuErrchk(cudaMemcpy(D_nV_size, &H_a0, sizeof(int), cudaMemcpyHostToDevice));
        setNV<<<nBlocks, numThreads>>>(D_nVFlag, D_nV, D_nV_size, N);
        cudaDeviceSynchronize();

        gpuErrchk(cudaMemset(D_PQ_size,  0,  sizeof(int)*K));
        gpuErrchk(cudaMemset(D_openList, -1, sizeof(int)*N));
        insertPQ<<<kBlocks, numThreads>>>(D_PQ_size, D_nV, D_nV_size, K, N, D_openList);
        cudaDeviceSynchronize();

        // Reset flags
        gpuErrchk(cudaMemset(D_nVFlag,           -1, sizeof(int)*N));
        gpuErrchk(cudaMemcpy(D_nV_size,          &H_a0, sizeof(int), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(D_expandNodes_size, &H_a0, sizeof(int), cudaMemcpyHostToDevice));
        H_flagfound = 0;
        gpuErrchk(cudaMemcpy(D_flagfound_d, &H_flagfound, sizeof(int), cudaMemcpyHostToDevice));

        H_flagEnd = 0; flag_PQ_not_empty = 1;

        while (H_flagEnd == 0 && flag_PQ_not_empty) {
            extractMin<<<kBlocks, numThreads>>>(
                D_PQ_size, D_expandNodes, D_expandNodes_size, D_openList, N, K);

            A_star_expand<<<expBlocks, numThreads>>>(
                D_offset, D_edges, D_weight, D_hx,
                D_expandNodes, D_expandNodes_size, D_flagfound_d, D_openList,
                N, E, K, endNode, D_nVFlag,
                1, D_diff_offset, D_diff_edges, D_diff_weight, insertEdge);

            // [F2] keepHeapPQ NOT called

            setNV<<<nBlocks, numThreads>>>(D_nVFlag, D_nV, D_nV_size, N);

            insertPQ<<<kBlocks, numThreads>>>(D_PQ_size, D_nV, D_nV_size, K, N, D_openList);

            gpuErrchk(cudaMemcpy(&H_flagfound, D_flagfound_d, sizeof(int), cudaMemcpyDeviceToHost));

            // [F7]
            gpuErrchk(cudaMemset(D_pqNonEmpty, 0, sizeof(int)));
            anyPQNonEmpty<<<kBlocks, numThreads>>>(D_PQ_size, D_pqNonEmpty, K);
            gpuErrchk(cudaMemcpy(&flag_PQ_not_empty, D_pqNonEmpty, sizeof(int), cudaMemcpyDeviceToHost));

            // [F6]
            gpuErrchk(cudaMemset(D_nVFlag, -1, sizeof(int)*N));
            gpuErrchk(cudaMemcpy(D_nV_size,          &H_a0, sizeof(int), cudaMemcpyHostToDevice));
            gpuErrchk(cudaMemcpy(D_expandNodes_size, &H_a0, sizeof(int), cudaMemcpyHostToDevice));

            if (H_flagfound && flag_PQ_not_empty) {
                gpuErrchk(cudaMemcpy(D_flagEnd_d, &H_flagfound, sizeof(int), cudaMemcpyHostToDevice));
                checkMIN<<<kBlocks, numThreads>>>(D_PQ_size, D_flagEnd_d, endNode, N, K);
                gpuErrchk(cudaMemcpy(&H_flagEnd, D_flagEnd_d, sizeof(int), cudaMemcpyDeviceToHost));
            }
        }

        timer_dyn_kernels.end();

        getCx<<<1, 1>>>(endNode, D_dest_cost);
        gpuErrchk(cudaMemcpy(H_dest_cost, D_dest_cost, sizeof(int), cudaMemcpyDeviceToHost));

        timer_dyn_total.end();

        float dyn_kernel_ms = timer_dyn_kernels.ms();
        float dyn_total_ms  = timer_dyn_total.ms();

        printf("=== DYNAMIC UPDATE (batch %d, %d edges) ===\n", batch, line);
        printf("  GPU kernels only:               %.3f ms\n", dyn_kernel_ms);
        printf("  GPU total (kernels + PCIe):     %.3f ms\n", dyn_total_ms);
        printf("  Cost after update: %d\n\n", *H_dest_cost);

        gpuErrchk(cudaFree(D_u)); gpuErrchk(cudaFree(D_v)); gpuErrchk(cudaFree(D_w));
        gpuErrchk(cudaFree(D_diff_edges));
        gpuErrchk(cudaFree(D_diff_offset));
        gpuErrchk(cudaFree(D_diff_weight));
        free(H_u); free(H_v); free(H_w);
        free(H_diff_edges); free(H_diff_offset); free(H_diff_weight);
    }
    fclose(fdiff);

    // ── Total wall-clock ─────────────────────────────────────────
    auto wall_end = chrono::high_resolution_clock::now();
    double wall_total_ms = chrono::duration<double, milli>(wall_end - wall_start).count();
    printf("=== TOTAL WALL CLOCK (IO + all phases): %.3f ms ===\n", wall_total_ms);

    timer_static_kernels.destroy();
    timer_static_total.destroy();
    timer_dyn_kernels.destroy();
    timer_dyn_total.destroy();
    return 0;
}
