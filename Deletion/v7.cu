// v7 file
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define  MAX_NODE  100000000
#define  DEBUG 0 

#define INF_COST 0x7FFFFFFF 

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

using namespace std;

__device__ volatile unsigned long long Cx[MAX_NODE];


__device__ __host__ inline unsigned long long pack(unsigned int cost, int parent) {
    return ((unsigned long long)cost << 32) | (unsigned int)parent;
}

__device__ __host__ inline unsigned int unpackCost(unsigned long long val) {
    return (unsigned int)(val >> 32);
}

__device__ __host__ inline int unpackParent(unsigned long long val) {
    return (int)(val & 0xFFFFFFFF);
}

// ── STATIC PHASE KERNELS (unchanged from v6) ────────────────────────

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
    unsigned int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    unsigned int lane_id = threadIdx.x % 32;

    if (warp_id >= (unsigned int)frontier_size) return;

    int node = frontier[warp_id];
    unsigned int node_f = unpackCost(Cx[node]);

    if (node_f > threshold) {
        if (lane_id == 0) next_queue[node] = 1;
        return;
    }

    unsigned int node_hx = (unsigned int)Hx[node];
    unsigned int node_g  = node_f - node_hx;

    int start = off[node];
    int end   = (node != N-1) ? off[node+1] : E;

    for (int i = start + (int)lane_id; i < end; i += 32) {
        int child = edge[i];
        if (child < 0) continue;

        unsigned int w = W[i];
        unsigned int hx_child = (unsigned int)Hx[child];
        unsigned int new_cost = node_g + w + hx_child;
        unsigned long long new_val = pack(new_cost, node);

        unsigned long long old_val = atomicMin((unsigned long long*)&Cx[child], new_val);
        if (new_val < old_val) {
            next_queue[child] = 1;
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

// v6 OPT (kept): Binary-search edge marking — 1 thread per deletion, O(log degree).
__global__ void markDeletedBinarySearch(
    const int* __restrict__ u, const int* __restrict__ v,
    const int* __restrict__ offset, const int* __restrict__ edges,
    unsigned int* __restrict__ weight, int N, int E, int dE)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= dE) return;

    int from = u[id];
    int to   = v[id];
    if (from < 0 || from >= N || to < 0 || to >= N) return;

    int lo = offset[from];
    int hi = (from == N - 1) ? E : offset[from + 1];
    while (lo < hi) {
        int mid = lo + ((hi - lo) >> 1);
        if (edges[mid] < to) lo = mid + 1;
        else hi = mid;
    }
    if (lo < ((from == N - 1) ? E : offset[from + 1]) && edges[lo] == to)
        weight[lo] = INT_MAX;
}

__global__ void markAsAffected(int* u, int *v, int *affected, int N, int dE)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < dE)
    {
        int from = u[id];
        int to = v[id];
        if (from < 0 || from >= N || to < 0 || to >= N) return;
        
        if (unpackParent(Cx[to]) == from)
        {
            affected[to] = 1; 
            Cx[to] = pack(INF_COST, -1); 
        }
    }
}

// ── v7 FUSION 2: Persistent descendant marking (cooperative groups) ──
// Single cooperative kernel replaces InitializeWorklist + host loop of markDescendantLockFree.
// Phase 0 builds the initial worklist from affected[], then iterates.

__global__ void persistentDescendantMarking(
    const int* __restrict__ edges, const int* __restrict__ offset,
    int* __restrict__ worklist_A, int* __restrict__ size_A,
    int* __restrict__ worklist_B, int* __restrict__ size_B,
    int* __restrict__ affected, int N, int E)
{
    cg::grid_group grid = cg::this_grid();
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = gridDim.x * blockDim.x;
    unsigned int warp_id = tid / 32;
    unsigned int lane_id = tid % 32;
    unsigned int total_warps = stride / 32;

    // Phase 0: Build initial worklist from affected[]
    for (unsigned int id = tid; id < (unsigned int)N; id += stride) {
        if (affected[id] == 1) {
            int pos = atomicAdd(size_A, 1);
            worklist_A[pos] = id;
        }
    }

    grid.sync();

    int *curr_wl = worklist_A, *next_wl = worklist_B;
    int *curr_sz = size_A,     *next_sz = size_B;

    while (true) {
        int ws = *curr_sz;
        if (ws == 0) break;

        for (unsigned int wi = warp_id; wi < (unsigned int)ws; wi += total_warps) {
            int node = curr_wl[wi];
            int start = offset[node];
            int end   = (node == N - 1) ? E : offset[node + 1];

            for (int i = start + (int)lane_id; i < end; i += 32) {
                int child = edges[i];
                if (child < 0 || child >= N || affected[child] == 1)
                    continue;
                if (unpackParent(Cx[child]) == node) {
                    int old = atomicExch(&affected[child], 1);
                    if (old == 0) {
                        Cx[child] = pack(INF_COST, -1);
                        int index = atomicAdd(next_sz, 1);
                        next_wl[index] = child;
                    }
                }
            }
        }

        grid.sync();

        if (tid == 0) *curr_sz = 0;
        int *tmp_wl = curr_wl; curr_wl = next_wl; next_wl = tmp_wl;
        int *tmp_sz = curr_sz; curr_sz = next_sz; next_sz = tmp_sz;

        grid.sync();
    }
}

// ── v7 FUSION 3: Fused Update-weights + Build-frontier (cooperative, 3 phases) ──

__global__ void fuseUpdateAndBuildFrontier(
    const int* __restrict__ affected,
    const int* __restrict__ r_offset, const int* __restrict__ r_edges,
    const unsigned int* __restrict__ r_weight,
    const int* __restrict__ hx,
    int* __restrict__ worklist, int* __restrict__ worklist_size,
    int* __restrict__ flag_array,
    int* __restrict__ frontier, int* __restrict__ frontier_size,
    int N, int E)
{
    cg::grid_group grid = cg::this_grid();
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = gridDim.x * blockDim.x;

    // Phase 1: Build worklist from affected nodes
    for (unsigned int id = tid; id < (unsigned int)N; id += stride) {
        if (affected[id] == 1) {
            int pos = atomicAdd(worklist_size, 1);
            worklist[pos] = id;
        }
    }
    grid.sync();

    // Phase 2: Update weights (warp-cooperative per worklist entry)
    int num_affected = *worklist_size;
    unsigned int warp_id_global = tid / 32;
    unsigned int lane_id = tid % 32;
    unsigned int total_warps = stride / 32;

    for (unsigned int wi = warp_id_global; wi < (unsigned int)num_affected; wi += total_warps) {
        int node = worklist[wi];
        int start = r_offset[node];
        int end   = (node == N - 1) ? E : r_offset[node + 1];

        unsigned int minCost = INF_COST;
        int minParent = -1;

        for (int i = start + (int)lane_id; i < end; i += 32) {
            int parentNode = r_edges[i];
            unsigned int parentCost = unpackCost(Cx[parentNode]);
            if (parentCost == INF_COST || r_weight[i] == INT_MAX)
                continue;
            unsigned int cost = (parentCost - hx[parentNode]) + r_weight[i] + hx[node];
            if (cost < minCost) {
                minCost = cost;
                minParent = parentNode;
            }
        }

        for (int off = 16; off > 0; off /= 2) {
            unsigned int other_cost = __shfl_down_sync(0xffffffff, minCost, off);
            int other_parent = __shfl_down_sync(0xffffffff, minParent, off);
            if (other_cost < minCost) {
                minCost = other_cost;
                minParent = other_parent;
            }
        }

        if (lane_id == 0) {
            Cx[node] = pack(minCost, minParent);
            if (minParent != -1) {
                flag_array[node] = 1;
            }
        }
    }
    grid.sync();

    // Phase 3: Compact flag[] → frontier
    for (unsigned int id = tid; id < (unsigned int)N; id += stride) {
        if (flag_array[id] == 1) {
            int pos = atomicAdd(frontier_size, 1);
            frontier[pos] = id;
            flag_array[id] = 0;
        }
    }
}

// ── v7 FUSION 4: Persistent Delta-stepping A* (cooperative groups) ──
// Replaces the host loop of {find_min_f + readCxCost + expand_delta_append}.
// Everything stays on-device: reduction, termination check, expansion, swap.

__global__ void persistentAStarDelta(
    const int* __restrict__ off,
    const int* __restrict__ edge,
    const unsigned int* __restrict__ W,
    const int* __restrict__ Hx,
    int* __restrict__ frontier_A, int* __restrict__ size_A,
    int* __restrict__ frontier_B, int* __restrict__ size_B,
    int* __restrict__ gen_visited,
    unsigned int* __restrict__ min_f_scratch,
    int N, int E, unsigned int DELTA_DYN, int endNode)
{
    cg::grid_group grid = cg::this_grid();

    int *curr_f = frontier_A, *next_f = frontier_B;
    int *curr_sz = size_A,    *next_sz = size_B;
    int generation = 0;

    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_threads = gridDim.x * blockDim.x;

    while (true) {
        int fsize = *curr_sz;
        if (fsize == 0) break;

        // ── Phase 1: Find min f-value (grid-wide reduction) ──
        // Reset scratch
        if (tid == 0) *min_f_scratch = INF_COST;
        grid.sync();

        unsigned int local_min = INF_COST;

        // Each thread handles one or more frontier entries
        for (unsigned int idx = tid; idx < (unsigned int)fsize; idx += total_threads) {
            unsigned int f = unpackCost(Cx[curr_f[idx]]);
            local_min = min(local_min, f);
        }

        // Warp reduction
        for (int off = 16; off > 0; off /= 2)
            local_min = min(local_min, __shfl_down_sync(0xffffffff, local_min, off));

        // Block reduction
        __shared__ unsigned int smin[32];
        unsigned int lane = threadIdx.x % 32;
        unsigned int wid  = threadIdx.x / 32;

        if (lane == 0) smin[wid] = local_min;
        __syncthreads();

        if (wid == 0) {
            local_min = (lane < (blockDim.x / 32)) ? smin[lane] : INF_COST;
            for (int off = 16; off > 0; off /= 2)
                local_min = min(local_min, __shfl_down_sync(0xffffffff, local_min, off));
            if (lane == 0 && local_min != INF_COST)
                atomicMin(min_f_scratch, local_min);
        }

        grid.sync();

        // ── Phase 2: Termination check ──
        unsigned int mf = *min_f_scratch;
        unsigned int dest_cost = unpackCost(Cx[endNode]);
        if (mf >= dest_cost) break;

        unsigned int threshold = mf + DELTA_DYN;
        generation++;

        // Reset next frontier size
        if (tid == 0) *next_sz = 0;
        grid.sync();

        // ── Phase 3: Expand (warp-per-node, with loop for large frontiers) ──
        unsigned int warp_id_global = tid / 32;
        unsigned int lane_id = tid % 32;
        unsigned int total_warps = total_threads / 32;

        for (unsigned int wi = warp_id_global; wi < (unsigned int)fsize; wi += total_warps) {
            int node = curr_f[wi];
            unsigned int node_f = unpackCost(Cx[node]);

            if (node_f > threshold) {
                // Above threshold — carry over to next frontier
                if (lane_id == 0) {
                    int old = atomicExch(&gen_visited[node], generation);
                    if (old != generation) {
                        int pos = atomicAdd(next_sz, 1);
                        next_f[pos] = node;
                    }
                }
                continue;
            }

            unsigned int node_hx = (unsigned int)Hx[node];
            unsigned int node_g  = node_f - node_hx;

            int start = off[node];
            int end   = (node != N - 1) ? off[node + 1] : E;

            for (int i = start + (int)lane_id; i < end; i += 32) {
                int child = edge[i];
                if (child < 0) continue;

                unsigned int w = W[i];
                unsigned int hx_child = (unsigned int)Hx[child];
                unsigned int new_cost = node_g + w + hx_child;
                unsigned long long new_val = pack(new_cost, node);

                unsigned long long old_val = atomicMin((unsigned long long*)&Cx[child], new_val);
                if (new_val < old_val) {
                    int old = atomicExch(&gen_visited[child], generation);
                    if (old != generation) {
                        int pos = atomicAdd(next_sz, 1);
                        next_f[pos] = child;
                    }
                }
            }
        }

        grid.sync();

        // Swap frontier buffers (all threads do this identically)
        int *tmp_f  = curr_f;  curr_f  = next_f;  next_f  = tmp_f;
        int *tmp_sz = curr_sz; curr_sz = next_sz;  next_sz = tmp_sz;
    }
}

// ── Host utilities (unchanged) ──────────────────────────────────────

void build_reverse_csr(int N, int E,
                       const int *H_offset,
                       const int *H_edges,
                       const unsigned int *H_weight,
                       int *r_H_offset,          
                       int *r_H_edges,           
                       unsigned int *r_H_weight) 
{
    int *indeg = (int *)calloc(N, sizeof(int));
    for (int u = 0; u < N; u++) {
        int start = H_offset[u];
        int end = (u == N - 1) ? E : H_offset[u + 1];
        for (int e = start; e < end; e++) {
            int v = H_edges[e];
            indeg[v]++;
        }
    }
    r_H_offset[0] = 0;
    for (int i = 1; i < N; i++)
        r_H_offset[i] = r_H_offset[i - 1] + indeg[i - 1];
    int *counter = (int *)calloc(N, sizeof(int));
    for (int u = 0; u < N; u++) {
        int start = H_offset[u];
        int end = (u == N - 1) ? E : H_offset[u + 1];
        for (int e = start; e < end; e++) {
            int v = H_edges[e];
            unsigned int w = H_weight[e];
            int pos = r_H_offset[v] + counter[v];
            r_H_edges[pos] = u; 
            r_H_weight[pos] = w;
            counter[v]++;
        }
    }
    free(indeg);
    free(counter);
}

void sortCSR(int N, int E, int* offset, int* edges, unsigned int* weight)
{
    struct EW { int edge; unsigned int wt; };
    int max_deg = 0;
    for (int u = 0; u < N; u++) {
        int end = (u == N - 1) ? E : offset[u + 1];
        max_deg = std::max(max_deg, end - offset[u]);
    }
    EW* buf = (EW*)malloc(sizeof(EW) * max_deg);
    for (int u = 0; u < N; u++) {
        int start = offset[u];
        int end = (u == N - 1) ? E : offset[u + 1];
        int deg = end - start;
        if (deg <= 1) continue;
        for (int i = 0; i < deg; i++)
            buf[i] = { edges[start + i], weight[start + i] };
        std::sort(buf, buf + deg, [](const EW& a, const EW& b) {
            return a.edge < b.edge;
        });
        for (int i = 0; i < deg; i++) {
            edges[start + i] = buf[i].edge;
            weight[start + i] = buf[i].wt;
        }
    }
    free(buf);
}

__global__ void InitializeWorklist(int *affected, int *worklist, int *worklist_size, int N)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < N) {
        if (affected[id] == 1) {
            int index = atomicAdd(worklist_size, 1);
            worklist[index] = id;
        }
    }
}

static inline unsigned int readCxCost(int node) {
    unsigned long long val;
    gpuErrchk(cudaMemcpyFromSymbol(&val, Cx,
              sizeof(unsigned long long),
              sizeof(unsigned long long) * node,
              cudaMemcpyDeviceToHost));
    return unpackCost(val);
}

// Helper: query max cooperative grid size for a kernel
static int getMaxCoopBlocks(const void* kernel, int blockSize, int sharedMem = 0) {
    int maxBlocksPerSM = 0;
    gpuErrchk(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxBlocksPerSM, kernel, blockSize, sharedMem));
    
    int deviceId;
    cudaGetDevice(&deviceId);
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, deviceId);
    
    return maxBlocksPerSM * numSMs;
}

int main(int argc, char *argv[]){

    auto start_wall_static = std::chrono::high_resolution_clock::now(); 
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " startNode endNode\n";
        return 1;
    }
    int startNode = std::stoi(argv[1]);
    int endNode = std::stoi(argv[2]);
    unsigned int DELTA = 200;
    unsigned int DELTA_DYN = 2000;  // v6 OPT: larger delta for dynamic phase

    FILE* fgraph = fopen("graph.txt","r");
    if (!fgraph) {
        std::cerr << "Failed to open graph.txt\n";
        return 1;
    }

    int N,E;
    fscanf(fgraph,"%d %d\n",&N,&E);

    if (startNode < 0 || startNode >= N || endNode < 0 || endNode >= N) {
        std::cerr << "FATAL: Start or end node out of bounds!\n";
        fclose(fgraph);
        return 1;
    }

    int* H_offset = (int*)malloc(sizeof(int)*N);
    int* H_edges  = (int*)malloc(sizeof(int)*E);
    unsigned int* H_weight = (unsigned int*)malloc(sizeof(unsigned int)*E);
    int* H_hx = (int*)malloc(sizeof(int)*N);
    unsigned long long* H_cx = (unsigned long long*)malloc(sizeof(unsigned long long)*N);
    int* H_parent = (int*)malloc(sizeof(int)*N);

    int *r_H_offset = (int *)malloc(sizeof(int) * N);
    int *r_H_edges = (int *)malloc(sizeof(int) * E);
    unsigned int *r_H_weight = (unsigned int *)malloc(sizeof(unsigned int) * E);

    memset(H_parent,-1,sizeof(int)*N);

    for(int i=0;i<N;i++)
        H_cx[i]=pack(INF_COST, -1);

    for(int i=0;i<E;i++)
        fscanf(fgraph,"%d",&H_edges[i]);
    for(int i=0;i<N;i++)
        fscanf(fgraph,"%d",&H_offset[i]);
    for(int i=0;i<E;i++)
        fscanf(fgraph,"%u",&H_weight[i]);

    sortCSR(N, E, H_offset, H_edges, H_weight);
    build_reverse_csr(N, E, H_offset, H_edges, H_weight,
                      r_H_offset, r_H_edges, r_H_weight);
    sortCSR(N, E, r_H_offset, r_H_edges, r_H_weight);

    FILE* fhx = fopen("Hx.txt","r");
    if (fhx) {
        for(int i=0;i<N;i++) H_hx[i] = 0;
        fclose(fhx);
    } else {
        for(int i=0;i<N;i++) H_hx[i] = 0;
    }
    fclose(fgraph);

    H_cx[startNode]=pack(H_hx[startNode], -1);

    // ── Device allocations ──
    int* D_offset;  int* D_edges;  unsigned int* D_weight;
    int* r_D_offset; int* r_D_edges; unsigned int* r_D_weight;
    int* D_hx; int* D_parent;
    int *D_frontier, *D_next_queue, *D_frontier_size;
    unsigned int *D_min_f;

    gpuErrchk(cudaMalloc(&D_offset, sizeof(int)*N));
    gpuErrchk(cudaMalloc(&D_edges, sizeof(int)*E));
    gpuErrchk(cudaMalloc(&D_weight, sizeof(unsigned int)*E));
    gpuErrchk(cudaMalloc(&r_D_offset, sizeof(int)*N));
    gpuErrchk(cudaMalloc(&r_D_edges, sizeof(int)*E));
    gpuErrchk(cudaMalloc(&r_D_weight, sizeof(unsigned int)*E));
    gpuErrchk(cudaMalloc(&D_hx, sizeof(int)*N));
    gpuErrchk(cudaMalloc(&D_parent, sizeof(int)*N));
    gpuErrchk(cudaMalloc(&D_frontier, sizeof(int)*N));
    gpuErrchk(cudaMalloc(&D_next_queue, sizeof(int)*N));
    gpuErrchk(cudaMalloc(&D_frontier_size, sizeof(int)));
    gpuErrchk(cudaMalloc(&D_min_f, sizeof(unsigned int)));

    gpuErrchk(cudaMemcpy(D_offset, H_offset, sizeof(int)*N, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(D_edges, H_edges, sizeof(int)*E, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(D_weight, H_weight, sizeof(unsigned int)*E, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(r_D_offset, r_H_offset, sizeof(int)*N, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(r_D_edges, r_H_edges, sizeof(int)*E, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(r_D_weight, r_H_weight, sizeof(unsigned int)*E, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(D_hx, H_hx, sizeof(int)*N, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(D_parent, H_parent, sizeof(int)*N, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(Cx, H_cx, sizeof(unsigned long long)*N, 0, cudaMemcpyHostToDevice));

    int numThreads = 512;
    int nBlocks = (N + numThreads - 1) / numThreads;

    int H_frontier_size = 1;
    gpuErrchk(cudaMemcpy(D_frontier, &startNode, sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(D_frontier_size, &H_frontier_size, sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemset(D_next_queue, 0, sizeof(int)*N));
    
    unsigned int H_min_f;
    unsigned int INF_VAL = INF_COST;

    // ── STATIC PHASE (unchanged from v6) ──────────────────────────
    cudaEvent_t t_static_start, t_static_stop;
    cudaEventCreate(&t_static_start);
    cudaEventCreate(&t_static_stop);
    cudaEventRecord(t_static_start);

    while (H_frontier_size > 0) {
        gpuErrchk(cudaMemcpy(D_min_f, &INF_VAL, sizeof(unsigned int), cudaMemcpyHostToDevice));
        int minBlocks = (H_frontier_size + numThreads - 1) / numThreads;
        find_min_f<<<minBlocks, numThreads>>>(D_frontier, H_frontier_size, D_min_f);
        gpuErrchk(cudaMemcpy(&H_min_f, D_min_f, sizeof(unsigned int), cudaMemcpyDeviceToHost));

        unsigned int dest_cost = readCxCost(endNode);
        if (H_min_f >= dest_cost) break;

        unsigned int threshold = H_min_f + DELTA;

        int expBlocks = (32 * H_frontier_size + numThreads - 1) / numThreads;
        expand_delta<<<expBlocks, numThreads>>>(
            D_offset, D_edges, D_weight, D_hx,
            D_frontier, H_frontier_size, threshold,
            D_next_queue, N, E, 0, nullptr, nullptr, nullptr, 0);

        int H_zero = 0;
        gpuErrchk(cudaMemcpy(D_frontier_size, &H_zero, sizeof(int), cudaMemcpyHostToDevice));
        compact_frontier<<<nBlocks, numThreads>>>(D_next_queue, D_frontier, D_frontier_size, N);
        gpuErrchk(cudaMemcpy(&H_frontier_size, D_frontier_size, sizeof(int), cudaMemcpyDeviceToHost));
    }

    cudaEventRecord(t_static_stop);
    cudaEventSynchronize(t_static_stop);
    
    float static_ms = 0;
    cudaEventElapsedTime(&static_ms, t_static_start, t_static_stop);

    unsigned int static_dest_cost = readCxCost(endNode);
    
    auto end_wall_static = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff_wall_static = end_wall_static - start_wall_static;
    
    printf("\n--- STATIC PHASE RESULTS ---\n");
    printf("Static A* (Wall Clock - CPU + Memcpy + GPU): %.3f ms\n", diff_wall_static.count() * 1000.0); 
    printf("Static A* (Pure GPU Execution Time)        : %.3f ms\n", static_ms);
    printf("Cost on static Graph                       : %d\n", (int)static_dest_cost);

    FILE* fdiff = fopen("Updates.txt","r");
    if (!fdiff) {
        std::cerr << "Failed to open Updates.txt\n";
        return 1;
    }

    // ── DYNAMIC PHASE ────────────────────────────────────────────
    cudaEvent_t t_dyn_start, t_dyn_stop;
    cudaEventCreate(&t_dyn_start);
    cudaEventCreate(&t_dyn_stop);

    // Pre-allocate reusable buffers for dynamic phase
    int* D_affected;
    int* D_curr_worklist;
    int* D_new_worklist;
    int* D_curr_sz;
    int* D_new_sz;
    gpuErrchk(cudaMalloc(&D_affected, sizeof(int) * N));
    gpuErrchk(cudaMalloc(&D_curr_worklist, sizeof(int) * N));
    gpuErrchk(cudaMalloc(&D_new_worklist, sizeof(int) * N));
    gpuErrchk(cudaMalloc(&D_curr_sz, sizeof(int)));
    gpuErrchk(cudaMalloc(&D_new_sz, sizeof(int)));

    // v7: Query max cooperative grid sizes (once, before batch loop)
    int coopBlockSize = 512;
    int maxBlocks_desc = getMaxCoopBlocks(
        (const void*)persistentDescendantMarking, coopBlockSize, 0);
    int maxBlocks_update = getMaxCoopBlocks(
        (const void*)fuseUpdateAndBuildFrontier, coopBlockSize, 0);
    int maxBlocks_astar = getMaxCoopBlocks(
        (const void*)persistentAStarDelta, coopBlockSize,
        sizeof(unsigned int) * 32);

    printf("\n--- COOPERATIVE KERNEL GRID LIMITS ---\n");
    printf("persistentDescendantMarking : max %d blocks\n", maxBlocks_desc);
    printf("fuseUpdateAndBuildFrontier  : max %d blocks\n", maxBlocks_update);
    printf("persistentAStarDelta        : max %d blocks\n", maxBlocks_astar);

    // Create non-blocking streams for concurrent deletion marking
    cudaStream_t s_fwd, s_rev, s_aff;
    cudaStreamCreateWithFlags(&s_fwd, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&s_rev, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&s_aff, cudaStreamNonBlocking);

    int line;
    while (fscanf(fdiff,"%d",&line)!=EOF)
    {
        int *H_u = (int *)malloc(sizeof(int) * line);
        int *H_v = (int *)malloc(sizeof(int) * line);

        for (size_t i = 0; i < line; i++) {
            int flag, wt;
            fscanf(fdiff,"%d %d %d %d",&flag,&H_u[i],&H_v[i], &wt);
        }
       
        int* D_u;
        int* D_v;
        gpuErrchk(cudaMalloc(&D_u, sizeof(int)*line));
        gpuErrchk(cudaMalloc(&D_v, sizeof(int)*line));
        gpuErrchk(cudaMemcpy(D_u, H_u, sizeof(int)*line, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(D_v, H_v, sizeof(int)*line, cudaMemcpyHostToDevice));

        // Reset buffers
        gpuErrchk(cudaMemset(D_affected, 0, sizeof(int) * N));
        gpuErrchk(cudaMemset(D_curr_sz, 0, sizeof(int)));
        gpuErrchk(cudaMemset(D_new_sz, 0, sizeof(int)));

        auto start_wall_dyn = std::chrono::high_resolution_clock::now();
        int threadsCnt = 512;

        // ── Step 1: Mark deleted edges (3 streams) + affected nodes ──
        int delBlocks = (line + threadsCnt - 1) / threadsCnt;
        cudaEventRecord(t_dyn_start, s_fwd);
        markDeletedBinarySearch<<<delBlocks, threadsCnt, 0, s_fwd>>>(
            D_u, D_v, D_offset, D_edges, D_weight, N, E, line);
        markDeletedBinarySearch<<<delBlocks, threadsCnt, 0, s_rev>>>(
            D_v, D_u, r_D_offset, r_D_edges, r_D_weight, N, E, line);
        markAsAffected<<<(line + threadsCnt - 1) / threadsCnt, threadsCnt, 0, s_aff>>>(
            D_u, D_v, D_affected, N, line);
        gpuErrchk(cudaDeviceSynchronize());

        // ── Step 2: Persistent descendant marking (includes InitializeWorklist as Phase 0) ──
        // Use small grid: Phase 0 scans N nodes (lightweight), descendant loop benefits
        // from fewer blocks → faster grid.sync() (~16 syncs in the loop)
        {
            int descGrid = min(maxBlocks_desc, 64);  // 64 blocks = 1024 warps, plenty for typical worklists
            descGrid = max(descGrid, 1);

            void* descArgs[] = {
                &D_edges, &D_offset,
                &D_curr_worklist, &D_curr_sz,
                &D_new_worklist, &D_new_sz,
                &D_affected, &N, &E
            };
            gpuErrchk(cudaLaunchCooperativeKernel(
                (const void*)persistentDescendantMarking,
                descGrid, threadsCnt, descArgs, 0, 0));
        }

        // ── Step 3: Fused update-weights + build-frontier ──
        gpuErrchk(cudaMemset(D_next_queue, 0, N * sizeof(int)));  // flag array
        gpuErrchk(cudaMemset(D_curr_sz, 0, sizeof(int)));
        gpuErrchk(cudaMemset(D_frontier_size, 0, sizeof(int)));

        {
            int updateGrid = min(maxBlocks_update, nBlocks);
            updateGrid = max(updateGrid, 1);

            void* updateArgs[] = {
                &D_affected,
                &r_D_offset, &r_D_edges, &r_D_weight,
                &D_hx,
                &D_curr_worklist, &D_curr_sz,
                &D_next_queue,
                &D_frontier, &D_frontier_size,
                &N, &E
            };
            gpuErrchk(cudaLaunchCooperativeKernel(
                (const void*)fuseUpdateAndBuildFrontier,
                updateGrid, threadsCnt, updateArgs, 0, 0));
        }
        gpuErrchk(cudaMemcpy(&H_frontier_size, D_frontier_size, sizeof(int), cudaMemcpyDeviceToHost));

        // ── Step 4: Persistent A* re-expansion ──
        if (H_frontier_size > 0) {
            gpuErrchk(cudaMemset(D_curr_sz, 0, sizeof(int)));

            int astarGrid = min(maxBlocks_astar, nBlocks);
            astarGrid = max(astarGrid, 1);

            void* astarArgs[] = {
                &D_offset, &D_edges, &D_weight, &D_hx,
                &D_frontier, &D_frontier_size,
                &D_curr_worklist, &D_curr_sz,
                &D_next_queue, &D_min_f,
                &N, &E, &DELTA_DYN, &endNode
            };
            gpuErrchk(cudaLaunchCooperativeKernel(
                (const void*)persistentAStarDelta,
                astarGrid, threadsCnt, astarArgs,
                sizeof(unsigned int) * 32, 0));
        }

        cudaEventRecord(t_dyn_stop);
        cudaEventSynchronize(t_dyn_stop);
        
        float dyn_ms = 0;
        cudaEventElapsedTime(&dyn_ms, t_dyn_start, t_dyn_stop);

        unsigned int dyn_dest_cost = readCxCost(endNode);
        
        auto end_wall_dyn = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff_wall_dyn = end_wall_dyn - start_wall_dyn;
        
        printf("\n--- DYNAMIC UPDATE RESULTS ---\n");
        printf("Dynamic Update (Wall Clock - CPU + GPU) : %.3f ms\n", diff_wall_dyn.count() * 1000.0); 
        printf("Dynamic Update (Pure GPU Execution Time): %.3f ms\n", dyn_ms);
        printf("Cost after deletion                     : %d\n", (int)dyn_dest_cost);

        free(H_u);
        free(H_v);  
        cudaFree(D_u);
        cudaFree(D_v);
    }

    // ── Cleanup ──
    fclose(fdiff);

    cudaFree(D_affected);
    cudaFree(D_curr_worklist);
    cudaFree(D_new_worklist);
    cudaFree(D_curr_sz);
    cudaFree(D_new_sz);

    cudaStreamDestroy(s_fwd);
    cudaStreamDestroy(s_rev);
    cudaStreamDestroy(s_aff);

    cudaFree(D_offset);
    cudaFree(D_edges);
    cudaFree(D_weight);
    cudaFree(r_D_offset);
    cudaFree(r_D_edges);
    cudaFree(r_D_weight);
    cudaFree(D_hx);
    cudaFree(D_parent);
    cudaFree(D_frontier);
    cudaFree(D_next_queue);
    cudaFree(D_frontier_size);
    cudaFree(D_min_f);

    free(H_offset);
    free(H_edges);
    free(H_weight);
    free(H_hx);
    free(H_cx);
    free(H_parent);
    free(r_H_offset);
    free(r_H_edges);
    free(r_H_weight);

    cudaEventDestroy(t_static_start);
    cudaEventDestroy(t_static_stop);
    cudaEventDestroy(t_dyn_start);
    cudaEventDestroy(t_dyn_stop);

    return 0;
}
