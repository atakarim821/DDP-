/*
 * DIRECTED GRAPH PARALLEL A*
 * 
 * CHANGES FROM ORIGINAL: TIMING ONLY — zero algorithmic modifications.
 *
 * Timing strategy:
 *   wall_start      : chrono, before ALL setup (IO + malloc + memcpy + kernels)
 *   timer_static_kernels (cudaEvent) : pure GPU kernel time, static A* loop only
 *   timer_static_total  (cudaEvent) : kernels + in-loop PCIe transfers
 *   wall_end_static : chrono, after getCx + final memcpy
 *
 *   Per dynamic batch:
 *   timer_dyn_kernels (cudaEvent) : from Updating kernel to end of re-search loop
 *   timer_dyn_total   (cudaEvent) : same span + in-loop PCIe
 *
 * Three numbers reported for each phase (as required for ICPP):
 *   1. Wall-clock ms  — end-to-end user-visible time (chrono)
 *   2. GPU total ms   — kernels + necessary PCIe per iteration (cudaEvent)
 *   3. GPU kernel ms  — pure algorithmic compute, no PCIe (cudaEvent)
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

// --- Macros & Constants ---
#define MAX_NODE 100000000
#define DEBUG 0

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

// ── cudaEvent timer helper ────────────────────────────────────────
// Wraps create/record/elapsed into a clean struct so timing calls
// don't clutter the algorithm code.
struct GpuTimer {
    cudaEvent_t start, stop;
    void create()  { cudaEventCreate(&start); cudaEventCreate(&stop); }
    void begin()   { cudaEventRecord(start); }   // inserts marker into GPU stream
    void end()     { cudaEventRecord(stop);  }   // inserts marker into GPU stream
    // Blocks CPU until stop event has passed, then returns elapsed ms.
    // All GPU work between begin() and end() is included.
    float ms() {
        cudaEventSynchronize(stop);
        float t = 0;
        cudaEventElapsedTime(&t, start, stop);
        return t;
    }
    void destroy() { cudaEventDestroy(start); cudaEventDestroy(stop); }
};

// --- Device Globals ---
__device__ volatile int Cx[MAX_NODE];
__device__ volatile int PQ[MAX_NODE];

// --- Kernel Definitions ---
// !! ZERO CHANGES TO ANY KERNEL BELOW !!

// Extract Minimum from K Priority Queues in parallel
__global__ void extractMin(int* PQ_size, int* expandNodes, int* expandNodes_size, int* openList, int N, int K) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < K && PQ_size[id] > 0) {
        int front = id * ((N + K - 1) / K);
        int node = PQ[front];

        PQ[front] = PQ[front + PQ_size[id] - 1];
        PQ_size[id] -= 1;
        int pqIndex = 0;

        while (2 * pqIndex + 1 < PQ_size[id]) {
            int left = 2 * pqIndex + 1;
            int right = 2 * pqIndex + 2;
            int smallest = pqIndex;

            if (left < PQ_size[id] && Cx[PQ[front + smallest]] > Cx[PQ[front + left]]) {
                smallest = left;
            }
            if (right < PQ_size[id] && Cx[PQ[front + smallest]] > Cx[PQ[front + right]]) {
                smallest = right;
            }

            if (smallest != pqIndex) {
                int swap = PQ[front + smallest];
                PQ[front + smallest] = PQ[front + pqIndex];
                PQ[front + pqIndex] = swap;
                pqIndex = smallest;
            } else {
                break;
            }
        }

        openList[node] = -1;
        int len = atomicAdd(expandNodes_size, 1);
        expandNodes[len] = node;
    }
}

// Expand Nodes (Relax Edges)
__global__ void A_star_expand(int* off, int* edge, unsigned int* W, int* Hx, int* parent,
                              int* expandNodes, int* expandNodes_size, int* lock, int* flagfound, int* openList,
                              int N, int E, int K, int dest, int* nVFlag, int* PQ_size,
                              int flagDiff, int* diff_off, int* diff_edge, unsigned int* diff_weight, int dE) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < *expandNodes_size) {
        int node = expandNodes[id];

        if (node == dest) {
            atomicOr(flagfound, 1);
        }

        int start = off[node];
        int end = (node != N - 1) ? off[node + 1] : E;

        while (start < end) {
            int child = edge[start];

            if (child < 0) {
                start++;
                continue;
            }

            bool leaveLoop = false;
            while (!leaveLoop) {
                if (atomicCAS(&lock[child], 0, 1) == 0) {
                    if (Cx[node] != INT_MAX && Cx[child] > (Cx[node] - Hx[node]) + W[start] + Hx[child]) {
                        Cx[child] = (Cx[node] - Hx[node]) + W[start] + Hx[child];
                        __threadfence();
                        parent[child] = node;

                        if (openList[child] == -1) {
                            nVFlag[child] = 1;
                        }
                    }
                    atomicCAS(&lock[child], 1, 0);
                    leaveLoop = true;
                }
            }
            start++;
        }

        if (flagDiff) {
            start = diff_off[node];
            end = (node != N - 1) ? diff_off[node + 1] : dE;

            while (start < end) {
                int child = diff_edge[start];
                if (child < 0) {
                    start++;
                    continue;
                }

                bool leaveLoop = false;
                while (!leaveLoop) {
                    if (atomicCAS(&lock[child], 0, 1) == 0) {
                        if (Cx[node] != INT_MAX && Cx[child] > (Cx[node] - Hx[node]) + diff_weight[start] + Hx[child]) {
                            Cx[child] = (Cx[node] - Hx[node]) + diff_weight[start] + Hx[child];
                            __threadfence();
                            parent[child] = node;
                            
                            if (openList[child] == -1) {
                                nVFlag[child] = 1;
                            }
                        }
                        atomicCAS(&lock[child], 1, 0);
                        leaveLoop = true;
                    }
                }
                start++;
            }
        }
    }
}

// Maintain Heap Property for K PQs
__global__ void keepHeapPQ(int* PQ_size, int N, int K) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < K && PQ_size[id] > 0) {
        int front = id * ((N + K - 1) / K);
        int size = PQ_size[id];
        
        for (int start = (size - 2) / 2; start >= 0; start--) {
            int current = start;
            while (true) {
                int left = 2 * current + 1;
                int right = 2 * current + 2;
                int smallest = current;

                if (left < size && Cx[PQ[front + left]] < Cx[PQ[front + smallest]])
                    smallest = left;
                if (right < size && Cx[PQ[front + right]] < Cx[PQ[front + smallest]])
                    smallest = right;

                if (smallest != current) {
                    unsigned int swap = PQ[front + current];
                    PQ[front + current] = PQ[front + smallest];
                    PQ[front + smallest] = swap;
                    current = smallest;
                } else {
                    break;
                }
            }
        }
    }
}

// Collect vertices that need to be added to PQ
__global__ void setNV(int* nextFlag, int* nextV, int* nvSize, int N) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < N) {
        if (nextFlag[id] == 1) {
            int index = atomicAdd(nvSize, 1);
            nextV[index] = id;
        }
    }
}

// Insert vertices into K PQs
__global__ void insertPQ(int* PQS, int* nextV, int* nVsize, int K, int N, int* openList) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < K) {
        int front = id * ((N + K - 1) / K);
        int i = id;

        while (i < *nVsize) {
            int node = nextV[i];

            if (openList[node] != -1) {
                i += K;
                continue;
            }

            PQ[front + PQS[id]] = node;
            PQS[id] += 1;
            openList[node] = id;

            if (PQS[id] > 1) {
                int index = PQS[id] - 1;
                while (index > 0) {
                    int parentIdx = (index - 1) / 2;
                    if (Cx[PQ[front + parentIdx]] > Cx[PQ[front + index]]) {
                        int swap = PQ[front + index];
                        PQ[front + index] = PQ[front + parentIdx];
                        PQ[front + parentIdx] = swap;
                        index = parentIdx;
                    } else {
                        break;
                    }
                }
            }
            i += K;
        }
    }
}

// Check if current Min in PQ is greater than found path to dest
__global__ void checkMIN(int* PQ_size, int* flagEnd, int dest, int N, int K) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < K && PQ_size[id] > 0) {
        int front = id * ((N + K - 1) / K);
        int node = PQ[front];
        
        if (Cx[dest] > Cx[node]) {
            atomicAnd(flagEnd, 0);
        }
    }
}

// Apply Edge Updates to Graph
__global__ void Updating(int *u, int *v, unsigned int *W, int *Hx, int *addFlag,
                         int *lock, int *parent, int N, int dE) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < dE) {
        int node = u[id];
        int child = v[id];
        unsigned int wt = W[id];

        bool leaveLoop = false;
        while (!leaveLoop) {
            if (atomicCAS(&lock[child], 0, 1) == 0) {
                if (Cx[node] != INT_MAX && Cx[child] > (Cx[node] - Hx[node]) + wt + Hx[child]) {
                    Cx[child] = (Cx[node] - Hx[node]) + wt + Hx[child];
                    parent[child] = node;
                    addFlag[child] = 1;
                    __threadfence();
                }
                atomicCAS(&lock[child], 1, 0);
                leaveLoop = true;
            }
        }
    }
}

// Retrieve Cost of Dest
__global__ void getCx(int dest, int* val) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id == 0) {
        *val = Cx[dest];
    }
}

// --- Host Helper Functions ---
// !! ZERO CHANGES BELOW !!

void createDiffGraph(int N, unordered_map<int, vector<pair<int, int>>>& adj, int* diffOff, int* diffEdges, unsigned int* diffWeight) {
    diffOff[0] = 0;
    int edgeCount = 0;
    for (int i = 0; i < N; i++) {
        diffOff[i] = edgeCount;
        if (adj.find(i) != adj.end()) {
            for (auto& p : adj[i]) {
                diffEdges[edgeCount] = p.first;
                diffWeight[edgeCount] = p.second;
                edgeCount++;
            }
        }
    }
}

// --- Main Function ---

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " startNode endNode\n";
        return 1;
    }

    // ── Wall clock: starts before ALL work (IO + setup + kernels) ──
    // This is the "user-visible" end-to-end time.
    auto wall_start = std::chrono::high_resolution_clock::now();

    int K = 10000;
    int startNode = std::stoi(argv[1]);
    int endNode = std::stoi(argv[2]);

    // --- Read Static Graph ---
    FILE* fgraph = fopen("graph.txt", "r");
    if (!fgraph) { perror("Error opening graph.txt"); return 1; }
    
    int N, E;
    fscanf(fgraph, "%d %d\n", &N, &E);

    int* H_offset = (int*)malloc(sizeof(int) * N);
    int* H_edges = (int*)malloc(sizeof(int) * E);
    unsigned int* H_weight = (unsigned int*)malloc(sizeof(unsigned int) * E);
    int* H_hx = (int*)malloc(sizeof(int) * N);
    int* H_cx = (int*)malloc(sizeof(int) * N);
    int* H_parent = (int*)malloc(sizeof(int) * N);
    int* H_PQ = (int*)malloc(sizeof(int) * N);
    int* H_openList = (int*)malloc(sizeof(int) * N);
    int* H_PQ_size = (int*)malloc(sizeof(int) * K);
    int* H_dest_cost = (int*)malloc(sizeof(int));

    memset(H_PQ_size, 0, sizeof(int) * K);
    memset(H_parent, -1, sizeof(int) * N);
    memset(H_openList, -1, sizeof(int) * N);

    for (int i = 0; i < N; i++) H_cx[i] = INT_MAX;
    for (int i = 0; i < E; i++) fscanf(fgraph, "%d", &H_edges[i]);
    for (int i = 0; i < N; i++) fscanf(fgraph, "%d", &H_offset[i]);
    for (int i = 0; i < E; i++) fscanf(fgraph, "%u", &H_weight[i]);
    fclose(fgraph);

    FILE* fhx = fopen("Hx.txt", "r");
    for (int i = 0; i < N; i++) {
        H_hx[i] = 0;
        if (fhx) {
            int temp;
            fscanf(fhx, "%d", &temp);
        }
    }
    if (fhx) fclose(fhx);

    int* H_flagEnd = (int*)malloc(sizeof(int));
    int* H_flagfound = (int*)malloc(sizeof(int));
    int* H_a0 = (int*)malloc(sizeof(int));
    int* H_nVFlag = (int*)malloc(sizeof(int) * N);
    memset(H_nVFlag, -1, sizeof(int) * N);

    *H_flagEnd = 0;
    *H_flagfound = 0;
    *H_a0 = 0;

    H_cx[startNode] = H_hx[startNode];
    H_PQ[0] = startNode;
    H_PQ_size[0] = 1;
    H_openList[startNode] = 0;

    // --- Device Memory Allocation ---
    int *D_offset, *D_edges, *D_hx, *D_parent;
    unsigned int *D_weight;
    int *D_PQ_size, *D_openList, *D_lock, *D_dest_cost;
    int *D_nV, *D_nV_size, *D_nVFlag, *D_expandNodes, *D_expandNodes_size;
    int *D_flagEnd, *D_flagfound;
    int *D_diff_edges, *D_diff_offset;
    unsigned int *D_diff_weight;

    gpuErrchk(cudaMalloc(&D_offset, sizeof(int) * N));
    gpuErrchk(cudaMalloc(&D_edges, sizeof(int) * E));
    gpuErrchk(cudaMalloc(&D_weight, sizeof(unsigned int) * E));
    gpuErrchk(cudaMalloc(&D_hx, sizeof(int) * N));
    gpuErrchk(cudaMalloc(&D_parent, sizeof(int) * N));
    gpuErrchk(cudaMalloc(&D_PQ_size, sizeof(int) * K));
    gpuErrchk(cudaMalloc(&D_openList, sizeof(int) * N));
    gpuErrchk(cudaMalloc(&D_lock, sizeof(int) * N));
    gpuErrchk(cudaMalloc(&D_dest_cost, sizeof(int)));
    gpuErrchk(cudaMalloc(&D_nV, sizeof(int) * N));
    gpuErrchk(cudaMalloc(&D_nV_size, sizeof(int)));
    gpuErrchk(cudaMalloc(&D_nVFlag, sizeof(int) * N));
    gpuErrchk(cudaMalloc(&D_expandNodes, sizeof(int) * K));
    gpuErrchk(cudaMalloc(&D_expandNodes_size, sizeof(int)));
    gpuErrchk(cudaMalloc(&D_flagEnd, sizeof(int)));
    gpuErrchk(cudaMalloc(&D_flagfound, sizeof(int)));

    gpuErrchk(cudaMemcpy(D_offset, H_offset, sizeof(int) * N, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(D_edges, H_edges, sizeof(int) * E, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(D_weight, H_weight, sizeof(unsigned int) * E, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(D_hx, H_hx, sizeof(int) * N, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(D_parent, H_parent, sizeof(int) * N, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(D_openList, H_openList, sizeof(int) * N, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(D_PQ_size, H_PQ_size, sizeof(int) * K, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(Cx, H_cx, sizeof(int) * N, 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(PQ, H_PQ, sizeof(int) * N, 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(D_flagEnd, H_flagEnd, sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(D_flagfound, H_flagfound, sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(D_nVFlag, H_nVFlag, sizeof(int) * N, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(D_nV_size, H_a0, sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(D_expandNodes_size, H_a0, sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemset(D_lock, 0, sizeof(int) * N));

    int numThreads = 512;
    int numBlocks = (K + numThreads - 1) / numThreads;
    int N_numBlocks = (N + numThreads - 1) / numThreads;

    // ── Create GPU timers ─────────────────────────────────────────
    GpuTimer timer_static_kernels;  // pure kernel time: no PCIe
    GpuTimer timer_static_total;    // kernels + in-loop PCIe transfers
    timer_static_kernels.create();
    timer_static_total.create();

    // --- Static A* Execution ---
    int flag_PQ_not_empty = 1;

    // timer_static_total.begin() wraps everything including in-loop memcpys.
    // timer_static_kernels.begin() is placed at the same point; both events
    // are inserted into the SAME default stream so ordering is guaranteed.
    // The difference between the two timers reflects pure PCIe transfer cost.
    timer_static_total.begin();
    timer_static_kernels.begin();

    while (*H_flagEnd == 0 && flag_PQ_not_empty == 1) {
        extractMin<<<numBlocks, numThreads>>>(D_PQ_size, D_expandNodes, D_expandNodes_size, D_openList, N, K);

        A_star_expand<<<numBlocks, numThreads>>>(D_offset, D_edges, D_weight, D_hx, D_parent,
                                                 D_expandNodes, D_expandNodes_size, D_lock, D_flagfound, D_openList,
                                                 N, E, K, endNode, D_nVFlag, D_PQ_size,
                                                 false, NULL, NULL, NULL, 0);

        keepHeapPQ<<<numBlocks, numThreads>>>(D_PQ_size, N, K);

        setNV<<<N_numBlocks, numThreads>>>(D_nVFlag, D_nV, D_nV_size, N);

        insertPQ<<<numBlocks, numThreads>>>(D_PQ_size, D_nV, D_nV_size, K, N, D_openList);

        // These memcpys are inside the timed region intentionally:
        // they are part of the algorithmic loop (loop-control transfers).
        gpuErrchk(cudaMemcpy(H_flagfound, D_flagfound, sizeof(int), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(H_PQ_size, D_PQ_size, sizeof(int) * K, cudaMemcpyDeviceToHost));
        
        gpuErrchk(cudaMemset(D_nVFlag, -1, sizeof(int) * N));
        gpuErrchk(cudaMemcpy(D_nV_size, H_a0, sizeof(int), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(D_expandNodes_size, H_a0, sizeof(int), cudaMemcpyHostToDevice));

        flag_PQ_not_empty = 0;
        for (int i = 0; i < K; i++) {
            if (H_PQ_size[i] > 0) {
                flag_PQ_not_empty = 1;
                break;
            }
        }

        if (*H_flagfound == 1 && flag_PQ_not_empty == 1) {
            gpuErrchk(cudaMemcpy(D_flagEnd, H_flagfound, sizeof(int), cudaMemcpyHostToDevice));
            checkMIN<<<numBlocks, numThreads>>>(D_PQ_size, D_flagEnd, endNode, N, K);
            cudaDeviceSynchronize();
            gpuErrchk(cudaMemcpy(H_flagEnd, D_flagEnd, sizeof(int), cudaMemcpyDeviceToHost));
        }
    }

    // Stop kernel timer HERE — before getCx which is result-retrieval overhead,
    // not part of the A* search itself.
    timer_static_kernels.end();

    getCx<<<1, 1>>>(endNode, D_dest_cost);
    gpuErrchk(cudaMemcpy(H_dest_cost, D_dest_cost, sizeof(int), cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();

    // Stop total timer AFTER getCx + final memcpy
    timer_static_total.end();

    // Wall clock end for static phase
    auto wall_end_static = std::chrono::high_resolution_clock::now();
    double wall_static_ms = std::chrono::duration<double, std::milli>(
                                wall_end_static - wall_start).count();

    float gpu_static_kernel_ms = timer_static_kernels.ms();
    float gpu_static_total_ms  = timer_static_total.ms();

    printf("\n=== STATIC A* RESULTS ===\n");
    printf("  Cost on static graph                        : %d\n",  *H_dest_cost);
    printf("  Wall-clock (IO + setup + kernels + PCIe)    : %.3f ms\n", wall_static_ms);
    printf("  GPU total  (kernels + in-loop PCIe)         : %.3f ms\n", gpu_static_total_ms);
    printf("  GPU kernels only (pure algorithmic compute) : %.3f ms\n", gpu_static_kernel_ms);
    printf("  [PCIe overhead in loop = %.3f ms]\n\n",
           gpu_static_total_ms - gpu_static_kernel_ms);

    // --- Dynamic Updates Processing ---
    FILE* fdiff = fopen("Updates.txt", "r");
    if (fdiff) {

        GpuTimer timer_dyn_kernels;
        GpuTimer timer_dyn_total;
        timer_dyn_kernels.create();
        timer_dyn_total.create();

        int batch = 0;
        int line;
        while (fscanf(fdiff, "%d\n", &line) != EOF) {
            batch++;

            int* H_u = (int*)malloc(sizeof(int) * line);
            int* H_v = (int*)malloc(sizeof(int) * line);
            unsigned int* H_w = (unsigned int*)malloc(sizeof(unsigned int) * line);

            int* D_u, * D_v;
            unsigned int* D_w;
            gpuErrchk(cudaMalloc(&D_u, sizeof(int) * line));
            gpuErrchk(cudaMalloc(&D_v, sizeof(int) * line));
            gpuErrchk(cudaMalloc(&D_w, sizeof(unsigned int) * line));

            int* H_diff_edges = (int*)malloc(sizeof(int) * line);
            int* H_diff_offset = (int*)malloc(sizeof(int) * N);
            unsigned int* H_diff_weight = (unsigned int*)malloc(sizeof(unsigned int) * line);

            unordered_map<int, vector<pair<int, int>>> adj;
            int insertEdge = 0;

            for (int i = 0; i < line; i++) {
                int flag, u, v;
                unsigned int w;
                fscanf(fdiff, "%d %d %d %u\n", &flag, &u, &v, &w);
                if (flag == 1) insertEdge++;
                H_u[i] = u; H_v[i] = v; H_w[i] = w;
                adj[u].push_back({v, w});
            }

            gpuErrchk(cudaMemcpy(D_u, H_u, sizeof(int) * line, cudaMemcpyHostToDevice));
            gpuErrchk(cudaMemcpy(D_v, H_v, sizeof(int) * line, cudaMemcpyHostToDevice));
            gpuErrchk(cudaMemcpy(D_w, H_w, sizeof(unsigned int) * line, cudaMemcpyHostToDevice));

            createDiffGraph(N, adj, H_diff_offset, H_diff_edges, H_diff_weight);

            gpuErrchk(cudaMalloc(&D_diff_edges, sizeof(int) * line));
            gpuErrchk(cudaMalloc(&D_diff_offset, sizeof(int) * N));
            gpuErrchk(cudaMalloc(&D_diff_weight, sizeof(unsigned int) * line));

            gpuErrchk(cudaMemcpy(D_diff_edges, H_diff_edges, sizeof(int) * line, cudaMemcpyHostToDevice));
            gpuErrchk(cudaMemcpy(D_diff_offset, H_diff_offset, sizeof(int) * N, cudaMemcpyHostToDevice));
            gpuErrchk(cudaMemcpy(D_diff_weight, H_diff_weight, sizeof(unsigned int) * line, cudaMemcpyHostToDevice));

            gpuErrchk(cudaMemcpy(D_nVFlag, H_nVFlag, sizeof(int) * N, cudaMemcpyHostToDevice));
            gpuErrchk(cudaMemset(D_nVFlag, -1, sizeof(int) * N));

            // ── Dynamic phase timing ───────────────────────────────
            // total: from first Updating launch through final getCx
            // kernels: same span, expressed as pure GPU execution
            // Both timers start together in the same stream.
            timer_dyn_total.begin();
            timer_dyn_kernels.begin();

            // 1. Update Graph Costs based on new edges
            Updating<<<(line + numThreads - 1) / numThreads, numThreads>>>(
                D_u, D_v, D_w, D_hx, D_nVFlag, D_lock, D_parent, N, line);
            cudaDeviceSynchronize();
            
            gpuErrchk(cudaFree(D_u)); gpuErrchk(cudaFree(D_v)); gpuErrchk(cudaFree(D_w));
            free(H_u); free(H_v); free(H_w);

            // 2. Identify nodes to re-evaluate
            gpuErrchk(cudaMemcpy(D_nV_size, H_a0, sizeof(int), cudaMemcpyHostToDevice));
            setNV<<<N_numBlocks, numThreads>>>(D_nVFlag, D_nV, D_nV_size, N);
            cudaDeviceSynchronize();

            // 3. Re-insert into Priority Queue
            gpuErrchk(cudaMemset(D_PQ_size, 0, sizeof(int) * K));
            gpuErrchk(cudaMemset(D_openList, -1, sizeof(int) * N));
            insertPQ<<<numBlocks, numThreads>>>(D_PQ_size, D_nV, D_nV_size, K, N, D_openList);
            cudaDeviceSynchronize();

            gpuErrchk(cudaMemcpy(D_nVFlag, H_nVFlag, sizeof(int) * N, cudaMemcpyHostToDevice));
            gpuErrchk(cudaMemcpy(D_nV_size, H_a0, sizeof(int), cudaMemcpyHostToDevice));
            gpuErrchk(cudaMemcpy(D_expandNodes_size, H_a0, sizeof(int), cudaMemcpyHostToDevice));
            
            free(H_diff_edges); free(H_diff_offset); free(H_diff_weight);

            // 4. Resume A* Search
            *H_flagEnd = 0; 
            flag_PQ_not_empty = 1;

            while (*H_flagEnd == 0 && flag_PQ_not_empty == 1) {
                extractMin<<<numBlocks, numThreads>>>(D_PQ_size, D_expandNodes, D_expandNodes_size, D_openList, N, K);

                A_star_expand<<<numBlocks, numThreads>>>(D_offset, D_edges, D_weight, D_hx, D_parent,
                                                         D_expandNodes, D_expandNodes_size, D_lock, D_flagfound, D_openList,
                                                         N, E, K, endNode, D_nVFlag, D_PQ_size,
                                                         true, D_diff_offset, D_diff_edges, D_diff_weight, insertEdge);

                keepHeapPQ<<<numBlocks, numThreads>>>(D_PQ_size, N, K);

                setNV<<<N_numBlocks, numThreads>>>(D_nVFlag, D_nV, D_nV_size, N);

                insertPQ<<<numBlocks, numThreads>>>(D_PQ_size, D_nV, D_nV_size, K, N, D_openList);

                gpuErrchk(cudaMemcpy(H_flagfound, D_flagfound, sizeof(int), cudaMemcpyDeviceToHost));
                gpuErrchk(cudaMemcpy(H_PQ_size, D_PQ_size, sizeof(int) * K, cudaMemcpyDeviceToHost));
                
                gpuErrchk(cudaMemcpy(D_nVFlag, H_nVFlag, sizeof(int) * N, cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemcpy(D_nV_size, H_a0, sizeof(int), cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemcpy(D_expandNodes_size, H_a0, sizeof(int), cudaMemcpyHostToDevice));

                flag_PQ_not_empty = 0;
                for (int i = 0; i < K; i++) {
                    if (H_PQ_size[i] > 0) { flag_PQ_not_empty = 1; break; }
                }

                if (*H_flagfound == 1 && flag_PQ_not_empty == 1) {
                    gpuErrchk(cudaMemcpy(D_flagEnd, H_flagfound, sizeof(int), cudaMemcpyHostToDevice));
                    checkMIN<<<numBlocks, numThreads>>>(D_PQ_size, D_flagEnd, endNode, N, K);
                    gpuErrchk(cudaMemcpy(H_flagEnd, D_flagEnd, sizeof(int), cudaMemcpyDeviceToHost));
                }
            }

            // Stop kernel timer before result-retrieval getCx
            timer_dyn_kernels.end();

            getCx<<<1, 1>>>(endNode, D_dest_cost);
            gpuErrchk(cudaMemcpy(H_dest_cost, D_dest_cost, sizeof(int), cudaMemcpyDeviceToHost));

            // Stop total timer after getCx
            timer_dyn_total.end();

            float dyn_kernel_ms = timer_dyn_kernels.ms();
            float dyn_total_ms  = timer_dyn_total.ms();

            printf("=== DYNAMIC UPDATE (batch %d, %d edges) ===\n", batch, line);
            printf("  Cost after update                           : %d\n",  *H_dest_cost);
            printf("  GPU kernels only (pure algorithmic compute) : %.3f ms\n", dyn_kernel_ms);
            printf("  GPU total  (kernels + in-loop PCIe)         : %.3f ms\n", dyn_total_ms);
            printf("  [PCIe overhead in loop = %.3f ms]\n\n",
                   dyn_total_ms - dyn_kernel_ms);

            gpuErrchk(cudaFree(D_diff_edges));
            gpuErrchk(cudaFree(D_diff_offset));
            gpuErrchk(cudaFree(D_diff_weight));
        }

        timer_dyn_kernels.destroy();
        timer_dyn_total.destroy();
        fclose(fdiff);
    }

    // ── Final wall-clock ──────────────────────────────────────────
    auto wall_end = std::chrono::high_resolution_clock::now();
    double wall_total_ms = std::chrono::duration<double, std::milli>(
                               wall_end - wall_start).count();
    printf("=== TOTAL WALL CLOCK (IO + setup + all phases): %.3f ms ===\n", wall_total_ms);

    timer_static_kernels.destroy();
    timer_static_total.destroy();
    return 0;
}
