/*
 * DIRECTED GRAPH PARALLEL A* (Lock-Free atomicMin Version)
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

// High value for infinity in 32-bit (used for packing)
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

// --- Device Globals ---
// Cx now stores: [32-bit Cost | 32-bit Parent]
__device__ volatile unsigned long long Cx[MAX_NODE];
__device__ volatile int PQ[MAX_NODE];

// --- Helper Functions ---
__device__ __host__ inline unsigned long long pack(unsigned int cost, int parent) {
    return ((unsigned long long)cost << 32) | (unsigned int)parent;
}

__device__ __host__ inline unsigned int unpackCost(unsigned long long val) {
    return (unsigned int)(val >> 32);
}

__device__ __host__ inline int unpackParent(unsigned long long val) {
    return (int)(val & 0xFFFFFFFF);
}

// --- Kernel Definitions ---

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

            if (left < PQ_size[id] && unpackCost(Cx[PQ[front + smallest]]) > unpackCost(Cx[PQ[front + left]])) {
                smallest = left;
            }
            if (right < PQ_size[id] && unpackCost(Cx[PQ[front + smallest]]) > unpackCost(Cx[PQ[front + right]])) {
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

__global__ void A_star_expand(int* off, int* edge, unsigned int* W, int* Hx, 
                              int* expandNodes, int* expandNodes_size, int* flagfound, int* openList,
                              int N, int E, int K, int dest, int* nVFlag, int* PQ_size,
                              int flagDiff, int* diff_off, int* diff_edge, unsigned int* diff_weight, int dE) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < *expandNodes_size) {
        int node = expandNodes[id];

        if (node == dest) {
            atomicOr(flagfound, 1);
        }

        unsigned long long nodeData = Cx[node];
        unsigned int node_f_cost = unpackCost(nodeData);
        
        // g(node) = f(node) - h(node)
        unsigned int node_g_cost = node_f_cost - Hx[node];

        // --- Expand Static Edges ---
        int start = off[node];
        int end = (node != N - 1) ? off[node + 1] : E;

        for (; start < end; start++) {
            int child = edge[start];
            if (child < 0) continue;

            unsigned int weight = W[start];
            unsigned int new_f_cost = node_g_cost + weight + Hx[child];
            unsigned long long new_val = pack(new_f_cost, node);

            // ATOMIC MIN: replaces lock loop
            unsigned long long old_val = atomicMin((unsigned long long*)&Cx[child], new_val);

            if (new_val < old_val) {
                if (openList[child] == -1) {
                    nVFlag[child] = 1;
                }
            }
        }

        // --- Expand Dynamic (Diff) Edges ---
        if (flagDiff) {
            start = diff_off[node];
            end = (node != N - 1) ? diff_off[node + 1] : dE;

            for (; start < end; start++) {
                int child = diff_edge[start];
                if (child < 0) continue;

                unsigned int weight = diff_weight[start];
                unsigned int new_f_cost = node_g_cost + weight + Hx[child];
                unsigned long long new_val = pack(new_f_cost, node);

                unsigned long long old_val = atomicMin((unsigned long long*)&Cx[child], new_val);

                if (new_val < old_val) {
                    if (openList[child] == -1) {
                        nVFlag[child] = 1;
                    }
                }
            }
        }
    }
}

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

                if (left < size && unpackCost(Cx[PQ[front + left]]) < unpackCost(Cx[PQ[front + smallest]]))
                    smallest = left;
                if (right < size && unpackCost(Cx[PQ[front + right]]) < unpackCost(Cx[PQ[front + smallest]]))
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

__global__ void setNV(int* nextFlag, int* nextV, int* nvSize, int N) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < N) {
        if (nextFlag[id] == 1) {
            int index = atomicAdd(nvSize, 1);
            nextV[index] = id;
        }
    }
}

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
                    if (unpackCost(Cx[PQ[front + parentIdx]]) > unpackCost(Cx[PQ[front + index]])) {
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

__global__ void checkMIN(int* PQ_size, int* flagEnd, int dest, int N, int K) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < K && PQ_size[id] > 0) {
        int front = id * ((N + K - 1) / K);
        int node = PQ[front];
        if (unpackCost(Cx[dest]) > unpackCost(Cx[node])) {
            atomicAnd(flagEnd, 0);
        }
    }
}

__global__ void Updating(int *u, int *v, unsigned int *W, int *Hx, int *addFlag, int N, int dE) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < dE) {
        int node = u[id];
        int child = v[id];
        unsigned int wt = W[id];

        unsigned long long nodeData = Cx[node];
        unsigned int node_f = unpackCost(nodeData);
        
        if (node_f != INF_COST) {
            unsigned int node_g = node_f - Hx[node];
            unsigned int new_f = node_g + wt + Hx[child];
            unsigned long long new_val = pack(new_f, node);

            unsigned long long old_val = atomicMin((unsigned long long*)&Cx[child], new_val);
            if (new_val < old_val) {
                addFlag[child] = 1;
            }
        }
    }
}

__global__ void getCx(int dest, int* val) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *val = (int)unpackCost(Cx[dest]);
    }
}

// --- Host Helper Functions ---

void createDiffGraph(int N, unordered_map<int, vector<pair<int, int>>>& adj, int* diffOff, int* diffEdges, unsigned int* diffWeight) {
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

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " startNode endNode\n";
        return 1;
    }

    auto start_time = std::chrono::high_resolution_clock::now();
    int K = 10000; 
    int startNode = std::stoi(argv[1]);
    int endNode = std::stoi(argv[2]);

    FILE* fgraph = fopen("graph.txt", "r");
    if (!fgraph) { perror("Error opening graph.txt"); return 1; }
    
    int N, E;
    fscanf(fgraph, "%d %d\n", &N, &E);

    int* H_offset = (int*)malloc(sizeof(int) * N);
    int* H_edges = (int*)malloc(sizeof(int) * E);
    unsigned int* H_weight = (unsigned int*)malloc(sizeof(unsigned int) * E);
    int* H_hx = (int*)malloc(sizeof(int) * N);
    unsigned long long* H_cx = (unsigned long long*)malloc(sizeof(unsigned long long) * N);
    int* H_PQ = (int*)malloc(sizeof(int) * N);
    int* H_openList = (int*)malloc(sizeof(int) * N);
    int* H_PQ_size = (int*)malloc(sizeof(int) * K);
    int* H_dest_cost = (int*)malloc(sizeof(int));

    memset(H_PQ_size, 0, sizeof(int) * K);
    memset(H_openList, -1, sizeof(int) * N);

    // Initialize packed Cx with Infinity
    for (int i = 0; i < N; i++) H_cx[i] = pack(INF_COST, -1);
    for (int i = 0; i < E; i++) fscanf(fgraph, "%d", &H_edges[i]);
    for (int i = 0; i < N; i++) fscanf(fgraph, "%d", &H_offset[i]);
    for (int i = 0; i < E; i++) fscanf(fgraph, "%u", &H_weight[i]);
    fclose(fgraph);

    FILE* fhx = fopen("Hx.txt", "r");
    for (int i = 0; i < N; i++) {
        H_hx[i] = 0;
        if (fhx) {
            int temp;
            if(fscanf(fhx, "%d", &temp) == 1) H_hx[i] = temp;
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

    // Start node setup
    H_cx[startNode] = pack(H_hx[startNode], -1);
    H_PQ[0] = startNode;
    H_PQ_size[0] = 1;
    H_openList[startNode] = 0;

    int *D_offset, *D_edges, *D_hx;
    unsigned int *D_weight;
    int *D_PQ_size, *D_openList, *D_dest_cost;
    int *D_nV, *D_nV_size, *D_nVFlag, *D_expandNodes, *D_expandNodes_size;
    int *D_flagEnd, *D_flagfound;
    int *D_diff_edges, *D_diff_offset;
    unsigned int *D_diff_weight;

    gpuErrchk(cudaMalloc(&D_offset, sizeof(int) * N));
    gpuErrchk(cudaMalloc(&D_edges, sizeof(int) * E));
    gpuErrchk(cudaMalloc(&D_weight, sizeof(unsigned int) * E));
    gpuErrchk(cudaMalloc(&D_hx, sizeof(int) * N));
    gpuErrchk(cudaMalloc(&D_PQ_size, sizeof(int) * K));
    gpuErrchk(cudaMalloc(&D_openList, sizeof(int) * N));
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
    gpuErrchk(cudaMemcpy(D_openList, H_openList, sizeof(int) * N, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(D_PQ_size, H_PQ_size, sizeof(int) * K, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(Cx, H_cx, sizeof(unsigned long long) * N, 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(PQ, H_PQ, sizeof(int) * N, 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(D_flagEnd, H_flagEnd, sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(D_flagfound, H_flagfound, sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(D_nVFlag, H_nVFlag, sizeof(int) * N, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(D_nV_size, H_a0, sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(D_expandNodes_size, H_a0, sizeof(int), cudaMemcpyHostToDevice));

    int numThreads = 512; 
    int numBlocks = (K + numThreads - 1) / numThreads;
    int N_numBlocks = (N + numThreads - 1) / numThreads;

    int flag_PQ_not_empty = 1;
    while (*H_flagEnd == 0 && flag_PQ_not_empty == 1) {
        extractMin<<<numBlocks, numThreads>>>(D_PQ_size, D_expandNodes, D_expandNodes_size, D_openList, N, K);
        A_star_expand<<<numBlocks, numThreads>>>(D_offset, D_edges, D_weight, D_hx,
                                                                          D_expandNodes, D_expandNodes_size, D_flagfound, D_openList,
                                                                          N, E, K, endNode, D_nVFlag, D_PQ_size,
                                                                          0, NULL, NULL, NULL, 0);
        keepHeapPQ<<<numBlocks, numThreads>>>(D_PQ_size, N, K);
        setNV<<<N_numBlocks, numThreads>>>(D_nVFlag, D_nV, D_nV_size, N);
        insertPQ<<<numBlocks, numThreads>>>(D_PQ_size, D_nV, D_nV_size, K, N, D_openList);

        gpuErrchk(cudaMemcpy(H_flagfound, D_flagfound, sizeof(int), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(H_PQ_size, D_PQ_size, sizeof(int) * K, cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemset(D_nVFlag, -1, sizeof(int) * N));
        gpuErrchk(cudaMemcpy(D_nV_size, H_a0, sizeof(int), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(D_expandNodes_size, H_a0, sizeof(int), cudaMemcpyHostToDevice));

        flag_PQ_not_empty = 0;
        for (int i = 0; i < K; i++) { if (H_PQ_size[i] > 0) { flag_PQ_not_empty = 1; break; } }

        if (*H_flagfound == 1 && flag_PQ_not_empty == 1) {
            gpuErrchk(cudaMemcpy(D_flagEnd, H_flagfound, sizeof(int), cudaMemcpyHostToDevice));
            checkMIN<<<numBlocks, numThreads>>>(D_PQ_size, D_flagEnd, endNode, N, K);
            gpuErrchk(cudaMemcpy(H_flagEnd, D_flagEnd, sizeof(int), cudaMemcpyDeviceToHost));
        }
    }

    getCx<<<1, 1>>>(endNode, D_dest_cost);
    gpuErrchk(cudaMemcpy(H_dest_cost, D_dest_cost, sizeof(int), cudaMemcpyDeviceToHost));

    auto end_time = std::chrono::high_resolution_clock::now();
    auto diff = std::chrono::duration<double>(end_time - start_time);
    printf("Time taken for static A*: %f seconds\n", diff.count());
    printf("Cost on static Graph : %d\n", *H_dest_cost);
    // --- Dynamic Updates ---
    FILE* fdiff = fopen("Updates.txt", "r");
    if (fdiff) {
        int line;
        while (fscanf(fdiff, "%d\n", &line) != EOF) {
            int* H_u = (int*)malloc(sizeof(int) * line);
            int* H_v = (int*)malloc(sizeof(int) * line);
            unsigned int* H_w = (unsigned int*)malloc(sizeof(unsigned int) * line);
            int* H_diff_edges = (int*)malloc(sizeof(int) * line);
            int* H_diff_offset = (int*)malloc(sizeof(int) * N);
            unsigned int* H_diff_weight = (unsigned int*)malloc(sizeof(unsigned int) * line);
            unordered_map<int, vector<pair<int, int>>> adj;

            for (int i = 0; i < line; i++) {
                int flag, u, v; unsigned int w;
                fscanf(fdiff, "%d %d %d %u\n", &flag, &u, &v, &w);
                H_u[i] = u; H_v[i] = v; H_w[i] = w;
                adj[u].push_back({v, w});
            }

            int *D_u, *D_v; unsigned int *D_w;
            gpuErrchk(cudaMalloc(&D_u, sizeof(int) * line));
            gpuErrchk(cudaMalloc(&D_v, sizeof(int) * line));
            gpuErrchk(cudaMalloc(&D_w, sizeof(unsigned int) * line));
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

            auto start_update = std::chrono::high_resolution_clock::now();
            
            gpuErrchk(cudaMemset(D_nVFlag, -1, sizeof(int) * N));
            Updating<<<(line + numThreads - 1) / numThreads, numThreads>>>(D_u, D_v, D_w, D_hx, D_nVFlag, N, line);
            
            gpuErrchk(cudaMemcpy(D_nV_size, H_a0, sizeof(int), cudaMemcpyHostToDevice));
            setNV<<<N_numBlocks, numThreads>>>(D_nVFlag, D_nV, D_nV_size, N);
            
            gpuErrchk(cudaMemset(D_PQ_size, 0, sizeof(int) * K));
            gpuErrchk(cudaMemset(D_openList, -1, sizeof(int) * N));
            insertPQ<<<numBlocks, numThreads>>>(D_PQ_size, D_nV, D_nV_size, K, N, D_openList);

            *H_flagEnd = 0; flag_PQ_not_empty = 1;
            while (*H_flagEnd == 0 && flag_PQ_not_empty == 1) {
                extractMin<<<numBlocks, numThreads>>>(D_PQ_size, D_expandNodes, D_expandNodes_size, D_openList, N, K);
                A_star_expand<<<numBlocks, numThreads>>>(D_offset, D_edges, D_weight, D_hx,
                                                         D_expandNodes, D_expandNodes_size, D_flagfound, D_openList,
                                                         N, E, K, endNode, D_nVFlag, D_PQ_size,
                                                         1, D_diff_offset, D_diff_edges, D_diff_weight, line);
                keepHeapPQ<<<numBlocks, numThreads>>>(D_PQ_size, N, K);
                setNV<<<N_numBlocks, numThreads>>>(D_nVFlag, D_nV, D_nV_size, N);
                insertPQ<<<numBlocks, numThreads>>>(D_PQ_size, D_nV, D_nV_size, K, N, D_openList);
                
                gpuErrchk(cudaMemcpy(H_flagfound, D_flagfound, sizeof(int), cudaMemcpyDeviceToHost));
                gpuErrchk(cudaMemcpy(H_PQ_size, D_PQ_size, sizeof(int) * K, cudaMemcpyDeviceToHost));
                gpuErrchk(cudaMemset(D_nVFlag, -1, sizeof(int) * N));
                gpuErrchk(cudaMemcpy(D_nV_size, H_a0, sizeof(int), cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemcpy(D_expandNodes_size, H_a0, sizeof(int), cudaMemcpyHostToDevice));

                flag_PQ_not_empty = 0;
                for (int i = 0; i < K; i++) { if (H_PQ_size[i] > 0) { flag_PQ_not_empty = 1; break; } }
                if (*H_flagfound == 1 && flag_PQ_not_empty == 1) {
                    gpuErrchk(cudaMemcpy(D_flagEnd, H_flagfound, sizeof(int), cudaMemcpyHostToDevice));
                    checkMIN<<<numBlocks, numThreads>>>(D_PQ_size, D_flagEnd, endNode, N, K);
                    gpuErrchk(cudaMemcpy(H_flagEnd, D_flagEnd, sizeof(int), cudaMemcpyDeviceToHost));
                }
            }

            getCx<<<1, 1>>>(endNode, D_dest_cost);
            gpuErrchk(cudaMemcpy(H_dest_cost, D_dest_cost, sizeof(int), cudaMemcpyDeviceToHost));
            auto end_update = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff_update = end_update - start_update;
            printf("Time taken for static A*: %f seconds\n", diff_update.count());
            printf("Cost on static Graph : %d\n", *H_dest_cost);

            gpuErrchk(cudaFree(D_u)); gpuErrchk(cudaFree(D_v)); gpuErrchk(cudaFree(D_w));
            gpuErrchk(cudaFree(D_diff_edges)); gpuErrchk(cudaFree(D_diff_offset)); gpuErrchk(cudaFree(D_diff_weight));
            free(H_u); free(H_v); free(H_w); free(H_diff_edges); free(H_diff_offset); free(H_diff_weight);
        }
        fclose(fdiff);
    }

    return 0;
}