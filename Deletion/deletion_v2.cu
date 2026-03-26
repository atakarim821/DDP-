#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <iostream>
#include <chrono>

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
__device__ volatile int PQ[MAX_NODE];

__device__ __host__ inline unsigned long long pack(unsigned int cost, int parent) {
    return ((unsigned long long)cost << 32) | (unsigned int)parent;
}

__device__ __host__ inline unsigned int unpackCost(unsigned long long val) {
    return (unsigned int)(val >> 32);
}

__device__ __host__ inline int unpackParent(unsigned long long val) {
    return (int)(val & 0xFFFFFFFF);
}

__global__ void extractMin(int* PQ_size, int* expandNodes,int* expandNodes_size,int* openList,int N,int K){
    
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    
    if(id<K && PQ_size[id]>0){

        int front = id* ( (N+K-1)/K );
        int node = PQ[front];

        // Defend against corrupted ghost nodes crashing the GPU
        if (node < 0 || node >= N) {
            return;
        }

        PQ[front]=PQ[front+PQ_size[id]-1];
        PQ_size[id]-=1;
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

        int len = atomicAdd(expandNodes_size,1);
        expandNodes[len]=node;
    }

} 

__global__ void A_star_expand(int* off,int* edge,unsigned int* W,int* Hx,int* parent,
    int* expandNodes,int* expandNodes_size, int* lock ,int* flagfound,int* openList,
    int N,int E, int K,int dest,int* nVFlag,int* PQ_size,
    int flagDiff,int* diff_off,int* diff_edge,unsigned int* diff_weight,int dE ){

    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int warp_id = id / 32;
    unsigned int lane_id = id % 32;
    
    if(warp_id < *expandNodes_size ){

        int node = expandNodes[warp_id];
        
        if(lane_id == 0 && node == dest){
            atomicOr(flagfound,1);
           if(DEBUG)
                printf("found %d\n",id);
        }

        unsigned long long nodeData = Cx[node];
        unsigned int node_f_cost = unpackCost(nodeData);
        unsigned int node_g_cost = node_f_cost - Hx[node];

        int start = off[node];
        int end = (node != N - 1) ? off[node + 1] : E;

        for (int i = start + lane_id; i < end; i += 32) {
            int child = edge[i];
            if (child < 0) continue;

            unsigned int weight = W[i];
            unsigned int new_f_cost = node_g_cost + weight + Hx[child];
            unsigned long long new_val = pack(new_f_cost, node);

            unsigned long long old_val = atomicMin((unsigned long long*)&Cx[child], new_val);

            if (new_val < old_val) {
                if (openList[child] == -1) {
                    nVFlag[child] = 1;
                }
            }
        }

        if (flagDiff) {
            start = diff_off[node];
            end = (node != N - 1) ? diff_off[node + 1] : dE;

            for (int i = start + lane_id; i < end; i += 32) {
                int child = diff_edge[i];
                if (child < 0) continue;

                unsigned int weight = diff_weight[i];
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

__global__ void keepHeapPQ(int* PQ_size, int N,int K){
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id < K && PQ_size[id] > 0){
        int front  = id*( (N+K-1)/K );
        int size = PQ_size[id];
        for (int start = (size - 2) / 2; start >= 0; start--)
        {
            int current = start;
            while (true)
            {
                int left = 2 * current + 1;
                int right = 2 * current + 2;
                int smallest = current;
                if (left < size && unpackCost(Cx[PQ[front + left]]) < unpackCost(Cx[PQ[front + smallest]]))
                    smallest = left;
                if (right < size && unpackCost(Cx[PQ[front + right]]) < unpackCost(Cx[PQ[front + smallest]]))
                    smallest = right;
                if (smallest != current)
                {
                    unsigned int swap = PQ[front + current];
                    PQ[front + current] = PQ[front + smallest];
                    PQ[front + smallest] = swap;
                    current = smallest;
                }
                else
                {
                    break;
                }
            }
        }
    }
}

__global__ void setNV(int* nextFlag,int* nextV,int* nvSize,int N){
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id < N){
        if(nextFlag[id]==1){
            int index = atomicAdd(nvSize,1);
            nextV[index]=id;
        }
    }
}

__global__ void insertPQ(int* PQS,int* nextV,int* nVsize,int K,int N,int* openList){
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id < K){
        int front = id*( (N+K-1)/K );
        int i = id;
        while(i<*nVsize){
            
            if(openList[nextV[i]]!=-1){
                i+=K;
                continue;
            }

            PQ[front+PQS[id]]= nextV[i];
            PQS[id]+=1;

            openList[nextV[i]] = id;

            if(PQS[id] > 1){
                int index = PQS[id] - 1;
                while(index > 0){
                    int parentIdx = (index - 1) / 2;
                    if(unpackCost(Cx[PQ[front + parentIdx]]) > unpackCost(Cx[PQ[front + index]])){
                        int swap = PQ[front + index];
                        PQ[front + index] = PQ[front + parentIdx];
                        PQ[front + parentIdx] = swap;
                        index = parentIdx;
                    }
                    else
                        break;
                }
            }
            i += K;
        }
    }
}

__global__ void checkMIN(int* PQ_size,int* flagEnd,int dest,int N,int K){
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    
    if(id < K && PQ_size[id] > 0 ){
        int front = id* ( (N+K-1)/K );
        int node = PQ[front];
        if(unpackCost(Cx[dest]) > unpackCost(Cx[node])) {
            atomicAnd(flagEnd,0);
        }
    }
}

__global__ void getCx(int dest,int* val){
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id==0){
        *val = unpackCost(Cx[dest]);
    }
}

void build_reverse_csr(int N, int E,
                       const int *H_offset,
                       const int *H_edges,
                       const unsigned int *H_weight,
                       int *r_H_offset,          
                       int *r_H_edges,           
                       unsigned int *r_H_weight) 
{
    int *indeg = (int *)calloc(N, sizeof(int));
    for (int u = 0; u < N; u++)
    {
        int start = H_offset[u];
        int end = (u == N - 1) ? E : H_offset[u + 1];
        for (int e = start; e < end; e++)
        {
            int v = H_edges[e];
            indeg[v]++;
        }
    }

    r_H_offset[0] = 0;
    for (int i = 1; i < N; i++)
    {
        r_H_offset[i] = r_H_offset[i - 1] + indeg[i - 1];
    }

    int *counter = (int *)calloc(N, sizeof(int));
    for (int u = 0; u < N; u++)
    {
        int start = H_offset[u];
        int end = (u == N - 1) ? E : H_offset[u + 1];
        for (int e = start; e < end; e++)
        {
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

__global__ void markAsDeleted(int *u, int *v, int *offset, int *edges, unsigned int *weight, int N, int E, int dE)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < dE)
    {
        int from = u[id];
        int to = v[id];
        
        if (from < 0 || from >= N || to < 0 || to >= N) return;
        
        int start = offset[from];
        int end = (from == N - 1) ? E : offset[from + 1];
        if(start >= end) return; 
        for(int i = start; i < end; i++){
            if(edges[i] == to){
                weight[i] = INT_MAX;
                return;
            }
        }
    }   
}

__global__ void markAsAffected(int* u, int *v, int *affected,int* parent ,int N, int dE)
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

__global__ void markDescendent(int *edges, int *offset, unsigned int *weight, int *parent,
                               int *curr_worklist, int *curr_sz, int *new_worklist,
                               int *new_sz, int *affected, int* lock ,int N, int E)
{
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int warp_id = id / 32;
    unsigned int lane_id = id % 32;
    if (warp_id < *curr_sz)
    {
        int node = curr_worklist[warp_id];
        int start = offset[node];
        int end = (node == N - 1) ? E : offset[node + 1];
        for (int i = start + lane_id; i < end; i += 32)
        {
            int child = edges[i];
            if (child < 0 || child >= N || affected[child] == 1)
                continue;
            bool leave = false;
            while(!leave){
                if(atomicCAS(&lock[child],0,1)==0){
                    if (unpackParent(Cx[child]) == node)
                    {
                        affected[child] = 1;
                        Cx[child] = pack(INF_COST, -1); 
                        int index = atomicAdd(new_sz, 1);
                        new_worklist[index] = child;
                    }
                    leave = true;
                    atomicCAS(&lock[child],1,0);
                }
            }
        }
    }
}

__global__ void Update_weights(int *affected, int *parent, int *offset, int *edges, unsigned int *weight,
        int *hx, int *flag,  int N, int E){
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int warp_id = id / 32;
    unsigned int lane_id = id % 32;
    if (warp_id < N)
    {
        if(affected[warp_id] == 0) return;
        int start = offset[warp_id];
        int end = (warp_id == N-1) ? E : offset[warp_id+1];
        unsigned int minCost = INF_COST;
        int minParent = -1;
        for (int i = start + lane_id; i < end; i += 32)
        {
            int parentNode = edges[i];
            unsigned int parentCost = unpackCost(Cx[parentNode]);
            if (parentCost == INF_COST || weight[i] == INT_MAX)
                continue; 
            unsigned int cost = (parentCost - hx[parentNode]) + weight[i] + hx[warp_id];
            if (cost < minCost)
            {
                minCost = cost;
                minParent = parentNode;
            }
        }
        
        for (int offset_shfl = 16; offset_shfl > 0; offset_shfl /= 2) {
            unsigned int other_minCost = __shfl_down_sync(0xffffffff, minCost, offset_shfl);
            int other_minParent = __shfl_down_sync(0xffffffff, minParent, offset_shfl);
            if (other_minCost < minCost) {
                minCost = other_minCost;
                minParent = other_minParent;
            }
        }
        
        if (lane_id == 0) {
            Cx[warp_id] = pack(minCost, minParent);
            if(minParent != -1){
                flag[warp_id] = 1;
            }
        }
    }
}

__global__ void InitializeWorklist(int *affected, int *worklist, int *worklist_size, int N)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < N)
    {
        if (affected[id] == 1)
        {
            int index = atomicAdd(worklist_size, 1);
            worklist[index] = id;
        }
    }
}

int main(int argc, char *argv[]){

    auto start_wall_static = std::chrono::high_resolution_clock::now(); 
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " startNode endNode\n";
        return 1;
    }
    int K = 10000;
    int startNode = std::stoi(argv[1]);
    int endNode = std::stoi(argv[2]);

    FILE* fgraph = fopen("graph.txt","r");
    if (!fgraph) {
        std::cerr << "Failed to open graph.txt\n";
        return 1;
    }

    int N,E;
    fscanf(fgraph,"%d %d\n",&N,&E);

    if (startNode < 0 || startNode >= N || endNode < 0 || endNode >= N) {
        std::cerr << "FATAL: Start or end node out of bounds! Must be between 0 and " << N - 1 << ".\n";
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

    int* H_PQ = (int*)malloc(sizeof(int)*N);
    int* H_openList = (int*)malloc(sizeof(int)*N);
    int* H_PQ_size = (int*)malloc(sizeof(int)*K);

    int* H_dest_cost = (int*)malloc(sizeof(int));
    
    memset(H_PQ_size,0,sizeof(int)*K);
    memset(H_parent,-1,sizeof(int)*N);
    memset(H_openList,-1,sizeof(int)*N);

    for(int i=0;i<N;i++){
        H_cx[i]=pack(INF_COST, -1);
    }

    for(int i=0;i<E;i++){
        fscanf(fgraph,"%d",&H_edges[i]);
    }

    for(int i=0;i<N;i++){
        fscanf(fgraph,"%d",&H_offset[i]);
    }

    for(int i=0;i<E;i++){
        fscanf(fgraph,"%u",&H_weight[i]);
    }

    build_reverse_csr(N, E, H_offset, H_edges, H_weight,
                      r_H_offset, r_H_edges, r_H_weight);

    FILE* fhx = fopen("Hx.txt","r");
    if (fhx) {
        for(int i=0;i<N;i++){
            H_hx[i] = 0;
        }
        fclose(fhx);
    } else {
        for(int i=0;i<N;i++) H_hx[i] = 0;
    }
    fclose(fgraph);

    int* H_flagEnd = (int*)malloc(sizeof(int));
    int* H_flagfound = (int*)malloc(sizeof(int));
    int* H_a0 = (int*)malloc(sizeof(int));

    int* H_nV_size = (int*)malloc(sizeof(int));

    int* H_nVFlag = (int*)malloc(sizeof(int)*N);
    memset(H_nVFlag,-1,sizeof(int)*N);   

    *H_flagEnd = 0;
    *H_flagfound = 0;
    *H_a0 = 0;

    H_cx[startNode]=pack(H_hx[startNode], -1);
    H_PQ[0]=startNode;
    H_PQ_size[0]=1;
    H_openList[startNode]=0;

    int* D_offset;
    int* D_edges ;
    unsigned int* D_weight;

    int *r_D_offset;
    int *r_D_edges;
    unsigned int *r_D_weight;
    
    int* D_hx;
    int* D_parent;
    
    int* D_PQ_size;

    int* D_openList;
    int* D_lock;

    int* D_diff_edges = nullptr;
    int* D_diff_offset = nullptr;
    unsigned int* D_diff_weight = nullptr;

    int* D_nVFlag;
    int* D_nV;
    int* D_nV_size;
    
    int* D_expandNodes;
    int* D_expandNodes_size;
    
    int* D_flagEnd;
    int* D_flagfound;

    int* D_dest_cost;

    gpuErrchk ( cudaMalloc(&D_offset,sizeof(int)*N) );
    gpuErrchk ( cudaMalloc(&D_edges,sizeof(int)*E) );
    gpuErrchk ( cudaMalloc(&D_weight,sizeof(unsigned int)*E) );

    gpuErrchk ( cudaMalloc(&r_D_offset,sizeof(int)*N) );
    gpuErrchk ( cudaMalloc(&r_D_edges,sizeof(int)*E) );
    gpuErrchk ( cudaMalloc(&r_D_weight,sizeof(unsigned int)*E) );
    
    gpuErrchk ( cudaMalloc(&D_hx,sizeof(int)*N) );
    gpuErrchk ( cudaMalloc(&D_parent,sizeof(int)*N) );
   
    gpuErrchk ( cudaMalloc(&D_PQ_size,sizeof(int)*K) );
    
    gpuErrchk ( cudaMalloc(&D_openList,sizeof(int)*N) );
    gpuErrchk ( cudaMalloc(&D_lock,sizeof(int)*N) );

    gpuErrchk ( cudaMalloc(&D_dest_cost,sizeof(int)) );

    gpuErrchk ( cudaMalloc(&D_nV,sizeof(int)*N) );
    gpuErrchk ( cudaMalloc(&D_nV_size,sizeof(int)) );
    gpuErrchk ( cudaMalloc(&D_nVFlag,sizeof(int)*N) );

    gpuErrchk ( cudaMalloc(&D_expandNodes,sizeof(int)*K) );  
    gpuErrchk ( cudaMalloc(&D_expandNodes_size,sizeof(int)) );

    gpuErrchk( cudaMalloc(&D_flagEnd,sizeof(int)) );
    gpuErrchk( cudaMalloc(&D_flagfound,sizeof(int)) );
   
    gpuErrchk ( cudaMemcpy(D_offset,H_offset,sizeof(int)*N,cudaMemcpyHostToDevice) );
    gpuErrchk ( cudaMemcpy(D_edges,H_edges,sizeof(int)*E,cudaMemcpyHostToDevice) );
    gpuErrchk ( cudaMemcpy(D_weight,H_weight,sizeof(unsigned int)*E,cudaMemcpyHostToDevice) );

    gpuErrchk ( cudaMemcpy(r_D_offset,r_H_offset,sizeof(int)*N,cudaMemcpyHostToDevice) );
    gpuErrchk ( cudaMemcpy(r_D_edges,r_H_edges,sizeof(int)*E,cudaMemcpyHostToDevice) );
    gpuErrchk ( cudaMemcpy(r_D_weight,r_H_weight,sizeof(unsigned int)*E,cudaMemcpyHostToDevice) ); 

    gpuErrchk ( cudaMemcpy(D_hx,H_hx,sizeof(int)*N,cudaMemcpyHostToDevice) );
    gpuErrchk ( cudaMemcpy(D_parent,H_parent,sizeof(int)*N,cudaMemcpyHostToDevice) );
    gpuErrchk ( cudaMemcpy(D_openList,H_openList,sizeof(int)*N,cudaMemcpyHostToDevice) );

    gpuErrchk ( cudaMemcpy(D_PQ_size,H_PQ_size,sizeof(int)*K,cudaMemcpyHostToDevice) );

    gpuErrchk ( cudaMemcpyToSymbol(Cx,H_cx, sizeof(unsigned long long)*N, 0, cudaMemcpyHostToDevice) );
    gpuErrchk ( cudaMemcpyToSymbol(PQ,H_PQ, sizeof(int)*N, 0, cudaMemcpyHostToDevice) );

    gpuErrchk ( cudaMemcpy(D_flagEnd,H_flagEnd,sizeof(int),cudaMemcpyHostToDevice) );
    gpuErrchk ( cudaMemcpy(D_flagfound,H_flagfound,sizeof(int),cudaMemcpyHostToDevice) );
   
    gpuErrchk ( cudaMemcpy(D_nVFlag,H_nVFlag,sizeof(int)*N,cudaMemcpyHostToDevice) );

    gpuErrchk ( cudaMemcpy(D_nV_size,H_a0,sizeof(int),cudaMemcpyHostToDevice) );
    gpuErrchk ( cudaMemcpy(D_expandNodes_size,H_a0,sizeof(int),cudaMemcpyHostToDevice) );

    gpuErrchk ( cudaMemset(D_lock,0,sizeof(int)*N) );

    int flag_PQ_not_empty = 0;
    for(int i=0;i<K;i++){
        if(H_PQ_size[i]>0)
            flag_PQ_not_empty=1;
    }

    int numThreads = 512;
    int numBlocks = (32*K+numThreads-1)/numThreads;
    int N_numBlocks = (N+numThreads-1)/numThreads;
    
    // ── STATIC PHASE CUDA EVENT SETUP ────────────────────────
    cudaEvent_t t_static_start, t_static_stop;
    cudaEventCreate(&t_static_start);
    cudaEventCreate(&t_static_stop);

    cudaEventRecord(t_static_start);

    while(*H_flagEnd==0 && flag_PQ_not_empty==1){

        extractMin<<<numBlocks,numThreads>>>(D_PQ_size, D_expandNodes,D_expandNodes_size,D_openList,N,K);
        gpuErrchk(cudaPeekAtLastError() );
        cudaDeviceSynchronize();
        
        A_star_expand<<<numBlocks,numThreads>>>(D_offset,D_edges,D_weight,D_hx,D_parent,
            D_expandNodes,D_expandNodes_size, D_lock ,D_flagfound,D_openList,
            N,E,K,endNode,D_nVFlag,D_PQ_size,
            false,D_diff_offset,D_diff_edges,D_diff_weight,0);
        gpuErrchk(cudaPeekAtLastError() );
        cudaDeviceSynchronize();

        keepHeapPQ<<<numBlocks,numThreads>>>(D_PQ_size,N,K);
        gpuErrchk(cudaPeekAtLastError() );
        cudaDeviceSynchronize();
        
        setNV<<<N_numBlocks,numThreads>>>(D_nVFlag,D_nV,D_nV_size,N);
        gpuErrchk(cudaPeekAtLastError() );
        cudaDeviceSynchronize();
        
        insertPQ<<<numBlocks,numThreads>>>(D_PQ_size,D_nV,D_nV_size,K,N,D_openList);
        gpuErrchk(cudaPeekAtLastError() );
        cudaDeviceSynchronize();
        
        gpuErrchk( cudaMemcpy(H_PQ_size,D_PQ_size, sizeof(int)*K,cudaMemcpyDeviceToHost) );
        gpuErrchk( cudaMemcpy(D_nVFlag,H_nVFlag,sizeof(int)*N,cudaMemcpyHostToDevice) );
        gpuErrchk( cudaMemcpy(D_nV_size,H_a0,sizeof(int),cudaMemcpyHostToDevice) );
        gpuErrchk( cudaMemcpy(D_expandNodes_size,H_a0,sizeof(int),cudaMemcpyHostToDevice) );
        
        flag_PQ_not_empty = 0;
        for(int i=0;i<K;i++){
            if(H_PQ_size[i]>0)
                flag_PQ_not_empty=1;
        }
    }

    cudaEventRecord(t_static_stop);
    cudaEventSynchronize(t_static_stop);
    
    float static_ms = 0;
    cudaEventElapsedTime(&static_ms, t_static_start, t_static_stop);

    getCx<<<1,1>>>(endNode,D_dest_cost);
    gpuErrchk( cudaMemcpy(H_dest_cost,D_dest_cost, sizeof(int),cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(H_parent,D_parent, sizeof(int)*N,cudaMemcpyDeviceToHost) );
    gpuErrchk(cudaDeviceSynchronize());
    
    auto end_wall_static = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff_wall_static = end_wall_static - start_wall_static;
    
    printf("\n--- STATIC PHASE RESULTS ---\n");
    printf("Static A* (Wall Clock - CPU + Memcpy + GPU): %.3f ms\n", diff_wall_static.count() * 1000.0); 
    printf("Static A* (Pure GPU Execution Time)        : %.3f ms\n", static_ms);
    printf("Cost on static Graph                       : %d\n", *H_dest_cost);

    FILE* fdiff = fopen("Updates.txt","r");
    if (!fdiff) {
        std::cerr << "Failed to open Updates.txt\n";
        return 1;
    }

    // ── DYNAMIC PHASE CUDA EVENT SETUP ───────────────────────
    cudaEvent_t t_dyn_start, t_dyn_stop;
    cudaEventCreate(&t_dyn_start);
    cudaEventCreate(&t_dyn_stop);

    int line;
    while (fscanf(fdiff,"%d",&line)!=EOF)
    {
        int *H_u = (int *)malloc(sizeof(int) * line);
        int *H_v = (int *)malloc(sizeof(int) * line);

        for (size_t i = 0; i < line; i++)
        {
            int flag, wt;
            fscanf(fdiff,"%d %d %d %d",&flag,&H_u[i],&H_v[i], &wt);
        }
       
        int* D_u;
        int* D_v;
        gpuErrchk ( cudaMalloc(&D_u,sizeof(int)*line) );
        gpuErrchk ( cudaMalloc(&D_v,sizeof(int)*line) );
        gpuErrchk ( cudaMemcpy(D_u,H_u,sizeof(int)*line,cudaMemcpyHostToDevice) );
        gpuErrchk ( cudaMemcpy(D_v,H_v,sizeof(int)*line,cudaMemcpyHostToDevice) );

        int *D_affected;
        gpuErrchk ( cudaMalloc(&D_affected,sizeof(int)*N) );
        gpuErrchk(cudaMemset(D_affected, 0, sizeof(int) * N));

        auto start_wall_dyn = std::chrono::high_resolution_clock::now();

        int threadsCnt = 512;
        
        // Start GPU pure execution timing
        cudaEventRecord(t_dyn_start);

        markAsDeleted<<< (line+threadsCnt-1)/threadsCnt , threadsCnt >>>(D_u, D_v, D_offset, D_edges, D_weight, N, E, line);
        markAsDeleted<<< (line+threadsCnt-1)/threadsCnt , threadsCnt >>>(D_v, D_u, r_D_offset, r_D_edges, r_D_weight, N, E, line);
        markAsAffected<<<(line + threadsCnt - 1) / threadsCnt, threadsCnt>>>(D_u,D_v, D_affected, D_parent, N, line);
        gpuErrchk(cudaPeekAtLastError() );
        gpuErrchk(cudaDeviceSynchronize());

        int* D_curr_worklist; gpuErrchk ( cudaMalloc(&D_curr_worklist,sizeof(int)*N) );
        int* D_new_worklist; gpuErrchk ( cudaMalloc(&D_new_worklist,sizeof(int)*N) );
        int* D_curr_sz; gpuErrchk ( cudaMalloc(&D_curr_sz,sizeof(int)) );
        int* D_new_sz; gpuErrchk ( cudaMalloc(&D_new_sz,sizeof(int)) );
        gpuErrchk(cudaMemset(D_curr_sz, 0, sizeof(int)));
        gpuErrchk(cudaMemset(D_new_sz, 0, sizeof(int)));
        gpuErrchk(cudaMemset(D_new_worklist, 0, sizeof(int) * N));
        gpuErrchk(cudaMemset(D_curr_worklist, 0, sizeof(int) * N));

        InitializeWorklist<<<(N + threadsCnt - 1) / threadsCnt, threadsCnt>>>(D_affected, D_curr_worklist, D_curr_sz, N);
        int WorkList_size = 0;
        gpuErrchk(cudaMemcpy(&WorkList_size, D_curr_sz, sizeof(int), cudaMemcpyDeviceToHost));
        
        gpuErrchk(cudaMemset(D_lock, 0, sizeof(int) * N));
        while (WorkList_size)
        {
            long long WL_warp = (long long)WorkList_size * 32;
            markDescendent<<<(WL_warp + threadsCnt - 1) / threadsCnt, threadsCnt>>>(D_edges, D_offset, D_weight, D_parent,
                                                                                          D_curr_worklist, D_curr_sz, D_new_worklist,
                                                                                          D_new_sz, D_affected, D_lock ,N, E);
            gpuErrchk(cudaPeekAtLastError());
            gpuErrchk(cudaDeviceSynchronize());

            gpuErrchk(cudaMemset(D_curr_worklist, 0, sizeof(int) * N));
            gpuErrchk(cudaMemset(D_curr_sz, 0, sizeof(int)));

            gpuErrchk(cudaMemcpy(D_curr_worklist, D_new_worklist, sizeof(int) * N, cudaMemcpyDeviceToDevice));
            gpuErrchk(cudaMemcpy(D_curr_sz, D_new_sz, sizeof(int), cudaMemcpyDeviceToDevice));

            gpuErrchk(cudaMemset(D_new_sz, 0, sizeof(int)));
            gpuErrchk(cudaMemset(D_new_worklist, 0, sizeof(int)));
            gpuErrchk(cudaMemcpy(&WorkList_size, D_curr_sz, sizeof(int), cudaMemcpyDeviceToHost));
        }

        int *flag;
        gpuErrchk(cudaMalloc(&flag, N * sizeof(int)));
        gpuErrchk(cudaMemset(flag, 0, N * sizeof(int)));
        int* sz;
        gpuErrchk(cudaMalloc(&sz, sizeof(int)));
        gpuErrchk(cudaMemset(sz, 0, sizeof(int)));  

        long long N_warp = (long long)N * 32;
        Update_weights<<<(N_warp + threadsCnt - 1) / threadsCnt, threadsCnt>>>(D_affected, D_parent, r_D_offset, r_D_edges, r_D_weight,
                                                                          D_hx, flag, N, E);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        keepHeapPQ<<<numBlocks,numThreads>>>(D_PQ_size,N,K);

        setNV<<<(N + threadsCnt - 1) / threadsCnt, threadsCnt>>>(flag, D_nV, sz, N);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        insertPQ<<<numBlocks, numThreads>>>(D_PQ_size, D_nV, sz, K, N, D_openList);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        gpuErrchk(cudaMemset(D_lock, 0, sizeof(int) * N));
        gpuErrchk(cudaMemset(D_flagfound, 0, sizeof(int)));
        gpuErrchk(cudaMemcpy(H_PQ_size, D_PQ_size, sizeof(int) * K, cudaMemcpyDeviceToHost));
        *H_flagEnd = 0;
        int flag_PQ_not_empty = 0;
        for (int i = 0; i < K; i++)
        {
            if (H_PQ_size[i] > 0){
                flag_PQ_not_empty = 1;
                break;
            }
        }

        while (*H_flagEnd == 0 && flag_PQ_not_empty == 1)
        {
            extractMin<<<numBlocks, numThreads>>>(D_PQ_size, D_expandNodes, D_expandNodes_size, D_openList, N, K);
            gpuErrchk(cudaPeekAtLastError());
            gpuErrchk(cudaDeviceSynchronize());

            A_star_expand<<<numBlocks, numThreads>>>(D_offset, D_edges, D_weight, D_hx, D_parent,
                                                     D_expandNodes, D_expandNodes_size, D_lock, D_flagfound, D_openList,
                                                     N, E, K, endNode, D_nVFlag, D_PQ_size,
                                                     false, D_diff_offset, D_diff_edges, D_diff_weight, 0);
            gpuErrchk(cudaPeekAtLastError());
            gpuErrchk(cudaDeviceSynchronize());

            keepHeapPQ<<<numBlocks, numThreads>>>(D_PQ_size, N, K);
            gpuErrchk(cudaPeekAtLastError());
            gpuErrchk(cudaDeviceSynchronize());

            setNV<<<N_numBlocks, numThreads>>>(D_nVFlag, D_nV, D_nV_size, N);
            gpuErrchk(cudaPeekAtLastError());
            gpuErrchk(cudaDeviceSynchronize());

            insertPQ<<<numBlocks, numThreads>>>(D_PQ_size, D_nV, D_nV_size, K, N, D_openList);
            gpuErrchk(cudaPeekAtLastError());
            gpuErrchk(cudaDeviceSynchronize());

            gpuErrchk(cudaMemcpy(H_flagfound, D_flagfound, sizeof(int), cudaMemcpyDeviceToHost));
            gpuErrchk(cudaMemcpy(H_PQ_size, D_PQ_size, sizeof(int) * K, cudaMemcpyDeviceToHost));

            gpuErrchk(cudaMemcpy(D_nVFlag, H_nVFlag, sizeof(int) * N, cudaMemcpyHostToDevice));

            gpuErrchk(cudaMemcpy(D_nV_size, H_a0, sizeof(int), cudaMemcpyHostToDevice));
            gpuErrchk(cudaMemcpy(D_expandNodes_size, H_a0, sizeof(int), cudaMemcpyHostToDevice));

            flag_PQ_not_empty = 0;
            for (int i = 0; i < K; i++)
            {
                if (H_PQ_size[i] > 0){
                    flag_PQ_not_empty = 1;
                    break;
                }
            }

            if (*H_flagfound == 1 && flag_PQ_not_empty == 1)
            {
                gpuErrchk(cudaMemcpy(D_flagEnd, H_flagfound, sizeof(int), cudaMemcpyHostToDevice));

                checkMIN<<<numBlocks, numThreads>>>(D_PQ_size, D_flagEnd, endNode, N, K);

                gpuErrchk(cudaPeekAtLastError());
                gpuErrchk(cudaDeviceSynchronize());
                gpuErrchk(cudaMemcpy(H_flagEnd, D_flagEnd, sizeof(int), cudaMemcpyDeviceToHost));
            }
        }
        
        cudaEventRecord(t_dyn_stop);
        cudaEventSynchronize(t_dyn_stop);
        
        float dyn_ms = 0;
        cudaEventElapsedTime(&dyn_ms, t_dyn_start, t_dyn_stop);

        getCx<<<1, 1>>>(endNode, D_dest_cost);
        gpuErrchk(cudaMemcpy(H_dest_cost, D_dest_cost, sizeof(int), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(H_parent, D_parent, sizeof(int) * N, cudaMemcpyDeviceToHost));
        gpuErrchk(cudaDeviceSynchronize());
        
        auto end_wall_dyn = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff_wall_dyn = end_wall_dyn - start_wall_dyn;
        
        printf("\n--- DYNAMIC UPDATE RESULTS ---\n");
        printf("Dynamic Update (Wall Clock - CPU + GPU) : %.3f ms\n", diff_wall_dyn.count() * 1000.0); 
        printf("Dynamic Update (Pure GPU Execution Time): %.3f ms\n", dyn_ms);
        printf("Cost after deletion                     : %d\n", *H_dest_cost);

        free(H_u);
        free(H_v);  
        cudaFree(D_u);
        cudaFree(D_v);
        cudaFree(D_affected);
        cudaFree(D_curr_worklist);
        cudaFree(D_new_worklist);
        cudaFree(D_curr_sz);
        cudaFree(D_new_sz);
        cudaFree(flag);
        cudaFree(sz);
    }

    // Clean up
    fclose(fdiff);
    cudaEventDestroy(t_static_start);
    cudaEventDestroy(t_static_stop);
    cudaEventDestroy(t_dyn_start);
    cudaEventDestroy(t_dyn_stop);

    return 0;
}
