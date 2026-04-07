# GPU Dynamic Shortest Path — Optimization Report (v4 – v7)

## 1. Problem Statement

We maintain shortest paths on a large road-network graph (281,903 nodes, 2,312,497 edges) under a batch of ~1.4 million edge deletions. The algorithm has four phases executed entirely on the GPU:

1. **Deletion Marking** — set deleted edge weights to `INT_MAX` in both forward and reverse CSR.
2. **Affected-Node Identification** — mark nodes whose shortest-path-tree parent edge was deleted, then propagate to all descendants.
3. **Weight Update** — recompute costs for affected nodes using incoming (reverse) edges.
4. **A\* Re-expansion** — delta-stepping A\* to converge on new shortest paths.

All versions produce **identical results** (verified byte-identical Cx arrays across v3–v7). The endpoint cost transitions from 123 (static) to 416 (after deletion).

---

## 2. Performance Summary

| Version | Dynamic GPU Time | Speedup over v3 | Key Technique |
|---------|-----------------|------------------|---------------|
| **v3** (baseline) | 69.77 ms | 1.0x | Spin-lock descendants, scan-all Update\_weights |
| **v4** | 25.25 ms | 2.8x | Lock-free descendants, worklist-based Update\_weights |
| **v5** | 10.84 ms | 6.4x | Warp-cooperative deletion marking, 3-stream concurrency |
| **v6** | 2.07 ms | 33.7x | Binary-search deletion, sorted CSR, inline frontier append, DELTA\_DYN |
| **v7** | 1.41 ms | 49.5x | Cooperative-groups kernel fusion (persistent kernels) |

**Hardware**: NVIDIA A10 (Ampere, sm\_86, 72 SMs, 32 GB)

---

## 3. Baseline: v3

Before describing optimizations, we establish what v3 does so each improvement can be understood in context.

### 3.1 Kernel Inventory

| Kernel | Granularity | Purpose |
|--------|-------------|---------|
| `find_min_f` | 1 thread/node | Warp+block reduction to find minimum f-value in frontier |
| `expand_delta` | 1 warp/node | Expand frontier nodes with f &le; threshold; atomic Cx updates |
| `compact_frontier` | 1 thread/node | Scan N-sized flag array, atomicAdd into frontier |
| `getCx` | 1 thread | Read destination cost from device symbol |
| `markAsDeleted` | 1 thread/edge | **Linear scan** through source node's adjacency list to find and delete edge |
| `markAsAffected` | 1 thread/edge | If parent(v)==u, mark v as affected, set Cx\[v\]=INF |
| `markDescendent` | 1 warp/node | BFS propagation using **spin-locks** (`atomicCAS` acquire/release) |
| `Update_weights` | 1 warp/node | Scans **all N nodes** to recompute cost from reverse edges |
| `InitializeWorklist` | 1 thread/node | Compact affected array into worklist |

### 3.2 Dynamic Phase Host Loop

```
for each deletion batch:
    markAsDeleted(forward)          // sequential
    markAsDeleted(reverse)          // sequential
    markAsAffected(...)             // sequential
    
    InitializeWorklist(...)
    while worklist_size > 0:        // host loop
        markDescendent(...)         // spin-lock based
        cudaMemcpy D2D (worklist swap)
    
    Update_weights(all N nodes)     // scans every node
    compact_frontier(...)
    
    while frontier_size > 0:        // host loop
        find_min_f(...)
        cudaMemcpyFromSymbol(...)   // read min_f to host
        expand_delta(...)
        compact_frontier(...)
```

### 3.3 Key Bottlenecks

| Bottleneck | Impact |
|------------|--------|
| **Linear-scan deletion** (`markAsDeleted`) | O(degree) per deletion; ~95% of GPU time for high-degree nodes |
| **Spin-lock descendant marking** (`markDescendent`) | Heavy contention; threads spin-wait on `atomicCAS` locks |
| **Scan-all-N Update\_weights** | Processes 281,903 nodes even if only ~6,000 are affected |
| **cudaMemcpy D2D for worklist swap** | Copies entire N-element array every BFS iteration |
| **Sequential kernel launches** | Deletion marking kernels (fwd, rev, affected) launched one after another |
| **Host-device round-trips** | `cudaMemcpyFromSymbol` in every A\* iteration to read `min_f` and `frontier_size` |

---

## 4. Version 4 — Lock-Free Descendants & Worklist-Based Update

**Speedup: 2.8x over v3 (69.77 ms &rarr; 25.25 ms)**

### 4.1 Optimization 1: Lock-Free Descendant Marking

**Problem**: v3's `markDescendent` uses per-node spin-locks (`atomicCAS` acquire/release pattern) to prevent races when marking children as affected. Under high contention (many warps competing for the same child), threads waste cycles spinning.

**Solution**: Replace spin-locks with a single `atomicExch`:

```cuda
// v3: spin-lock pattern
while (atomicCAS(&lock[child], 0, 1) != 0);  // spin until acquired
if (unpackParent(Cx[child]) == node) {
    affected[child] = 1;
    Cx[child] = pack(INF_COST, -1);
    // add to worklist...
}
atomicExch(&lock[child], 0);  // release

// v4: lock-free pattern
int old = atomicExch(&affected[child], 1);    // mark + check in one atomic
if (old == 0) {                                // first thread wins
    Cx[child] = pack(INF_COST, -1);
    int index = atomicAdd(new_sz, 1);
    new_worklist[index] = child;
}
```

**Why it works**: `atomicExch` returns the previous value atomically. If `old == 0`, this thread was the first to mark the child — it proceeds with the invalidation. All other threads see `old == 1` and skip. No spinning, no lock array needed.

**Benefit**: Eliminates the entire `D_lock` array allocation, removes spin-wait contention, and guarantees each node is processed exactly once.

### 4.2 Optimization 2: Worklist-Based Weight Update

**Problem**: v3's `Update_weights` launches one warp per node for **all N = 281,903 nodes**, even though only ~6,176 nodes are affected. Over 97% of warps read `affected[node] == 0` and immediately return — wasted parallelism.

**Solution**: Compact affected nodes into a worklist first, then launch warps only for affected nodes:

```cuda
// v3: scan all N nodes
__global__ void Update_weights(..., int N) {
    int warp_id = ...;
    if (warp_id >= N) return;
    if (affected[warp_id] == 0) return;  // 97% of warps exit here
    // ... process node ...
}

// v4: worklist-based
InitializeWorklist(affected, worklist, &num_affected, N);
__global__ void Update_weights_worklist(worklist, num_affected, ...) {
    int warp_id = ...;
    if (warp_id >= num_affected) return;  // only ~6,176 warps launched
    int node = worklist[warp_id];
    // ... process node ...
}
```

**Benefit**: Launch grid shrinks from `ceil(32*281903/512) = 17,619` blocks to `ceil(32*6176/512) = 386` blocks. Dramatically reduces wasted thread-blocks and improves SM utilization.

### 4.3 Optimization 3: Pointer Swap Instead of D2D Memcpy

**Problem**: v3 copies the entire new worklist to the current worklist via `cudaMemcpy(..., cudaMemcpyDeviceToDevice)` every BFS iteration — an O(N) transfer for each level.

**Solution**: Swap the device pointers on the host side:

```cuda
// v3:
cudaMemcpy(D_curr_worklist, D_new_worklist, N * sizeof(int), cudaMemcpyDeviceToDevice);

// v4:
std::swap(D_curr_worklist, D_new_worklist);  // O(1) pointer swap
```

**Benefit**: Eliminates a device-to-device memcpy of up to 1.1 MB per BFS level.

### 4.4 Optimization 4: Host-Side Cx Read via `cudaMemcpyFromSymbol`

**Problem**: v3 launches a single-thread kernel `getCx` just to read the destination node's cost from the `__device__` symbol `Cx[]`.

**Solution**: Replace with a direct `cudaMemcpyFromSymbol` call:

```cuda
// v3:
getCx<<<1,1>>>(dest, D_val);
cudaMemcpy(&H_val, D_val, sizeof(int), cudaMemcpyDeviceToHost);

// v4:
unsigned long long cx_val;
cudaMemcpyFromSymbol(&cx_val, Cx, sizeof(unsigned long long),
                     dest * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
unsigned int cost = (unsigned int)(cx_val >> 32);
```

**Benefit**: Eliminates a kernel launch + device allocation; single API call reads the value directly.

---

## 5. Version 5 — Warp-Cooperative Deletion & Stream Concurrency

**Speedup: 6.4x over v3 (69.77 ms &rarr; 10.84 ms), 2.3x over v4**

### 5.1 Profiling Insight

Profiling v4 revealed that **`markAsDeleted` consumed ~95% of the dynamic phase GPU time**. Each thread linearly scans the adjacency list of the source node to find the target edge — O(degree) work per deletion. For high-degree nodes in road networks (degree > 100), this is a severe bottleneck.

### 5.2 Optimization 1: Warp-Cooperative Deletion Marking

**Problem**: One thread scans the entire adjacency list. For a node with degree 200, one thread does 200 comparisons while 31 other threads in its warp sit idle.

**Solution**: Assign one **warp** (32 threads) per deletion. All 32 lanes search the adjacency list in parallel using `__ballot_sync` for early exit:

```cuda
__global__ void markAsDeletedWarp(int *u, int *v, int *offset, int *edges,
                                   unsigned int *weight, int N, int E, int dE) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int warp_id = id / 32;
    unsigned int lane_id = id % 32;
    if (warp_id >= dE) return;

    int from = u[warp_id], to = v[warp_id];
    int start = offset[from];
    int end = (from == N - 1) ? E : offset[from + 1];
    int num_edges = end - start;

    for (int base = 0; base < num_edges; base += 32) {
        int idx = start + base + lane_id;
        bool is_match = (idx < end) && (edges[idx] == to);
        unsigned int match_mask = __ballot_sync(0xFFFFFFFF, is_match);
        if (match_mask) {
            int finder = __ffs(match_mask) - 1;
            if ((int)lane_id == finder)
                weight[idx] = INT_MAX;
            return;  // entire warp exits
        }
    }
}
```

**How it works**: Each iteration of the loop, 32 lanes check 32 consecutive edges simultaneously. `__ballot_sync` collects match results from all lanes into a bitmask. If any lane found the target, `__ffs` identifies which lane, and that lane marks the edge. The entire warp returns immediately — no unnecessary scanning.

**Benefit**: Reduces deletion marking from O(degree) serial work to O(degree/32) parallel work per warp. For the dominant bottleneck, this is a ~30x speedup.

### 5.3 Optimization 2: 3-Stream Concurrent Kernel Execution

**Problem**: v3/v4 launch deletion marking kernels sequentially:

```
markAsDeleted(forward)  →  markAsDeleted(reverse)  →  markAsAffected
```

Each kernel waits for the previous to finish, even though forward and reverse marking are independent.

**Solution**: Launch on three non-blocking CUDA streams:

```cuda
cudaStream_t s_fwd, s_rev, s_aff;
cudaStreamCreateWithFlags(&s_fwd, cudaStreamNonBlocking);
cudaStreamCreateWithFlags(&s_rev, cudaStreamNonBlocking);
cudaStreamCreateWithFlags(&s_aff, cudaStreamNonBlocking);

markAsDeletedWarp<<<blocks, threads, 0, s_fwd>>>(D_u, D_v, D_offset, ...);
markAsDeletedWarp<<<blocks, threads, 0, s_rev>>>(D_v, D_u, r_D_offset, ...);
markAsAffected<<<blocks, threads, 0, s_aff>>>(D_u, D_v, D_affected, ...);
cudaDeviceSynchronize();
```

**Benefit**: Forward marking, reverse marking, and affected identification overlap on the GPU's 72 SMs. The three kernels execute concurrently, reducing the critical path.

---

## 6. Version 6 — Binary-Search Deletion, Sorted CSR, Inline Append & DELTA\_DYN

**Speedup: 33.7x over v3 (69.77 ms &rarr; 2.07 ms), 5.2x over v5**

### 6.1 Optimization 1: Sorted CSR + Binary-Search Deletion Marking

**Problem**: v5's warp-cooperative linear search still does O(degree/32) work per deletion. Can we do better?

**Solution**: Pre-sort each node's adjacency list by destination ID on the host before uploading to the GPU. Then use O(log degree) binary search with a single thread per deletion:

```cuda
// Host preprocessing (one-time cost, not timed):
void sortCSR(int* offset, int* edges, unsigned int* weight, int N, int E) {
    for (int u = 0; u < N; u++) {
        int start = offset[u];
        int end = (u == N - 1) ? E : offset[u + 1];
        // sort edges[start..end) and weight[start..end) by edge destination
        // using index-based sort to keep weights aligned
    }
}

// GPU kernel (1 thread per deletion):
__global__ void markDeletedBinarySearch(int* u, int* v, int* offset, int* edges,
                                         unsigned int* weight, int N, int E, int dE) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= dE) return;

    int from = u[id], to = v[id];
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
```

**Tradeoff**: Uses only 1 thread per deletion (not 1 warp), which means 32x fewer threads. But O(log degree) is far less work per thread than O(degree/32). For a node with degree 256: binary search = 8 comparisons vs warp-linear = 8 iterations of 32-wide comparison. Binary search wins because each thread is fully utilized — no idle lanes.

**Benefit**: Deletion marking drops from the dominant cost to a negligible fraction. Applied to both forward and reverse CSR.

### 6.2 Optimization 2: Inline Frontier Append with Generation Counter

**Problem**: The A\* re-expansion loop calls three kernels per iteration:

```
find_min_f → expand_delta → compact_frontier
```

`compact_frontier` scans all N nodes checking a flag array, which is expensive (N = 281,903). It also requires a `cudaMemset` of the flag array between iterations.

**Solution**: Merge expansion and frontier building into a single kernel `expand_delta_append`. Use a **generation counter** instead of a flag array to track which nodes are already in the frontier:

```cuda
__global__ void expand_delta_append(
    ..., int* gen_visited, int generation,
    int* next_frontier, int* next_frontier_size, ...) {

    // During expansion, when a child node is relaxed:
    unsigned long long old_val = atomicMin((unsigned long long*)&Cx[child], new_val);
    if (new_val < old_val) {
        int old = atomicExch(&gen_visited[child], generation);
        if (old != generation) {
            int pos = atomicAdd(next_frontier_size, 1);
            next_frontier[pos] = child;
        }
    }
}
```

**How the generation counter works**: Instead of memset-ting the flag array to 0 every iteration, we increment a generation counter. A node is "unvisited this round" if `gen_visited[node] != generation`. `atomicExch` sets it to the current generation and returns the old value — if it was already the current generation, another thread already added this node.

**Benefit**: Eliminates the `compact_frontier` kernel and the `cudaMemset` per A\* iteration. The A\* loop becomes just two kernels: `find_min_f` + `expand_delta_append`.

### 6.3 Optimization 3: Larger DELTA for Dynamic Phase (DELTA\_DYN = 2000)

**Problem**: The static phase uses DELTA = 200, which provides fine-grained expansion suitable for an initial search over the entire graph. But the dynamic re-expansion only needs to fix a small set of affected nodes — most of the graph is already converged. A small DELTA means many iterations of the host loop, each with a kernel launch + host readback overhead.

**Solution**: Use a 10x larger DELTA\_DYN = 2000 for the dynamic phase:

```cuda
unsigned int DELTA     = 200;   // static phase
unsigned int DELTA_DYN = 2000;  // dynamic phase
```

**Benefit**: Fewer A\* iterations (more nodes expanded per round), which means fewer kernel launches and fewer host-device synchronization points. The dynamic frontier is small enough that the coarser bucket width doesn't hurt parallelism.

### 6.4 Combined Effect

The v6 optimizations attack different parts of the pipeline:

| Phase | v5 Time Contribution | v6 Optimization | Effect |
|-------|---------------------|-----------------|--------|
| Deletion marking | ~15% | Binary search (O(log d) vs O(d/32)) | Near-zero |
| Descendant marking | ~10% | (unchanged from v4) | Same |
| Weight update | ~5% | (unchanged from v4) | Same |
| A\* re-expansion | ~70% | Inline append + DELTA\_DYN=2000 | Dramatically fewer iterations |

---

## 7. Version 7 — Cooperative Groups Kernel Fusion

**Speedup: 49.5x over v3 (69.77 ms &rarr; 1.41 ms), 1.5x over v6**

### 7.1 Motivation: Host-Device Synchronization Overhead

Profiling v6 revealed that the remaining ~2 ms was dominated not by compute but by **kernel launch and host-device synchronization overhead**. The dynamic phase still has multiple host-controlled loops:

```
// v6 dynamic phase host loop structure:
markDeletedBinarySearch (stream 1)
markDeletedBinarySearch (stream 2)
markAsAffected (stream 3)
cudaDeviceSynchronize()

InitializeWorklist()
while (worklist_size > 0):       ← host reads D_new_sz each iteration
    markDescendantLockFree()
    swap pointers
    cudaMemcpy(worklist_size)    ← D2H transfer

InitializeWorklist()             ← re-compact affected nodes
Update_weights_worklist()
compact_frontier()

while (frontier_size > 0):      ← host reads frontier_size + min_f each iteration
    find_min_f()
    cudaMemcpyFromSymbol(min_f)  ← D2H transfer
    expand_delta_append()
    cudaMemcpy(frontier_size)    ← D2H transfer
```

Each `cudaMemcpy` and `cudaDeviceSynchronize` stalls the GPU pipeline. For a 2 ms kernel, even microsecond-level stalls add up.

### 7.2 Solution: CUDA Cooperative Groups

CUDA Cooperative Groups provide `grid.sync()` — a barrier that synchronizes **all thread-blocks in a grid**. This enables "persistent kernels" that stay resident on the GPU and loop internally, eliminating the need for host-side loop control.

**Constraint**: All blocks must be simultaneously resident on the GPU. The grid size is capped at `maxActiveBlocksPerSM * numSMs`.

**Compilation**: Requires `-rdc=true` (relocatable device code) and linking with `-lcudadevrt`. Launch via `cudaLaunchCooperativeKernel` instead of `<<<...>>>`.

### 7.3 Fusion 1: Persistent Descendant Marking (`persistentDescendantMarking`)

**Replaces**: `InitializeWorklist` + host while-loop of `markDescendantLockFree` + pointer swaps + `cudaMemcpy` size reads.

**Architecture**:

```cuda
__global__ void persistentDescendantMarking(
    int* offset, int* edges, int* affected,
    int* worklist_A, int* worklist_B,
    int* size_A, int* size_B,
    int N, int E)
{
    cg::grid_group grid = cg::this_grid();
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = gridDim.x * blockDim.x;

    // Phase 0: Initialize worklist (replaces InitializeWorklist kernel)
    for (unsigned int id = tid; id < N; id += stride)
        if (affected[id] == 1) {
            int pos = atomicAdd(size_A, 1);
            worklist_A[pos] = id;
        }
    grid.sync();

    // Phase 1+: BFS propagation (replaces host while-loop)
    int* curr_wl = worklist_A, *next_wl = worklist_B;
    int* curr_sz = size_A,     *next_sz = size_B;

    while (true) {
        int ws = *curr_sz;
        if (ws == 0) break;

        // Warp-per-node edge expansion
        unsigned int warp_id = tid / 32, lane_id = tid % 32;
        unsigned int total_warps = stride / 32;
        for (unsigned int wi = warp_id; wi < ws; wi += total_warps) {
            int node = curr_wl[wi];
            int start = offset[node];
            int end = (node == N-1) ? E : offset[node+1];
            for (int i = start + lane_id; i < end; i += 32) {
                int child = edges[i];
                if (unpackParent(Cx[child]) == node) {
                    int old = atomicExch(&affected[child], 1);
                    if (old == 0) {
                        Cx[child] = pack(INF_COST, -1);
                        next_wl[atomicAdd(next_sz, 1)] = child;
                    }
                }
            }
        }
        grid.sync();

        // Swap worklists (on-device pointer swap)
        int* tmp; tmp = curr_wl; curr_wl = next_wl; next_wl = tmp;
        tmp = curr_sz; curr_sz = next_sz; next_sz = tmp;
        if (tid == 0) *next_sz = 0;
        grid.sync();
    }
}
```

**Grid sizing**: Capped at **64 blocks** (not the full SM capacity). Fewer blocks = faster `grid.sync()`, and the BFS worklist is typically small (~6,000 nodes), so 64 blocks of 512 threads (32,768 threads) provide ample parallelism.

**Benefit**: The entire descendant propagation — which previously required ~16 kernel launches (one per BFS level) plus 16 host-device round-trips — now executes as a single kernel with on-device synchronization.

### 7.4 Fusion 2: Fused Update + Build Frontier (`fuseUpdateAndBuildFrontier`)

**Replaces**: `InitializeWorklist` + `Update_weights_worklist` + `compact_frontier` — three separate kernels with host synchronization between them.

**Architecture** (3 phases within one kernel):

```cuda
__global__ void fuseUpdateAndBuildFrontier(...) {
    cg::grid_group grid = cg::this_grid();

    // Phase 1: Build worklist from affected array
    for (id = tid; id < N; id += stride)
        if (affected[id] == 1) worklist[atomicAdd(worklist_size, 1)] = id;
    grid.sync();

    // Phase 2: Warp-cooperative weight update (reverse graph)
    int num_affected = *worklist_size;
    for (wi = warp_id; wi < num_affected; wi += total_warps) {
        int node = worklist[wi];
        // ... find min-cost parent via reverse edges ...
        // ... warp shuffle reduction ...
        if (lane_id == 0) {
            Cx[node] = pack(minCost, minParent);
            if (minParent != -1) flag_array[node] = 1;
        }
    }
    grid.sync();

    // Phase 3: Compact to frontier
    for (id = tid; id < N; id += stride)
        if (flag_array[id] == 1) {
            frontier[atomicAdd(frontier_size, 1)] = id;
            flag_array[id] = 0;
        }
}
```

**Grid sizing**: Uses full SM capacity (up to `maxActiveBlocksPerSM * 72` blocks) since Phase 2's warp-cooperative update benefits from maximum parallelism.

**Benefit**: Eliminates two kernel launch boundaries and the associated host-device synchronization.

### 7.5 Fusion 3: Persistent A\* Delta-Stepping (`persistentAStarDelta`)

**Replaces**: The host while-loop of `find_min_f` + `expand_delta_append` + host readbacks of `min_f` and `frontier_size`.

**Architecture**:

```cuda
__global__ void persistentAStarDelta(
    int* off, int* edge, unsigned int* W, int* Hx,
    int* frontier_A, int* frontier_B,
    int* size_A, int* size_B,
    int* gen_visited, unsigned int* min_f_scratch,
    int N, int E, unsigned int DELTA_DYN, int endNode)
{
    cg::grid_group grid = cg::this_grid();
    int* curr_f = frontier_A, *next_f = frontier_B;
    int* curr_sz = size_A,    *next_sz = size_B;
    int generation = 1;

    while (true) {
        int fsize = *curr_sz;
        if (fsize == 0) break;

        // Phase 1: Grid-wide min-f reduction
        unsigned int local_min = INF_COST;
        for (idx = tid; idx < fsize; idx += stride)
            local_min = min(local_min, unpackCost(Cx[curr_f[idx]]));
        // warp shuffle → shared memory → atomicMin
        grid.sync();

        // Phase 2: Termination check
        unsigned int mf = *min_f_scratch;
        unsigned int dest_cost = unpackCost(Cx[endNode]);
        if (mf >= dest_cost) break;
        unsigned int threshold = mf + DELTA_DYN;

        // Phase 3: Warp-per-node expansion
        for (wi = warp_id; wi < fsize; wi += total_warps) {
            int node = curr_f[wi];
            if (unpackCost(Cx[node]) > threshold) {
                // Above threshold — carry to next frontier
                ...
                continue;
            }
            // Expand edges, atomicMin Cx[child], append to next frontier
            ...
        }
        grid.sync();

        // Swap frontiers, increment generation, reset
        swap(curr_f, next_f); swap(curr_sz, next_sz);
        generation++;
        if (tid == 0) { *next_sz = 0; *min_f_scratch = INF_COST; }
        grid.sync();
    }
}
```

**Key details**:
- **On-device min-reduction**: Three-level reduction (warp shuffle &rarr; shared memory &rarr; `atomicMin`) replaces the separate `find_min_f` kernel + host readback.
- **On-device termination**: Compares `min_f` against `Cx[endNode]` on the device — no host readback needed.
- **Generation counter**: Prevents duplicate frontier entries without clearing the visited array.

**Grid sizing**: Uses full SM capacity for maximum expansion parallelism.

**Benefit**: The entire A\* re-expansion runs as one kernel. For a typical dynamic update that takes 3–5 A\* iterations, this eliminates 6–10 kernel launches and the same number of host-device synchronization points.

### 7.6 Design Decisions and Failed Experiments

During v7 development, two alternative fusion strategies were tested and **reverted**:

#### Failed Experiment 1: Mega-Fusion (Descendant + Update in One Kernel)

**Idea**: Merge `persistentDescendantMarking` and `fuseUpdateAndBuildFrontier` into a single cooperative kernel to eliminate the launch boundary between them.

**Problem**: `grid.sync()` cost scales with the number of blocks. Descendant marking works best with a small grid (64 blocks) because it has ~16 sync points. Update\_weights needs a large grid (216 blocks) for sufficient parallelism. Merging forces 216 blocks for all phases, making each of the 16 descendant sync barriers ~3x more expensive.

**Result**: Slower than two separate cooperative kernels. Reverted.

#### Failed Experiment 2: Fused Dual-Deletion Kernel

**Idea**: Merge forward and reverse `markDeletedBinarySearch` into a single kernel to reduce launch overhead.

**Problem**: Lost the stream concurrency that allowed forward and reverse marking to overlap. One kernel at 493 &mu;s vs two concurrent kernels at ~328 &mu;s total.

**Result**: Slower. Reverted to 3-stream approach.

### 7.7 v7 Final Dynamic Phase Flow

```
// Step 1: Deletion marking (unchanged from v6 — 3 concurrent streams)
markDeletedBinarySearch ──┐
markDeletedBinarySearch ──┼── concurrent on 3 non-blocking streams
markAsAffected ───────────┘
cudaDeviceSynchronize()

// Step 2: Persistent descendant marking (1 cooperative kernel, 64 blocks)
cudaLaunchCooperativeKernel(persistentDescendantMarking, ...)
    Phase 0: compact affected → worklist          ─┐
    Phase 1+: BFS propagation with grid.sync()     ├── single kernel
    Worklist swap on-device                        ─┘

// Step 3: Fused update + frontier build (1 cooperative kernel, full grid)
cudaLaunchCooperativeKernel(fuseUpdateAndBuildFrontier, ...)
    Phase 1: compact affected → worklist           ─┐
    Phase 2: warp-cooperative weight update          ├── single kernel
    Phase 3: compact updated nodes → frontier      ─┘

// Step 4: Persistent A* (1 cooperative kernel, full grid)
cudaLaunchCooperativeKernel(persistentAStarDelta, ...)
    Loop: min-reduction → terminate? → expand → swap ── single kernel
```

**Total kernel launches in dynamic phase**: 3 (deletion) + 1 (descendant) + 1 (update) + 1 (A\*) = **6 kernels** vs v3's ~30+ kernel launches.

---

## 8. Optimization Progression Summary

The table below traces how each bottleneck was addressed across versions:

| Bottleneck | v3 (baseline) | v4 | v5 | v6 | v7 |
|------------|---------------|----|----|----|----|
| **Deletion marking** | Linear scan, 1 thread/edge | (same) | Warp-coop, `__ballot_sync` | Binary search, sorted CSR | (same as v6) |
| **Descendant marking** | Spin-locks | Lock-free `atomicExch` | (same as v4) | (same as v4) | Persistent kernel, `grid.sync()` |
| **Weight update** | Scan all N nodes | Worklist-based | (same as v4) | (same as v4) | Fused with frontier build |
| **A\* re-expansion** | 3 kernels/iter + host loop | (same) | (same) | 2 kernels/iter, DELTA\_DYN=2000 | 1 persistent kernel, on-device loop |
| **Worklist swap** | `cudaMemcpy` D2D | Pointer swap | (same as v4) | (same as v4) | On-device swap inside kernel |
| **Frontier compaction** | Separate kernel | (same) | (same) | Inline append + gen counter | On-device inside persistent A\* |
| **Kernel concurrency** | Sequential | (same) | 3 non-blocking streams | (same as v5) | (same as v5) |
| **Host-device sync** | Every iteration | (same) | (same) | (same) | Eliminated via cooperative groups |

---

## 9. Correctness Verification

All versions were verified to produce **byte-identical** results:

| Check | Result |
|-------|--------|
| Static cost (all versions) | 123 |
| Dynamic cost (all versions) | 416 |
| Cx array size | 2,255,224 bytes (281,903 nodes &times; 8 bytes) |
| Cx checksum | sum = 2,449,958, finite nodes = 6,176 |
| v3 vs v6 binary diff | Identical (with matching DELTA) |
| v6 vs v7 binary diff | Identical |
| cuda-memcheck | 0 errors |
| cuda racecheck | 0 hazards |

---

## 10. Build Instructions

```bash
cd Data/

# v3 (baseline)
nvcc -O2 -arch=sm_75 ../Deletion/deletion_v3.cu -o deletion_v3

# v4
nvcc -O2 -arch=sm_75 ../Deletion/deletion_v4.cu -o deletion_v4

# v5
nvcc -O2 -arch=sm_75 ../Deletion/deletion_v5.cu -o deletion_v5

# v6
nvcc -O2 -arch=sm_75 ../Deletion/deletion_v6.cu -o deletion_v6

# v7 (cooperative groups require -rdc=true and -lcudadevrt)
nvcc -O2 -arch=sm_75 -rdc=true ../Deletion/deletion_v7.cu -o deletion_v7 -lcudadevrt

# Run any version:
./deletion_vX <startNode> <endNode>
# Example: ./deletion_v7 10986 25198
```

**Note**: Replace `-arch=sm_75` with your GPU's compute capability (e.g., `sm_86` for A10/A100).
