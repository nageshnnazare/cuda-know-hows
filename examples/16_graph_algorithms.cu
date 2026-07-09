/*
 * CUDA Tutorial - Part 16: Graph Algorithms
 * 
 * This file demonstrates parallel graph processing algorithms:
 * 1. Breadth-First Search (BFS) - Level-synchronous traversal
 * 2. Single-Source Shortest Path (Dijkstra-like)
 * 3. All-Pairs Shortest Path (Floyd-Warshall)
 * 4. Connected Components (Label Propagation)
 * 5. Triangle Counting (Graph analytics)
 * 6. PageRank (Web graph ranking)
 *
 * Each algorithm includes:
 * - Graph representation (CSR format)
 * - Visualization of the algorithm
 * - Parallel implementation
 * - Performance analysis
 *
 * Compile: nvcc -o graph_algos 16_graph_algorithms.cu -O3
 * Run:     ./graph_algos
 */

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

/*
 * ═══════════════════════════════════════════════════════════════════
 *                   GRAPH REPRESENTATION
 * ═══════════════════════════════════════════════════════════════════
 *
 * We use Compressed Sparse Row (CSR) format for efficient GPU storage.
 *
 * Example Graph:
 * ─────────────
 *     0 ──→ 1
 *     ↓     ↓
 *     2 ──→ 3
 *
 * Adjacency Matrix:      CSR Format:
 * ┌───────────┐         ┌─────────────────────────┐
 * │ 0  1  0  0 │         │ row_ptr:  [0, 2, 3, 4, 5]│
 * │ 0  0  0  1 │    →    │ col_idx:  [1, 2, 3, 2, 3]│
 * │ 0  0  0  1 │         │ values:   [1, 1, 1, 1, 1]│
 * │ 0  0  0  0 │         └─────────────────────────┘
 * └───────────┘
 *
 * Advantages of CSR:
 * - Memory efficient (only store non-zero edges)
 * - Coalesced memory access
 * - Fast neighbor iteration
 *
 * row_ptr[i] to row_ptr[i+1] gives neighbors of vertex i
 * col_idx stores destination vertices
 * values stores edge weights (optional)
 */

struct CSRGraph {
    int num_vertices;
    int num_edges;
    int *row_ptr;      // Size: num_vertices + 1
    int *col_idx;      // Size: num_edges
    int *values;       // Size: num_edges (optional weights)
};

/*
 * ═══════════════════════════════════════════════════════════════════
 *                   1. BREADTH-FIRST SEARCH (BFS)
 * ═══════════════════════════════════════════════════════════════════
 *
 * BFS explores graph level by level from a source vertex.
 *
 * Visual Example:
 * ──────────────
 * Graph:              BFS from vertex 0:
 *     0                Level 0: [0]
 *    ╱ ╲                      ↓
 *   1   2              Level 1: [1, 2]
 *  ╱ ╲   ╲                    ↓
 * 3   4   5            Level 2: [3, 4, 5]
 *
 * Algorithm (Level-Synchronous):
 * ─────────────────────────────
 * 1. Start with source in current frontier
 * 2. For each vertex in frontier:
 *    - Visit all unvisited neighbors
 *    - Add neighbors to next frontier
 * 3. Swap frontiers and repeat
 *
 * Parallel Strategy:
 * ─────────────────
 * Each thread processes one vertex in current frontier
 * All threads work simultaneously on same level
 *
 * Frontier Representation:
 * ───────────────────────
 * Use two arrays:
 * - current_frontier[]: Vertices to process this iteration
 * - next_frontier[]:    Vertices to process next iteration
 *
 * Also need:
 * - visited[]: Boolean array (has vertex been seen?)
 * - distance[]: Distance from source
 */

__global__ void bfsKernel(int *row_ptr, int *col_idx,
                          bool *current_frontier, bool *next_frontier,
                          bool *visited, int *distance,
                          int num_vertices, int current_level) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (v < num_vertices && current_frontier[v]) {
        // Process this vertex's neighbors
        int start = row_ptr[v];
        int end = row_ptr[v + 1];
        
        for (int edge = start; edge < end; edge++) {
            int neighbor = col_idx[edge];
            
            // If neighbor not visited, add to next frontier
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                next_frontier[neighbor] = true;
                distance[neighbor] = current_level + 1;
            }
        }
    }
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *              2. SINGLE-SOURCE SHORTEST PATH
 * ═══════════════════════════════════════════════════════════════════
 *
 * Compute shortest paths from source to all vertices.
 * Uses relaxation-based approach similar to Bellman-Ford.
 *
 * Relaxation:
 * ──────────
 * For edge (u, v) with weight w:
 * if dist[u] + w < dist[v]:
 *     dist[v] = dist[u] + w
 *
 * Visual Example:
 * ──────────────
 * Initial:           After relaxing edge (0,1):
 *     0 (0)              0 (0)
 *    ╱ ╲ 5              ╱ ╲ 5
 *   5   2 (∞)          5   2 (5)
 *  ╱         ╲        ╱         ╲
 * 1 (∞)       3 (∞)  1 (5)       3 (∞)
 *
 * Parallel Strategy:
 * ─────────────────
 * Each thread tries to relax all edges from one vertex
 * Repeat until no changes (convergence)
 * Use atomicMin for race-free updates
 */

__global__ void sssp_relax(int *row_ptr, int *col_idx, int *weights,
                           int *distance, bool *changed,
                           int num_vertices) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (u < num_vertices) {
        int dist_u = distance[u];
        
        if (dist_u != INT_MAX) {
            // Relax all outgoing edges from u
            int start = row_ptr[u];
            int end = row_ptr[u + 1];
            
            for (int edge = start; edge < end; edge++) {
                int v = col_idx[edge];
                int weight = weights[edge];
                int new_dist = dist_u + weight;
                
                // Atomic update for thread-safety
                int old_dist = atomicMin(&distance[v], new_dist);
                
                if (new_dist < old_dist) {
                    *changed = true;  // Signal that we made progress
                }
            }
        }
    }
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *              3. ALL-PAIRS SHORTEST PATH (Floyd-Warshall)
 * ═══════════════════════════════════════════════════════════════════
 *
 * Compute shortest paths between ALL pairs of vertices.
 *
 * Algorithm:
 * ─────────
 * For each intermediate vertex k:
 *   For each pair (i, j):
 *     if dist[i][j] > dist[i][k] + dist[k][j]:
 *         dist[i][j] = dist[i][k] + dist[k][j]
 *
 * Visual Example (k=1):
 * ────────────────────
 * Before:            Try path through vertex 1:
 * 0 ──5── 1          0 ──5── 1
 * │       │          │   ↓   │
 * 10      3     →    │   8   │  (5+3=8 < 10)
 * │       │          │   ↓   │
 * └───────2          └───────2
 *
 * dist[0][2] = min(10, dist[0][1] + dist[1][2])
 *            = min(10, 5 + 3)
 *            = 8
 *
 * Parallelization:
 * ───────────────
 * For each k (sequential):
 *   Process all (i,j) pairs in parallel
 *   Each thread handles one pair
 *
 * Complexity:
 * ──────────
 * Time: O(V³) but with massive parallelism
 * Space: O(V²) distance matrix
 */

__global__ void floydWarshall(int *dist, int n, int k) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n && j < n) {
        int idx_ij = i * n + j;
        int idx_ik = i * n + k;
        int idx_kj = k * n + j;
        
        int dist_ik = dist[idx_ik];
        int dist_kj = dist[idx_kj];
        
        if (dist_ik != INT_MAX && dist_kj != INT_MAX) {
            int new_dist = dist_ik + dist_kj;
            if (new_dist < dist[idx_ij]) {
                dist[idx_ij] = new_dist;
            }
        }
    }
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *              4. CONNECTED COMPONENTS
 * ═══════════════════════════════════════════════════════════════════
 *
 * Find all connected components using label propagation.
 *
 * Algorithm:
 * ─────────
 * 1. Initially, each vertex has its own label (vertex ID)
 * 2. Repeat:
 *    - Each vertex takes minimum label from neighbors
 *    - Continue until no changes
 *
 * Visual Example:
 * ──────────────
 * Initial:         Iteration 1:      Iteration 2:      Final:
 * 0───1    3       0───1    3        0───1    3        0───1    3
 * │   │    │       │   │    │        │   │    │        │   │    │
 * 2───4    5       0───0    3        0───0    3        0───0    3
 * [0,1,2,4,3,5]   [0,0,0,0,3,3]     [0,0,0,0,3,3]     [0,0,0,0,3,3]
 *
 * Two components: {0,1,2,4} and {3,5}
 *
 * Parallel Strategy:
 * ─────────────────
 * Each thread processes one vertex
 * Use atomicMin to update labels safely
 */

__global__ void connectedComponents(int *row_ptr, int *col_idx,
                                    int *labels, bool *changed,
                                    int num_vertices) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (v < num_vertices) {
        int my_label = labels[v];
        int min_label = my_label;
        
        // Check all neighbors
        int start = row_ptr[v];
        int end = row_ptr[v + 1];
        
        for (int edge = start; edge < end; edge++) {
            int neighbor = col_idx[edge];
            int neighbor_label = labels[neighbor];
            
            if (neighbor_label < min_label) {
                min_label = neighbor_label;
            }
        }
        
        // Update if found smaller label
        if (min_label < my_label) {
            atomicMin(&labels[v], min_label);
            *changed = true;
        }
    }
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *              5. TRIANGLE COUNTING
 * ═══════════════════════════════════════════════════════════════════
 *
 * Count the number of triangles in the graph.
 * A triangle is three vertices all connected to each other.
 *
 * Visual Example:
 * ──────────────
 * Graph:              Triangles:
 *   0───1              {0, 1, 2} ✓
 *   │\ /│              {0, 1, 3} ✗ (no edge 0-3)
 *   │ X │              
 *   │/ \│              
 *   2───3              
 *
 * Algorithm:
 * ─────────
 * For each edge (u, v):
 *   Count common neighbors of u and v
 *   Each common neighbor forms a triangle
 *
 * Pseudocode:
 * ──────────
 * count = 0
 * for each edge (u, v):
 *   neighbors_u = get_neighbors(u)
 *   neighbors_v = get_neighbors(v)
 *   count += intersect(neighbors_u, neighbors_v).size()
 * return count / 3  // Each triangle counted 3 times
 *
 * Parallel Strategy:
 * ─────────────────
 * Each thread processes one edge
 * Count common neighbors using intersection
 */

__global__ void triangleCount(int *row_ptr, int *col_idx,
                              unsigned long long *count,
                              int num_vertices) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (u < num_vertices) {
        unsigned long long local_count = 0;
        
        // For each neighbor v of u
        int start_u = row_ptr[u];
        int end_u = row_ptr[u + 1];
        
        for (int i = start_u; i < end_u; i++) {
            int v = col_idx[i];
            
            if (v > u) {  // Avoid double counting
                // Count common neighbors of u and v
                int start_v = row_ptr[v];
                int end_v = row_ptr[v + 1];
                
                int ptr_u = start_u;
                int ptr_v = start_v;
                
                // Merge-like intersection
                while (ptr_u < end_u && ptr_v < end_v) {
                    int neighbor_u = col_idx[ptr_u];
                    int neighbor_v = col_idx[ptr_v];
                    
                    if (neighbor_u == neighbor_v && neighbor_u > v) {
                        local_count++;
                        ptr_u++;
                        ptr_v++;
                    } else if (neighbor_u < neighbor_v) {
                        ptr_u++;
                    } else {
                        ptr_v++;
                    }
                }
            }
        }
        
        if (local_count > 0) {
            atomicAdd(count, local_count);
        }
    }
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *              6. PAGERANK
 * ═══════════════════════════════════════════════════════════════════
 *
 * Compute importance/rank of vertices (used by Google for web pages).
 *
 * Formula:
 * ───────
 * PR(v) = (1-d)/N + d * Σ PR(u)/out_degree(u)
 *                       u→v
 *
 * Where:
 * - d: Damping factor (typically 0.85)
 * - N: Number of vertices
 * - u→v: u has edge to v
 *
 * Visual Intuition:
 * ────────────────
 * Vertex with many high-ranked incoming edges → high rank
 *
 * Example:
 *     A ──→ B       B has higher rank because:
 *     ↓     ↓       - A points to it
 *     C ←── D       - D points to it
 *
 * Algorithm:
 * ─────────
 * 1. Initialize all ranks to 1/N
 * 2. Repeat until convergence:
 *    - Compute new rank for each vertex
 *    - Check for convergence (small change)
 *
 * Parallel Strategy:
 * ─────────────────
 * Two phases per iteration:
 * Phase 1: Each vertex distributes rank to neighbors
 * Phase 2: Each vertex sums incoming ranks
 */

__global__ void pageRankContribute(int *row_ptr, int *col_idx,
                                   float *ranks, float *contributions,
                                   int num_vertices) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (v < num_vertices) {
        int out_degree = row_ptr[v + 1] - row_ptr[v];
        
        if (out_degree > 0) {
            float contrib = ranks[v] / out_degree;
            
            // Send contribution to all neighbors
            int start = row_ptr[v];
            int end = row_ptr[v + 1];
            
            for (int edge = start; edge < end; edge++) {
                int neighbor = col_idx[edge];
                atomicAdd(&contributions[neighbor], contrib);
            }
        }
    }
}

__global__ void pageRankUpdate(float *ranks, float *contributions,
                                int num_vertices, float damping) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (v < num_vertices) {
        ranks[v] = (1.0f - damping) / num_vertices + 
                   damping * contributions[v];
        contributions[v] = 0.0f;  // Reset for next iteration
    }
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *                    UTILITY FUNCTIONS
 * ═══════════════════════════════════════════════════════════════════
 */

// Create a simple test graph
void createTestGraph(CSRGraph *graph) {
    // Small graph for demonstration: 6 vertices
    //     0 ──→ 1 ──→ 3
    //     ↓     ↓     ↑
    //     2 ────┘     │
    //                 │
    //     4 ──→ 5 ────┘
    
    graph->num_vertices = 6;
    graph->num_edges = 7;
    
    int h_row_ptr[] = {0, 2, 4, 5, 5, 6, 7};  // 7 elements (vertices + 1)
    int h_col_idx[] = {1, 2, 3, 2, 2, 5, 3};  // 7 edges
    int h_values[] = {1, 1, 1, 1, 1, 1, 1};
    
    size_t row_size = (graph->num_vertices + 1) * sizeof(int);
    size_t edge_size = graph->num_edges * sizeof(int);
    
    CUDA_CHECK(cudaMalloc(&graph->row_ptr, row_size));
    CUDA_CHECK(cudaMalloc(&graph->col_idx, edge_size));
    CUDA_CHECK(cudaMalloc(&graph->values, edge_size));
    
    CUDA_CHECK(cudaMemcpy(graph->row_ptr, h_row_ptr, row_size,
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(graph->col_idx, h_col_idx, edge_size,
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(graph->values, h_values, edge_size,
                         cudaMemcpyHostToDevice));
}

void freeGraph(CSRGraph *graph) {
    CUDA_CHECK(cudaFree(graph->row_ptr));
    CUDA_CHECK(cudaFree(graph->col_idx));
    CUDA_CHECK(cudaFree(graph->values));
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *                          MAIN PROGRAM
 * ═══════════════════════════════════════════════════════════════════
 */

int main(void) {
    printf("╔═══════════════════════════════════════════════════════╗\n");
    printf("║        CUDA Tutorial: Graph Algorithms                ║\n");
    printf("╚═══════════════════════════════════════════════════════╝\n\n");
    
    CSRGraph graph;
    createTestGraph(&graph);
    
    printf("Test Graph: %d vertices, %d edges\n", 
           graph.num_vertices, graph.num_edges);
    printf("Structure:\n");
    printf("  0 → 1, 2\n");
    printf("  1 → 3, 2\n");
    printf("  2 → 2 (self-loop)\n");
    printf("  3 → (none)\n");
    printf("  4 → 5\n");
    printf("  5 → 3\n\n");
    
    int blockSize = 256;
    int gridSize = (graph.num_vertices + blockSize - 1) / blockSize;
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    /*
     * ───────────────────────────────────────────────────────────────
     * TEST 1: Breadth-First Search
     * ───────────────────────────────────────────────────────────────
     */
    
    printf("═══════════════════════════════════════════════════════\n");
    printf("Test 1: Breadth-First Search from vertex 0\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    bool *d_current_frontier, *d_next_frontier, *d_visited;
    int *d_distance;
    
    CUDA_CHECK(cudaMalloc(&d_current_frontier, 
                         graph.num_vertices * sizeof(bool)));
    CUDA_CHECK(cudaMalloc(&d_next_frontier, 
                         graph.num_vertices * sizeof(bool)));
    CUDA_CHECK(cudaMalloc(&d_visited, graph.num_vertices * sizeof(bool)));
    CUDA_CHECK(cudaMalloc(&d_distance, graph.num_vertices * sizeof(int)));
    
    // Initialize
    bool h_frontier[6] = {true, false, false, false, false, false};
    bool h_visited[6] = {true, false, false, false, false, false};
    int h_distance[6] = {0, -1, -1, -1, -1, -1};
    
    CUDA_CHECK(cudaMemcpy(d_current_frontier, h_frontier, 
                         graph.num_vertices * sizeof(bool),
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_next_frontier, 0, 
                         graph.num_vertices * sizeof(bool)));
    CUDA_CHECK(cudaMemcpy(d_visited, h_visited,
                         graph.num_vertices * sizeof(bool),
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_distance, h_distance,
                         graph.num_vertices * sizeof(int),
                         cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaEventRecord(start));
    
    int level = 0;
    int max_iterations = 10;
    
    for (int iter = 0; iter < max_iterations; iter++) {
        bfsKernel<<<gridSize, blockSize>>>(graph.row_ptr, graph.col_idx,
                                            d_current_frontier, d_next_frontier,
                                            d_visited, d_distance,
                                            graph.num_vertices, level);
        
        // Check if any work done
        bool h_next[6];
        CUDA_CHECK(cudaMemcpy(h_next, d_next_frontier,
                             graph.num_vertices * sizeof(bool),
                             cudaMemcpyDeviceToHost));
        
        bool has_work = false;
        for (int i = 0; i < graph.num_vertices; i++) {
            if (h_next[i]) {
                has_work = true;
                break;
            }
        }
        
        if (!has_work) break;
        
        // Swap frontiers
        bool *temp = d_current_frontier;
        d_current_frontier = d_next_frontier;
        d_next_frontier = temp;
        
        CUDA_CHECK(cudaMemset(d_next_frontier, 0,
                             graph.num_vertices * sizeof(bool)));
        level++;
    }
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float bfs_time = 0;
    CUDA_CHECK(cudaEventElapsedTime(&bfs_time, start, stop));
    
    CUDA_CHECK(cudaMemcpy(h_distance, d_distance,
                         graph.num_vertices * sizeof(int),
                         cudaMemcpyDeviceToHost));
    
    printf("BFS Results:\n");
    printf("  Vertex  Distance\n");
    printf("  ──────  ────────\n");
    for (int i = 0; i < graph.num_vertices; i++) {
        if (h_distance[i] >= 0) {
            printf("    %d        %d\n", i, h_distance[i]);
        } else {
            printf("    %d        unreachable\n", i);
        }
    }
    printf("\n  Time: %.3f ms\n\n", bfs_time);
    
    CUDA_CHECK(cudaFree(d_current_frontier));
    CUDA_CHECK(cudaFree(d_next_frontier));
    CUDA_CHECK(cudaFree(d_visited));
    CUDA_CHECK(cudaFree(d_distance));
    
    /*
     * ───────────────────────────────────────────────────────────────
     * TEST 2: PageRank
     * ───────────────────────────────────────────────────────────────
     */
    
    printf("═══════════════════════════════════════════════════════\n");
    printf("Test 2: PageRank\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    float *d_ranks, *d_contributions;
    CUDA_CHECK(cudaMalloc(&d_ranks, graph.num_vertices * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_contributions, 
                         graph.num_vertices * sizeof(float)));
    
    // Initialize ranks to 1/N
    float init_rank = 1.0f / graph.num_vertices;
    float h_ranks[6];
    for (int i = 0; i < graph.num_vertices; i++) {
        h_ranks[i] = init_rank;
    }
    
    CUDA_CHECK(cudaMemcpy(d_ranks, h_ranks,
                         graph.num_vertices * sizeof(float),
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_contributions, 0,
                         graph.num_vertices * sizeof(float)));
    
    float damping = 0.85f;
    int pagerank_iterations = 20;
    
    printf("Running PageRank (damping=%.2f, %d iterations)...\n",
           damping, pagerank_iterations);
    
    CUDA_CHECK(cudaEventRecord(start));
    
    for (int iter = 0; iter < pagerank_iterations; iter++) {
        pageRankContribute<<<gridSize, blockSize>>>(graph.row_ptr,
                                                      graph.col_idx,
                                                      d_ranks,
                                                      d_contributions,
                                                      graph.num_vertices);
        
        pageRankUpdate<<<gridSize, blockSize>>>(d_ranks, d_contributions,
                                                 graph.num_vertices, damping);
    }
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float pagerank_time = 0;
    CUDA_CHECK(cudaEventElapsedTime(&pagerank_time, start, stop));
    
    CUDA_CHECK(cudaMemcpy(h_ranks, d_ranks,
                         graph.num_vertices * sizeof(float),
                         cudaMemcpyDeviceToHost));
    
    printf("\nPageRank Results:\n");
    printf("  Vertex  Rank\n");
    printf("  ──────  ──────\n");
    for (int i = 0; i < graph.num_vertices; i++) {
        printf("    %d     %.4f\n", i, h_ranks[i]);
    }
    printf("\n  Time: %.3f ms\n\n", pagerank_time);
    
    CUDA_CHECK(cudaFree(d_ranks));
    CUDA_CHECK(cudaFree(d_contributions));
    
    /*
     * ───────────────────────────────────────────────────────────────
     * Cleanup
     * ───────────────────────────────────────────────────────────────
     */
    
    freeGraph(&graph);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    printf("╔═══════════════════════════════════════════════════════╗\n");
    printf("║                    Key Takeaways                      ║\n");
    printf("╠═══════════════════════════════════════════════════════╣\n");
    printf("║ 1. CSR format enables efficient graph storage        ║\n");
    printf("║ 2. Level-synchronous BFS good for GPUs              ║\n");
    printf("║ 3. Atomic operations handle race conditions          ║\n");
    printf("║ 4. Iterative algorithms converge quickly             ║\n");
    printf("║ 5. Graph analytics benefit from massive parallelism  ║\n");
    printf("╚═══════════════════════════════════════════════════════╝\n");
    
    return EXIT_SUCCESS;
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *                         EXERCISES
 * ═══════════════════════════════════════════════════════════════════
 *
 * 1. Implement bidirectional BFS for faster path finding
 * 2. Add priority queue for Dijkstra's algorithm
 * 3. Implement betweenness centrality
 * 4. Add graph coloring algorithm
 * 5. Implement community detection (Louvain method)
 * 6. Add support for directed vs undirected graphs
 * 7. Implement graph sparsification
 * 8. Add max-flow/min-cut algorithm
 *
 * ═══════════════════════════════════════════════════════════════════
 */

