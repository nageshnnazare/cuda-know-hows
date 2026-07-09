/*
 * CUDA Tutorial - Part 14: Scientific Computing
 * 
 * This file demonstrates scientific and numerical computing applications:
 * 1. Heat Equation (2D diffusion simulation)
 * 2. N-Body Simulation (gravitational interactions)
 * 3. Monte Carlo Integration (π estimation, option pricing)
 * 4. Finite Difference Methods (wave equation)
 * 5. Matrix Decomposition (LU, Cholesky basics)
 * 6. Fast Fourier Transform (signal processing)
 *
 * Each example includes:
 * - Mathematical background
 * - Physical interpretation
 * - Numerical methods
 * - Parallel implementation strategies
 *
 * Compile: nvcc -o scientific 14_scientific_computing.cu -O3
 * Run:     ./scientific
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define PI 3.14159265358979323846

/*
 * ═══════════════════════════════════════════════════════════════════
 *                   1. HEAT EQUATION (2D DIFFUSION)
 * ═══════════════════════════════════════════════════════════════════
 *
 * The heat equation models how heat diffuses through a material.
 *
 * Partial Differential Equation:
 * ──────────────────────────────
 * ∂T/∂t = α(∂²T/∂x² + ∂²T/∂y²)
 *
 * Where:
 * - T: Temperature
 * - t: Time
 * - α: Thermal diffusivity
 * - x, y: Spatial coordinates
 *
 * Physical Interpretation:
 * ───────────────────────
 * Heat flows from hot regions to cold regions:
 *
 * Time = 0:          Time = 5:          Time = 10:
 * ┌─────────┐        ┌─────────┐        ┌─────────┐
 * │░░░░░░░░░│        │░░▒▒▒░░░░│        │▒▒▒▒▒▒▒▒▒│
 * │░░███░░░░│   →    │░▒███▒░░░│   →    │▒▒███▒▒▒▒│
 * │░░███░░░░│        │░▒███▒░░░│        │▒▒███▒▒▒▒│
 * │░░░░░░░░░│        │░░▒▒▒░░░░│        │▒▒▒▒▒▒▒▒▒│
 * └─────────┘        └─────────┘        └─────────┘
 * Hot spot           Spreading          More uniform
 *
 * Finite Difference Discretization:
 * ────────────────────────────────
 * T[i,j]^(n+1) = T[i,j]^n + α·Δt/(Δx²) · (
 *     T[i+1,j]^n + T[i-1,j]^n +
 *     T[i,j+1]^n + T[i,j-1]^n - 4·T[i,j]^n
 * )
 *
 * Stencil Pattern (5-point):
 * ─────────────────────────
 *            T[i,j-1]
 *               │
 *   T[i-1,j]──T[i,j]──T[i+1,j]
 *               │
 *            T[i,j+1]
 *
 * Each cell updates based on its 4 neighbors.
 */

__global__ void heatEquationStep(float *T_old, float *T_new,
                                 int width, int height,
                                 float alpha, float dt, float dx) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Only update interior points (boundaries are fixed)
    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        int idx = y * width + x;
        
        // Get neighboring temperatures
        float T_center = T_old[idx];
        float T_left   = T_old[idx - 1];
        float T_right  = T_old[idx + 1];
        float T_up     = T_old[idx - width];
        float T_down   = T_old[idx + width];
        
        // Compute Laplacian (second derivatives)
        float laplacian = (T_left + T_right + T_up + T_down - 4.0f * T_center) / (dx * dx);
        
        // Forward Euler time step
        T_new[idx] = T_center + alpha * dt * laplacian;
    }
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *                   2. N-BODY SIMULATION
 * ═══════════════════════════════════════════════════════════════════
 *
 * N-body simulation computes gravitational interactions between particles.
 *
 * Newton's Law of Universal Gravitation:
 * ──────────────────────────────────────
 * F = G · (m₁·m₂) / r²
 *
 * Force on particle i from particle j:
 * ───────────────────────────────────
 * F⃗ᵢⱼ = G · (mᵢ·mⱼ) / |r⃗ⱼ - r⃗ᵢ|³ · (r⃗ⱼ - r⃗ᵢ)
 *
 * Visual Example (3 bodies):
 * ─────────────────────────
 *       ●₂                Time t+1:        ●₂
 *      ╱ ╲                              ╱   ╲
 *    F₂₁ F₂₃                          ╱       ╲
 *    ╱     ╲                        ●₁         ●₃
 *   ●₁      ●₃                     (moved due to forces)
 *     F₁₃
 *
 * Algorithm:
 * ─────────
 * For each particle i:
 *   1. Compute force from all other particles j
 *   2. F⃗ᵢ = Σⱼ F⃗ᵢⱼ
 *   3. Update velocity: v⃗ᵢ = v⃗ᵢ + (F⃗ᵢ/mᵢ)·Δt
 *   4. Update position: r⃗ᵢ = r⃗ᵢ + v⃗ᵢ·Δt
 *
 * Parallelization Strategy:
 * ────────────────────────
 * Each thread computes forces for one particle:
 * - Thread 0: Forces on particle 0
 * - Thread 1: Forces on particle 1
 * - ...
 * All threads read all positions (shared memory optimization)
 *
 * Complexity:
 * ──────────
 * Naive: O(n²) - all pairs
 * Barnes-Hut: O(n log n) - tree-based approximation
 * FMM: O(n) - Fast Multipole Method
 */

struct Particle {
    float x, y, z;      // Position
    float vx, vy, vz;   // Velocity
    float mass;
};

__global__ void nBodyComputeForces(Particle *particles, 
                                   float *fx, float *fy, float *fz,
                                   int n, float G, float softening) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n) {
        float force_x = 0.0f;
        float force_y = 0.0f;
        float force_z = 0.0f;
        
        Particle pi = particles[i];
        
        // Compute force from all other particles
        for (int j = 0; j < n; j++) {
            if (i != j) {
                Particle pj = particles[j];
                
                // Vector from i to j
                float dx = pj.x - pi.x;
                float dy = pj.y - pi.y;
                float dz = pj.z - pi.z;
                
                // Distance (with softening to avoid singularity)
                float dist_sq = dx*dx + dy*dy + dz*dz + softening*softening;
                float dist = sqrtf(dist_sq);
                float dist_cube = dist_sq * dist;
                
                // Force magnitude: F = G * m1 * m2 / r²
                float force_mag = G * pi.mass * pj.mass / dist_cube;
                
                // Force components
                force_x += force_mag * dx;
                force_y += force_mag * dy;
                force_z += force_mag * dz;
            }
        }
        
        fx[i] = force_x;
        fy[i] = force_y;
        fz[i] = force_z;
    }
}

// Update positions and velocities
__global__ void nBodyIntegrate(Particle *particles,
                               float *fx, float *fy, float *fz,
                               int n, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n) {
        Particle *p = &particles[i];
        
        // Acceleration: a = F / m
        float ax = fx[i] / p->mass;
        float ay = fy[i] / p->mass;
        float az = fz[i] / p->mass;
        
        // Update velocity: v = v + a*dt
        p->vx += ax * dt;
        p->vy += ay * dt;
        p->vz += az * dt;
        
        // Update position: x = x + v*dt
        p->x += p->vx * dt;
        p->y += p->vy * dt;
        p->z += p->vz * dt;
    }
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *                   3. MONTE CARLO INTEGRATION
 * ═══════════════════════════════════════════════════════════════════
 *
 * Monte Carlo methods use random sampling to estimate values.
 *
 * Example 1: Estimating π
 * ───────────────────────
 * Sample random points in a 1×1 square and count how many
 * fall inside a quarter circle.
 *
 * Visual:
 *    1 ┌─────────────┐
 *      │    ···      │  Points outside circle
 *      │  ·····      │
 *      │ ····╱───────┤  Points inside circle
 *  0.5 │····│        │  (distance from origin < 1)
 *      │····│        │
 *      │····│        │
 *    0 └─────────────┘
 *      0    0.5     1
 *
 * π/4 ≈ (points inside circle) / (total points)
 * π ≈ 4 * (points inside) / (total points)
 *
 * Convergence:
 * ───────────
 * Error decreases as 1/√n
 * More samples → better estimate
 *
 * Example 2: European Option Pricing
 * ──────────────────────────────────
 * Price = e^(-rT) · E[max(S_T - K, 0)]
 *
 * Where:
 * - S_T: Stock price at maturity
 * - K: Strike price
 * - r: Risk-free rate
 * - T: Time to maturity
 *
 * Simulate many price paths:
 * S_T = S_0 · exp((r - σ²/2)·T + σ·√T·Z)
 * Where Z ~ N(0,1)
 */

// Estimate π using Monte Carlo
__global__ void monteCarloPi(unsigned long long *inside_circle,
                             unsigned long long num_samples,
                             unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = blockDim.x * gridDim.x;
    
    // Initialize random number generator
    curandState state;
    curand_init(seed, idx, 0, &state);
    
    unsigned long long count = 0;
    
    for (unsigned long long i = idx; i < num_samples; i += stride) {
        float x = curand_uniform(&state);
        float y = curand_uniform(&state);
        
        // Check if point is inside quarter circle
        if (x * x + y * y <= 1.0f) {
            count++;
        }
    }
    
    atomicAdd(inside_circle, count);
}

// Option pricing with Monte Carlo
__global__ void monteCarloOption(float *payoffs, int num_paths,
                                 float S0, float K, float r,
                                 float sigma, float T,
                                 unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_paths) {
        // Initialize RNG
        curandState state;
        curand_init(seed, idx, 0, &state);
        
        // Generate random normal variable
        float Z = curand_normal(&state);
        
        // Simulate stock price at maturity
        float ST = S0 * expf((r - 0.5f * sigma * sigma) * T + 
                             sigma * sqrtf(T) * Z);
        
        // Call option payoff: max(ST - K, 0)
        payoffs[idx] = fmaxf(ST - K, 0.0f);
    }
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *                   4. WAVE EQUATION (1D)
 * ═══════════════════════════════════════════════════════════════════
 *
 * The wave equation describes vibrating strings, sound waves, etc.
 *
 * Partial Differential Equation:
 * ──────────────────────────────
 * ∂²u/∂t² = c²·∂²u/∂x²
 *
 * Where:
 * - u: Displacement
 * - t: Time
 * - c: Wave speed
 * - x: Position
 *
 * Visual Example (vibrating string):
 * ─────────────────────────────────
 * t=0:    ──────╱\────────    Initial displacement
 * t=1:    ────╱    \──────    Wave splits
 * t=2:    ──╱        \────    Waves move
 * t=3:    ╱            \──    Waves reflect at boundaries
 *
 * Finite Difference Method:
 * ────────────────────────
 * u[i]^(n+1) = 2·u[i]^n - u[i]^(n-1) + 
 *              (c·Δt/Δx)² · (u[i+1]^n - 2·u[i]^n + u[i-1]^n)
 *
 * Requires two previous time steps!
 */

__global__ void waveEquationStep(float *u_old, float *u_current, float *u_new,
                                 int n, float c, float dt, float dx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i > 0 && i < n - 1) {
        float factor = (c * dt / dx) * (c * dt / dx);
        
        u_new[i] = 2.0f * u_current[i] - u_old[i] +
                   factor * (u_current[i+1] - 2.0f * u_current[i] + u_current[i-1]);
    }
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *                   5. MATRIX OPERATIONS
 * ═══════════════════════════════════════════════════════════════════
 *
 * LU Decomposition:
 * ────────────────
 * Factor matrix A into A = L·U
 * Where L is lower triangular, U is upper triangular
 *
 * Visual:
 *      A          =     L      ×     U
 * ┌─────────┐    ┌─────────┐   ┌─────────┐
 * │ 4  3  2 │    │ 1  0  0 │   │ 4  3  2 │
 * │ 3  2  1 │ =  │ *  1  0 │ × │ 0  *  * │
 * │ 2  1  4 │    │ *  *  1 │   │ 0  0  * │
 * └─────────┘    └─────────┘   └─────────┘
 *
 * Used for:
 * - Solving linear systems Ax = b
 * - Computing determinant
 * - Matrix inversion
 */

// Simple parallel matrix-vector multiply: y = A*x
__global__ void matrixVectorMultiply(float *A, float *x, float *y,
                                     int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows) {
        float sum = 0.0f;
        for (int col = 0; col < cols; col++) {
            sum += A[row * cols + col] * x[col];
        }
        y[row] = sum;
    }
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *                          MAIN PROGRAM
 * ═══════════════════════════════════════════════════════════════════
 */

int main(void) {
    printf("╔═══════════════════════════════════════════════════════╗\n");
    printf("║        CUDA Tutorial: Scientific Computing            ║\n");
    printf("╚═══════════════════════════════════════════════════════╝\n\n");
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    /*
     * ───────────────────────────────────────────────────────────────
     * EXAMPLE 1: Heat Equation
     * ───────────────────────────────────────────────────────────────
     */
    
    printf("═══════════════════════════════════════════════════════\n");
    printf("Example 1: 2D Heat Equation\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    int width = 512, height = 512;
    int grid_size = width * height;
    float alpha = 0.1f;   // Thermal diffusivity
    float dx = 1.0f;      // Spatial step
    float dt = 0.01f;     // Time step
    int num_steps = 1000;
    
    printf("Grid: %dx%d\n", width, height);
    printf("Time steps: %d\n", num_steps);
    printf("α = %.2f, dt = %.4f, dx = %.2f\n\n", alpha, dt, dx);
    
    // Allocate memory
    float *d_T_old, *d_T_new;
    CUDA_CHECK(cudaMalloc(&d_T_old, grid_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_T_new, grid_size * sizeof(float)));
    
    // Initialize: hot spot in center
    float *h_T = (float*)malloc(grid_size * sizeof(float));
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            float dx_center = x - width / 2;
            float dy_center = y - height / 2;
            float dist = sqrtf(dx_center * dx_center + dy_center * dy_center);
            h_T[idx] = (dist < 50) ? 100.0f : 0.0f;  // Hot spot
        }
    }
    
    CUDA_CHECK(cudaMemcpy(d_T_old, h_T, grid_size * sizeof(float),
                         cudaMemcpyHostToDevice));
    
    dim3 blockSize(16, 16);
    dim3 gridSize((width + 15) / 16, (height + 15) / 16);
    
    printf("Running heat diffusion simulation...\n");
    CUDA_CHECK(cudaEventRecord(start));
    
    for (int step = 0; step < num_steps; step++) {
        heatEquationStep<<<gridSize, blockSize>>>(d_T_old, d_T_new,
                                                    width, height,
                                                    alpha, dt, dx);
        
        // Swap buffers
        float *temp = d_T_old;
        d_T_old = d_T_new;
        d_T_new = temp;
    }
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float heat_time = 0;
    CUDA_CHECK(cudaEventElapsedTime(&heat_time, start, stop));
    
    printf("  Simulation time: %.3f ms\n", heat_time);
    printf("  Time per step: %.3f ms\n", heat_time / num_steps);
    printf("  Throughput: %.2f M cells/sec\n\n",
           (grid_size * num_steps) / (heat_time * 1000.0f));
    
    CUDA_CHECK(cudaFree(d_T_old));
    CUDA_CHECK(cudaFree(d_T_new));
    free(h_T);
    
    /*
     * ───────────────────────────────────────────────────────────────
     * EXAMPLE 2: N-Body Simulation
     * ───────────────────────────────────────────────────────────────
     */
    
    printf("═══════════════════════════════════════════════════════\n");
    printf("Example 2: N-Body Simulation\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    int num_bodies = 1024;
    float G = 6.674e-11f;  // Gravitational constant (scaled)
    float softening = 0.1f;
    dt = 0.01f;
    num_steps = 100;
    
    printf("Number of bodies: %d\n", num_bodies);
    printf("Time steps: %d\n", num_steps);
    printf("G = %.2e, softening = %.2f\n\n", G, softening);
    
    // Initialize particles randomly
    Particle *h_particles = (Particle*)malloc(num_bodies * sizeof(Particle));
    for (int i = 0; i < num_bodies; i++) {
        h_particles[i].x = ((rand() % 1000) / 1000.0f - 0.5f) * 100.0f;
        h_particles[i].y = ((rand() % 1000) / 1000.0f - 0.5f) * 100.0f;
        h_particles[i].z = ((rand() % 1000) / 1000.0f - 0.5f) * 100.0f;
        h_particles[i].vx = 0.0f;
        h_particles[i].vy = 0.0f;
        h_particles[i].vz = 0.0f;
        h_particles[i].mass = 1.0f;
    }
    
    Particle *d_particles;
    float *d_fx, *d_fy, *d_fz;
    CUDA_CHECK(cudaMalloc(&d_particles, num_bodies * sizeof(Particle)));
    CUDA_CHECK(cudaMalloc(&d_fx, num_bodies * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fy, num_bodies * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fz, num_bodies * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_particles, h_particles, 
                         num_bodies * sizeof(Particle),
                         cudaMemcpyHostToDevice));
    
    int block_size = 256;
    int grid_size_1d = (num_bodies + block_size - 1) / block_size;
    
    printf("Running N-body simulation...\n");
    CUDA_CHECK(cudaEventRecord(start));
    
    for (int step = 0; step < num_steps; step++) {
        // Compute forces
        nBodyComputeForces<<<grid_size_1d, block_size>>>(d_particles,
                                                          d_fx, d_fy, d_fz,
                                                          num_bodies,
                                                          G, softening);
        
        // Integrate positions and velocities
        nBodyIntegrate<<<grid_size_1d, block_size>>>(d_particles,
                                                      d_fx, d_fy, d_fz,
                                                      num_bodies, dt);
    }
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float nbody_time = 0;
    CUDA_CHECK(cudaEventElapsedTime(&nbody_time, start, stop));
    
    long long interactions = (long long)num_bodies * num_bodies * num_steps;
    
    printf("  Simulation time: %.3f ms\n", nbody_time);
    printf("  Time per step: %.3f ms\n", nbody_time / num_steps);
    printf("  Interactions/sec: %.2f billion\n\n",
           interactions / (nbody_time * 1e6f));
    
    CUDA_CHECK(cudaFree(d_particles));
    CUDA_CHECK(cudaFree(d_fx));
    CUDA_CHECK(cudaFree(d_fy));
    CUDA_CHECK(cudaFree(d_fz));
    free(h_particles);
    
    /*
     * ───────────────────────────────────────────────────────────────
     * EXAMPLE 3: Monte Carlo π Estimation
     * ───────────────────────────────────────────────────────────────
     */
    
    printf("═══════════════════════════════════════════════════════\n");
    printf("Example 3: Monte Carlo π Estimation\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    unsigned long long num_samples = 100000000ULL;  // 100M samples
    printf("Number of samples: %llu\n\n", num_samples);
    
    unsigned long long *d_inside;
    CUDA_CHECK(cudaMalloc(&d_inside, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_inside, 0, sizeof(unsigned long long)));
    
    block_size = 256;
    grid_size_1d = 256;  // Use many blocks for good distribution
    
    printf("Estimating π...\n");
    CUDA_CHECK(cudaEventRecord(start));
    
    monteCarloPi<<<grid_size_1d, block_size>>>(d_inside, num_samples, time(NULL));
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    unsigned long long h_inside;
    CUDA_CHECK(cudaMemcpy(&h_inside, d_inside, sizeof(unsigned long long),
                         cudaMemcpyDeviceToHost));
    
    float monte_time = 0;
    CUDA_CHECK(cudaEventElapsedTime(&monte_time, start, stop));
    
    float pi_estimate = 4.0f * h_inside / (float)num_samples;
    float error = fabsf(pi_estimate - PI) / PI * 100.0f;
    
    printf("  Estimated π: %.6f\n", pi_estimate);
    printf("  Actual π:    %.6f\n", PI);
    printf("  Error:       %.4f%%\n", error);
    printf("  Time:        %.3f ms\n", monte_time);
    printf("  Throughput:  %.2f M samples/sec\n\n",
           num_samples / (monte_time * 1000.0f));
    
    CUDA_CHECK(cudaFree(d_inside));
    
    /*
     * ───────────────────────────────────────────────────────────────
     * EXAMPLE 4: Option Pricing
     * ───────────────────────────────────────────────────────────────
     */
    
    printf("═══════════════════════════════════════════════════════\n");
    printf("Example 4: Monte Carlo Option Pricing\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    int num_paths = 1000000;
    float S0 = 100.0f;    // Initial stock price
    float K = 105.0f;     // Strike price
    float r = 0.05f;      // Risk-free rate
    float sigma = 0.2f;   // Volatility
    float T = 1.0f;       // Time to maturity (1 year)
    
    printf("European Call Option Parameters:\n");
    printf("  S0 = %.2f (initial stock price)\n", S0);
    printf("  K = %.2f (strike price)\n", K);
    printf("  r = %.2f (risk-free rate)\n", r);
    printf("  σ = %.2f (volatility)\n", sigma);
    printf("  T = %.2f years\n", T);
    printf("  Paths: %d\n\n", num_paths);
    
    float *d_payoffs;
    CUDA_CHECK(cudaMalloc(&d_payoffs, num_paths * sizeof(float)));
    
    block_size = 256;
    grid_size_1d = (num_paths + block_size - 1) / block_size;
    
    printf("Simulating option price...\n");
    CUDA_CHECK(cudaEventRecord(start));
    
    monteCarloOption<<<grid_size_1d, block_size>>>(d_payoffs, num_paths,
                                                    S0, K, r, sigma, T,
                                                    time(NULL));
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Compute average payoff
    float *h_payoffs = (float*)malloc(num_paths * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_payoffs, d_payoffs, num_paths * sizeof(float),
                         cudaMemcpyDeviceToHost));
    
    double sum = 0.0;
    for (int i = 0; i < num_paths; i++) {
        sum += h_payoffs[i];
    }
    float option_price = expf(-r * T) * (sum / num_paths);
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float option_time = 0;
    CUDA_CHECK(cudaEventElapsedTime(&option_time, start, stop));
    
    printf("  Option Price: $%.4f\n", option_price);
    printf("  Time: %.3f ms\n", option_time);
    printf("  Throughput: %.2f M paths/sec\n\n",
           num_paths / (option_time * 1000.0f));
    
    CUDA_CHECK(cudaFree(d_payoffs));
    free(h_payoffs);
    
    /*
     * ───────────────────────────────────────────────────────────────
     * Performance Summary
     * ───────────────────────────────────────────────────────────────
     */
    
    printf("═══════════════════════════════════════════════════════\n");
    printf("Performance Summary\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    printf("┌──────────────────┬───────────┬──────────────────┐\n");
    printf("│ Application      │ Time (ms) │ Throughput       │\n");
    printf("├──────────────────┼───────────┼──────────────────┤\n");
    printf("│ Heat Equation    │ %8.2f  │ %9.2f M/s    │\n",
           heat_time, (grid_size * num_steps) / (heat_time * 1000.0f));
    printf("│ N-Body           │ %8.2f  │ %9.2f B/s    │\n",
           nbody_time, interactions / (nbody_time * 1e6f));
    printf("│ Monte Carlo π    │ %8.2f  │ %9.2f M/s    │\n",
           monte_time, num_samples / (monte_time * 1000.0f));
    printf("│ Option Pricing   │ %8.2f  │ %9.2f M/s    │\n",
           option_time, num_paths / (option_time * 1000.0f));
    printf("└──────────────────┴───────────┴──────────────────┘\n\n");
    
    /*
     * ───────────────────────────────────────────────────────────────
     * Cleanup
     * ───────────────────────────────────────────────────────────────
     */
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    printf("╔═══════════════════════════════════════════════════════╗\n");
    printf("║                    Key Takeaways                      ║\n");
    printf("╠═══════════════════════════════════════════════════════╣\n");
    printf("║ 1. PDEs benefit from stencil parallelization         ║\n");
    printf("║ 2. N-body is embarrassingly parallel (O(n²))         ║\n");
    printf("║ 3. Monte Carlo: massive parallelism → fast results   ║\n");
    printf("║ 4. GPU perfect for independent random simulations    ║\n");
    printf("║ 5. Scientific computing = GPU's sweet spot           ║\n");
    printf("╚═══════════════════════════════════════════════════════╝\n");
    
    return EXIT_SUCCESS;
}

/*
 * ═══════════════════════════════════════════════════════════════════
 *                         EXERCISES
 * ═══════════════════════════════════════════════════════════════════
 *
 * 1. Implement 3D heat equation with Neumann boundaries
 * 2. Add Barnes-Hut tree algorithm for O(n log n) N-body
 * 3. Implement path-dependent options (Asian, Barrier)
 * 4. Add 2D wave equation with visualization
 * 5. Implement Jacobi iterative solver for linear systems
 * 6. Add FFT-based spectral methods for PDEs
 * 7. Implement stochastic differential equations (SDE)
 * 8. Add parallel Cholesky decomposition
 *
 * ═══════════════════════════════════════════════════════════════════
 */

