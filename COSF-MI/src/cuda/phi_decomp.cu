/**
 * COSF-MI: CUDA φ-Decomposition Kernel
 * 
 * GPU-accelerated golden ratio shell decomposition for neural activations.
 * Computes activation magnitudes and assigns to Matryoshka shells.
 */

#include <cuda_runtime.h>
#include <math.h>

// Golden ratio constant
#define PHI 1.6180339887498948482f
#define N_SHELLS 17

// Precomputed φ powers
__constant__ float PHI_POWERS[N_SHELLS + 1];

/**
 * Initialize φ powers in constant memory
 */
extern "C" __host__ void init_phi_powers() {
    float powers[N_SHELLS + 1];
    powers[0] = 1.0f;
    for (int i = 1; i <= N_SHELLS; i++) {
        powers[i] = powers[i-1] * PHI;
    }
    cudaMemcpyToSymbol(PHI_POWERS, powers, sizeof(powers));
}

/**
 * Compute L2 magnitude of vectors
 * 
 * @param input: Input tensor (batch * seq, hidden_dim)
 * @param output: Output magnitudes (batch * seq,)
 * @param n_vectors: Number of vectors
 * @param hidden_dim: Dimension of each vector
 */
extern "C" __global__ void compute_magnitudes(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n_vectors,
    int hidden_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n_vectors) {
        float sum_sq = 0.0f;
        const float* vec = input + idx * hidden_dim;
        
        // Compute sum of squares
        for (int i = 0; i < hidden_dim; i++) {
            float val = vec[i];
            sum_sq += val * val;
        }
        
        output[idx] = sqrtf(sum_sq);
    }
}

/**
 * Assign magnitudes to φ-shells
 * 
 * @param magnitudes: Input magnitudes
 * @param shell_indices: Output shell assignments
 * @param base: Base magnitude for shell 0
 * @param n: Number of elements
 */
extern "C" __global__ void assign_phi_shells(
    const float* __restrict__ magnitudes,
    int* __restrict__ shell_indices,
    float base,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float mag = magnitudes[idx];
        
        if (mag <= 0.0f) {
            shell_indices[idx] = -1;
            return;
        }
        
        // Binary search for shell
        int shell = 0;
        float boundary = base;
        
        while (shell < N_SHELLS && mag >= boundary) {
            boundary *= PHI;
            shell++;
        }
        
        shell_indices[idx] = min(shell - 1, N_SHELLS - 1);
    }
}

/**
 * Compute φ-coherence for shell assignments
 * Uses parallel reduction
 */
extern "C" __global__ void compute_phi_coherence(
    const float* __restrict__ magnitudes,
    const int* __restrict__ shell_indices,
    const float* __restrict__ shell_centers,
    float* __restrict__ deviations,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        int shell = shell_indices[idx];
        
        if (shell >= 0 && shell < N_SHELLS) {
            float mag = magnitudes[idx];
            float expected = shell_centers[shell];
            deviations[idx] = fabsf(mag - expected) / expected;
        } else {
            deviations[idx] = 0.0f;
        }
    }
}

/**
 * Combined kernel: magnitude + shell assignment in one pass
 * More efficient for small hidden dimensions
 */
extern "C" __global__ void phi_decompose_fused(
    const float* __restrict__ input,
    float* __restrict__ magnitudes,
    int* __restrict__ shell_indices,
    float base,
    int n_vectors,
    int hidden_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n_vectors) {
        // Compute magnitude
        float sum_sq = 0.0f;
        const float* vec = input + idx * hidden_dim;
        
        for (int i = 0; i < hidden_dim; i++) {
            float val = vec[i];
            sum_sq += val * val;
        }
        
        float mag = sqrtf(sum_sq);
        magnitudes[idx] = mag;
        
        // Assign shell
        if (mag <= 0.0f) {
            shell_indices[idx] = -1;
        } else {
            int shell = 0;
            float boundary = base;
            
            while (shell < N_SHELLS && mag >= boundary) {
                boundary *= PHI;
                shell++;
            }
            
            shell_indices[idx] = min(shell - 1, N_SHELLS - 1);
        }
    }
}

// Wrapper structures for Python interface
typedef struct {
    float* magnitudes;
    int* shell_indices;
    int n_vectors;
} PhiDecompResult;
