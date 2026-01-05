/**
 * COSF-MI: CUDA Julia Set Stability Kernel
 * 
 * GPU-accelerated Julia set iteration for stability analysis.
 * z_{n+1} = z_n² + c
 * 
 * Stability bound: |z| > 2 indicates escape/instability
 */

#include <cuda_runtime.h>
#include <cuComplex.h>

#define JULIA_BOUND 2.0f
#define MAX_ITERATIONS 100

/**
 * Julia iteration for batch of complex numbers
 * 
 * @param z_real: Real parts of initial z values
 * @param z_imag: Imaginary parts of initial z values
 * @param c_real: Real part of Julia parameter c
 * @param c_imag: Imaginary part of Julia parameter c
 * @param escape_iter: Output escape iterations (-1 if stable)
 * @param max_mags: Output maximum magnitudes reached
 * @param n: Number of elements
 * @param max_iter: Maximum iterations to perform
 */
extern "C" __global__ void julia_iterate_batch(
    float* __restrict__ z_real,
    float* __restrict__ z_imag,
    float c_real,
    float c_imag,
    int* __restrict__ escape_iter,
    float* __restrict__ max_mags,
    int n,
    int max_iter
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float zr = z_real[idx];
        float zi = z_imag[idx];
        float max_mag = sqrtf(zr * zr + zi * zi);
        int escaped = -1;
        
        for (int i = 0; i < max_iter && escaped < 0; i++) {
            // z = z² + c
            float new_zr = zr * zr - zi * zi + c_real;
            float new_zi = 2.0f * zr * zi + c_imag;
            
            zr = new_zr;
            zi = new_zi;
            
            // Check magnitude
            float mag = sqrtf(zr * zr + zi * zi);
            max_mag = fmaxf(max_mag, mag);
            
            // Check escape
            if (mag > JULIA_BOUND) {
                escaped = i;
            }
        }
        
        escape_iter[idx] = escaped;
        max_mags[idx] = max_mag;
        z_real[idx] = zr;
        z_imag[idx] = zi;
    }
}

/**
 * Classify stability based on escape analysis
 * 0 = stable, 1 = marginal, 2 = unstable, 3 = chaotic
 */
extern "C" __global__ void classify_stability(
    const int* __restrict__ escape_iter,
    const float* __restrict__ max_mags,
    int* __restrict__ classes,
    float marginal_threshold,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        int escape = escape_iter[idx];
        float max_mag = max_mags[idx];
        
        if (escape >= 0) {
            // Escaped - check if chaotic (very fast escape)
            classes[idx] = (escape < 10) ? 3 : 2;
        } else if (max_mag > marginal_threshold) {
            // Didn't escape but got close
            classes[idx] = 1;
        } else {
            // Stable
            classes[idx] = 0;
        }
    }
}

/**
 * Convert activation pairs to complex for Julia analysis
 * Takes pairs of adjacent floats as (real, imag)
 */
extern "C" __global__ void activations_to_complex(
    const float* __restrict__ activations,
    float* __restrict__ z_real,
    float* __restrict__ z_imag,
    int n_vectors,
    int hidden_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_complex = hidden_dim / 2;
    int total = n_vectors * n_complex;
    
    if (idx < total) {
        int vec_idx = idx / n_complex;
        int pair_idx = idx % n_complex;
        
        int act_idx = vec_idx * hidden_dim + pair_idx * 2;
        
        z_real[idx] = activations[act_idx];
        z_imag[idx] = activations[act_idx + 1];
    }
}

/**
 * Fused kernel: convert + iterate + classify in one pass
 */
extern "C" __global__ void julia_analyze_fused(
    const float* __restrict__ activations,
    int* __restrict__ stability_classes,
    float* __restrict__ max_magnitudes,
    float c_real,
    float c_imag,
    float scale,
    float marginal_threshold,
    int n_vectors,
    int hidden_dim,
    int max_iter
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_complex = hidden_dim / 2;
    int total = n_vectors * n_complex;
    
    if (idx < total) {
        int vec_idx = idx / n_complex;
        int pair_idx = idx % n_complex;
        int act_idx = vec_idx * hidden_dim + pair_idx * 2;
        
        // Load and normalize
        float zr = activations[act_idx] / scale;
        float zi = activations[act_idx + 1] / scale;
        float max_mag = sqrtf(zr * zr + zi * zi);
        int escaped = -1;
        
        // Julia iteration
        for (int i = 0; i < max_iter && escaped < 0; i++) {
            float new_zr = zr * zr - zi * zi + c_real;
            float new_zi = 2.0f * zr * zi + c_imag;
            zr = new_zr;
            zi = new_zi;
            
            float mag = sqrtf(zr * zr + zi * zi);
            max_mag = fmaxf(max_mag, mag);
            
            if (mag > JULIA_BOUND) {
                escaped = i;
            }
        }
        
        // Classify
        int stability;
        if (escaped >= 0) {
            stability = (escaped < 10) ? 3 : 2;
        } else if (max_mag > marginal_threshold) {
            stability = 1;
        } else {
            stability = 0;
        }
        
        stability_classes[idx] = stability;
        max_magnitudes[idx] = max_mag;
    }
}

/**
 * Reduction kernel for computing stability statistics
 */
extern "C" __global__ void stability_stats_reduce(
    const int* __restrict__ classes,
    int* __restrict__ counts,  // [4] for each class
    int n
) {
    __shared__ int shared_counts[4];
    
    if (threadIdx.x < 4) {
        shared_counts[threadIdx.x] = 0;
    }
    __syncthreads();
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        int cls = classes[idx];
        if (cls >= 0 && cls < 4) {
            atomicAdd(&shared_counts[cls], 1);
        }
    }
    __syncthreads();
    
    if (threadIdx.x < 4) {
        atomicAdd(&counts[threadIdx.x], shared_counts[threadIdx.x]);
    }
}
