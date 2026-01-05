//! COSF-MI Core Rust Library
//!
//! High-performance implementations of COSF mathematical operations.

use std::f64::consts::E;

/// Golden ratio constant
pub const PHI: f64 = 1.6180339887498948482;

/// COSF constant (Schumann ratio)
pub const COSF: f64 = 5465.90;

/// Julia stability bound
pub const JULIA_BOUND: f64 = 2.0;

/// Number of Matryoshka shells
pub const N_SHELLS: usize = 17;

/// Precomputed φ powers
pub fn phi_powers() -> [f64; N_SHELLS + 1] {
    let mut powers = [0.0; N_SHELLS + 1];
    powers[0] = 1.0;
    for i in 1..=N_SHELLS {
        powers[i] = powers[i - 1] * PHI;
    }
    powers
}

/// Verify COSF convergence
pub fn verify_convergence() -> ConvergenceResult {
    let phi_17 = PHI.powi(17);
    let e_8_6 = E.powf(8.6);
    let ratio = phi_17 / e_8_6;
    let deviation = (1.0 - ratio).abs() * 100.0;
    
    ConvergenceResult {
        phi: PHI,
        phi_17,
        e_8_6,
        ratio,
        deviation_percent: deviation,
        is_valid: deviation < 1.0,
    }
}

#[derive(Debug, Clone)]
pub struct ConvergenceResult {
    pub phi: f64,
    pub phi_17: f64,
    pub e_8_6: f64,
    pub ratio: f64,
    pub deviation_percent: f64,
    pub is_valid: bool,
}

/// Find which φ-shell a magnitude belongs to
pub fn find_phi_shell(magnitude: f64, base: f64) -> i32 {
    if magnitude < base {
        return -1;
    }
    
    let ratio = magnitude / base;
    let shell = (ratio.ln() / PHI.ln()).floor() as i32;
    
    std::cmp::min(shell, (N_SHELLS - 1) as i32)
}

/// Julia set iteration
pub fn julia_iterate(mut z: num_complex::Complex64, c: num_complex::Complex64, max_iter: usize) -> (usize, bool) {
    for i in 0..max_iter {
        z = z * z + c;
        if z.norm() > JULIA_BOUND {
            return (i, false);
        }
    }
    (max_iter, true)
}

/// Compute phase coherence between two phase sequences
pub fn compute_phase_coherence(phases_a: &[f64], phases_b: &[f64]) -> f64 {
    if phases_a.len() != phases_b.len() || phases_a.is_empty() {
        return 0.0;
    }
    
    let n = phases_a.len() as f64;
    let mut sum_real = 0.0;
    let mut sum_imag = 0.0;
    
    for (a, b) in phases_a.iter().zip(phases_b.iter()) {
        let diff = a - b;
        sum_real += diff.cos();
        sum_imag += diff.sin();
    }
    
    (sum_real * sum_real + sum_imag * sum_imag).sqrt() / n
}

/// φ-shell decomposition result
#[derive(Debug, Clone)]
pub struct PhiShellResult {
    pub shell_indices: Vec<i32>,
    pub shell_counts: [usize; N_SHELLS],
    pub dominant_shell: usize,
    pub phi_coherence: f64,
}

/// Perform φ-shell decomposition on magnitude data
pub fn phi_decompose(magnitudes: &[f64], base_percentile: f64) -> PhiShellResult {
    let n = magnitudes.len();
    if n == 0 {
        return PhiShellResult {
            shell_indices: vec![],
            shell_counts: [0; N_SHELLS],
            dominant_shell: 0,
            phi_coherence: 0.0,
        };
    }
    
    // Compute base from percentile
    let mut sorted: Vec<f64> = magnitudes.iter().filter(|&&x| x > 0.0).cloned().collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let percentile_idx = ((sorted.len() as f64) * base_percentile / 100.0) as usize;
    let base = sorted.get(percentile_idx).copied().unwrap_or(1.0);
    
    // Assign shells
    let shell_indices: Vec<i32> = magnitudes.iter().map(|&m| find_phi_shell(m, base)).collect();
    
    // Count per shell
    let mut shell_counts = [0usize; N_SHELLS];
    for &shell in &shell_indices {
        if shell >= 0 && (shell as usize) < N_SHELLS {
            shell_counts[shell as usize] += 1;
        }
    }
    
    // Find dominant shell
    let dominant_shell = shell_counts.iter().enumerate()
        .max_by_key(|(_, &count)| count)
        .map(|(i, _)| i)
        .unwrap_or(0);
    
    // Compute coherence (simplified)
    let powers = phi_powers();
    let mut total_deviation = 0.0;
    let mut valid_count = 0;
    
    for (&mag, &shell) in magnitudes.iter().zip(&shell_indices) {
        if shell >= 0 && (shell as usize) < N_SHELLS {
            let expected = base * powers[shell as usize];
            total_deviation += (mag - expected).abs() / expected;
            valid_count += 1;
        }
    }
    
    let phi_coherence = if valid_count > 0 {
        (-total_deviation / valid_count as f64).exp()
    } else {
        0.0
    };
    
    PhiShellResult {
        shell_indices,
        shell_counts,
        dominant_shell,
        phi_coherence,
    }
}

/// Julia stability analysis result
#[derive(Debug, Clone)]
pub struct JuliaStabilityResult {
    pub stability_score: f64,
    pub n_stable: usize,
    pub n_unstable: usize,
}

/// Analyze stability of complex values using Julia iteration
pub fn julia_analyze(
    values: &[num_complex::Complex64],
    c: num_complex::Complex64,
    max_iter: usize,
) -> JuliaStabilityResult {
    let mut n_stable = 0;
    let mut n_unstable = 0;
    
    for &z in values {
        let (_, stable) = julia_iterate(z, c, max_iter);
        if stable {
            n_stable += 1;
        } else {
            n_unstable += 1;
        }
    }
    
    let total = n_stable + n_unstable;
    let stability_score = if total > 0 {
        n_stable as f64 / total as f64
    } else {
        0.0
    };
    
    JuliaStabilityResult {
        stability_score,
        n_stable,
        n_unstable,
    }
}

#[cfg(feature = "python")]
mod python {
    use pyo3::prelude::*;
    use super::*;
    
    #[pyfunction]
    fn py_verify_convergence() -> PyResult<(f64, f64, f64, f64, bool)> {
        let result = verify_convergence();
        Ok((result.phi_17, result.e_8_6, result.ratio, result.deviation_percent, result.is_valid))
    }
    
    #[pyfunction]
    fn py_find_phi_shell(magnitude: f64, base: f64) -> i32 {
        find_phi_shell(magnitude, base)
    }
    
    #[pymodule]
    fn cosf_core(_py: Python, m: &PyModule) -> PyResult<()> {
        m.add("PHI", PHI)?;
        m.add("COSF", COSF)?;
        m.add("JULIA_BOUND", JULIA_BOUND)?;
        m.add("N_SHELLS", N_SHELLS)?;
        m.add_function(wrap_pyfunction!(py_verify_convergence, m)?)?;
        m.add_function(wrap_pyfunction!(py_find_phi_shell, m)?)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_convergence() {
        let result = verify_convergence();
        assert!(result.is_valid);
        assert!(result.deviation_percent < 1.0);
    }
    
    #[test]
    fn test_phi_shell() {
        let shell = find_phi_shell(PHI * PHI, 1.0);
        assert_eq!(shell, 2);
    }
    
    #[test]
    fn test_julia_stable() {
        let z = num_complex::Complex64::new(0.1, 0.1);
        let c = num_complex::Complex64::new(-0.7, 0.27015);
        let (_, stable) = julia_iterate(z, c, 100);
        assert!(stable);
    }
}
