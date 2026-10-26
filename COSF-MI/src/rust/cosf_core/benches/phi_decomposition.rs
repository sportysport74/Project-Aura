//! Benchmarks for φ-shell decomposition and Julia stability analysis.
//!
//! Run with: cargo bench

use cosf_core::{julia_analyze, phi_decompose, PHI};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use num_complex::Complex64;

/// Deterministic magnitudes spread over several φ-shells (no RNG dependency).
fn magnitudes(n: usize) -> Vec<f64> {
    (0..n).map(|i| 1.0 + PHI.powf((i % 1000) as f64 / 100.0)).collect()
}

fn bench_phi_decompose(c: &mut Criterion) {
    let mags = magnitudes(1 << 20);
    c.bench_function("phi_decompose 1M", |b| b.iter(|| phi_decompose(black_box(&mags), 10.0)));
}

fn bench_julia_analyze(c: &mut Criterion) {
    let values: Vec<Complex64> = (0..10_000)
        .map(|i| Complex64::new((i % 100) as f64 / 100.0 - 0.5, (i / 100) as f64 / 100.0 - 0.5))
        .collect();
    let k = Complex64::new(-0.7, 0.27015);
    c.bench_function("julia_analyze 10k", |b| b.iter(|| julia_analyze(black_box(&values), k, 100)));
}

criterion_group!(benches, bench_phi_decompose, bench_julia_analyze);
criterion_main!(benches);
