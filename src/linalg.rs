//! Small dense symmetric eigensolver shared by the zero-dependency subspace
//! recognisers ([`crate::eigenface`] and [`crate::fisherface`]).
//!
//! Cyclic Jacobi rotations are exact enough for gallery-sized matrices (`n` below a
//! few hundred): no LAPACK, no BLAS, no third-party crate behind an optional feature.

/// Stop a sweep when the largest off-diagonal element is below
/// `|λ_max| * EIGEN_TOL_REL`.
pub(crate) const EIGEN_TOL_REL: f32 = 1e-10;

/// Maximum Jacobi rotation sweeps (30 is ample for the small gallery matrices here).
pub(crate) const JACOBI_SWEEPS: usize = 30;

/// Cyclic Jacobi eigenvalue algorithm on a packed `n x n` slice (row-major).
///
/// On return the matrix is (near-)diagonal and `(eigenvalues, eigenvectors)` is
/// returned; eigenvector `k` is column `k` of the returned flat matrix, eigenvalues
/// sorted **descending**. Input is symmetrised defensively in case of rounding drift.
pub(crate) fn jacobi_symmetric(a: &mut [f32], n: usize) -> (Vec<f32>, Vec<f32>) {
    for i in 0..n {
        for j in 0..i {
            let avg = (a[i * n + j] + a[j * n + i]) * 0.5;
            a[i * n + j] = avg;
            a[j * n + i] = avg;
        }
    }
    let mut v = vec![0.0f32; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }

    for _ in 0..JACOBI_SWEEPS {
        let mut max_off = 0.0f32;
        for p in 0..n {
            for q in (p + 1)..n {
                max_off = max_off.max(a[p * n + q].abs());
            }
        }
        let diag_scale = (0..n).map(|i| a[i * n + i].abs()).fold(0.0f32, f32::max);
        if max_off <= diag_scale.max(1.0) * EIGEN_TOL_REL {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p * n + q];
                if apq.abs() <= f32::EPSILON {
                    continue;
                }
                let app = a[p * n + p];
                let aqq = a[q * n + q];
                // tau = (a_qq - a_pp) / (2 a_pq); t solves t^2 + 2 tau t = 1 with |t|<=1.
                let tau = (aqq - app) / (2.0 * apq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                for k in 0..n {
                    let akp = a[k * n + p];
                    let akq = a[k * n + q];
                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let apk = a[p * n + k];
                    let aqk = a[q * n + k];
                    a[p * n + k] = c * apk - s * aqk;
                    a[q * n + k] = s * apk + c * aqk;
                }
                for k in 0..n {
                    let vkp = v[k * n + p];
                    let vkq = v[k * n + q];
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }

    let eigvals: Vec<f32> = (0..n).map(|i| a[i * n + i]).collect();
    // Sort eigenvalues (and columns) descending, strongest principal axis first.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| eigvals[j].total_cmp(&eigvals[i]));
    let sorted_vals = order.iter().map(|&i| eigvals[i]).collect();
    let mut sorted_vecs = vec![0.0f32; n * n];
    for (out_col, &src_col) in order.iter().enumerate() {
        for row in 0..n {
            sorted_vecs[row * n + out_col] = v[row * n + src_col];
        }
    }
    (sorted_vals, sorted_vecs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jacobi_recovers_known_2x2_eigenvalues() {
        // diag(2, 3): already diagonal.
        let mut diag = vec![2.0, 0.0, 0.0, 3.0];
        let (vals, _) = jacobi_symmetric(&mut diag, 2);
        assert!((vals[0] - 3.0).abs() < 1e-5);
        assert!((vals[1] - 2.0).abs() < 1e-5);

        // [[2,1],[1,2]] eigenvalues {3, 1}, eigenvectors (1,1)/sqrt2 etc.
        let mut mix = vec![2.0, 1.0, 1.0, 2.0];
        let (vals, vecs) = jacobi_symmetric(&mut mix, 2);
        assert!((vals[0] - 3.0).abs() < 1e-5);
        assert!((vals[1] - 1.0).abs() < 1e-5);
        // Columns stay orthonormal.
        let dot = vecs[0] * vecs[1] + vecs[2] * vecs[3];
        assert!(dot.abs() < 1e-5);
        let n0 = (vecs[0] * vecs[0] + vecs[2] * vecs[2]).sqrt();
        assert!((n0 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn jacobi_on_3x3_matches_characteristic_values() {
        // Symmetric matrix with known spectrum.
        let mut a = vec![
            4.0, 1.0, 2.0, //
            1.0, 2.0, 0.0, //
            2.0, 0.0, 3.0,
        ];
        let (vals, _) = jacobi_symmetric(&mut a, 3);
        // Trace is preserved and eigenvalues sum to it.
        assert!((vals.iter().sum::<f32>() - 9.0).abs() < 1e-4);
        // Determinant is the product of eigenvalues: det(A) = 24 - 3 - 8 = 13.
        assert!((vals[0] * vals[1] * vals[2] - 13.0).abs() < 1e-3);
    }
}
