//! SRMD blur-kernel conditioning with the original MATLAB PCA convention.

use thiserror::Error;

/// Invalid SRMD conditioning data.
#[derive(Debug, Error)]
pub enum SrmdConditioningError {
    /// The original PCA projection has fifteen rows and 225 columns.
    #[error("SRMD PCA projection needs 3375 row-major values, received {actual}")]
    ProjectionLength {
        /// Number of supplied matrix elements.
        actual: usize,
    },
    /// Projection or kernel data must be finite.
    #[error("SRMD conditioning contains a nonfinite value")]
    NonFinite,
    /// A Gaussian kernel needs a finite positive sigma with nonzero variance.
    #[error("SRMD Gaussian sigma must be finite and positive with representable variance")]
    Sigma,
}

/// The original SRMD fifteen-component PCA projection.
///
/// Matrix storage is row-major `[15, 225]`. Its columns correspond to MATLAB's
/// column-major vectorization of a spatial `[15, 15]` kernel, as used by KAIR
/// `main_test_srmd.py`. The matrix is caller supplied; this type does not decode
/// MATLAB files or regenerate the trained PCA basis.
#[derive(Clone, Debug)]
pub struct SrmdPcaProjection {
    matrix: Box<[f64]>,
}

impl SrmdPcaProjection {
    /// Retain the original projection in FP64, without changing its basis.
    ///
    /// # Errors
    /// Rejects a matrix with the wrong length or nonfinite elements.
    pub fn new(matrix: &[f64]) -> Result<Self, SrmdConditioningError> {
        if matrix.len() != 15 * 225 {
            return Err(SrmdConditioningError::ProjectionLength {
                actual: matrix.len(),
            });
        }
        if matrix.iter().any(|value| !value.is_finite()) {
            return Err(SrmdConditioningError::NonFinite);
        }
        Ok(Self {
            matrix: matrix.into(),
        })
    }

    /// Project a caller-supplied row-major spatial kernel using MATLAB ordering.
    ///
    /// The supplied kernel retains its values: upstream Gaussian constructors
    /// normalize before this operation. Multiplication and accumulation remain
    /// FP64 until the final FP32 conversion made by upstream `Tensor.float()`.
    ///
    /// # Errors
    /// Rejects nonfinite kernels or a nonfinite/unrepresentable projected value.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "KAIR converts the FP64 PCA result to FP32"
    )]
    pub fn project_kernel(&self, kernel: &[f64; 225]) -> Result<[f32; 15], SrmdConditioningError> {
        if kernel.iter().any(|value| !value.is_finite()) {
            return Err(SrmdConditioningError::NonFinite);
        }
        let mut coefficients = [0.0_f32; 15];
        for (component, row) in coefficients
            .iter_mut()
            .zip(self.matrix.as_chunks::<225>().0)
        {
            let mut sum = 0.0_f64;
            for (index, weight) in row.iter().enumerate() {
                // MATLAB vec(kernel) traverses rows within each column.
                let spatial_index = (index % 15) * 15 + index / 15;
                sum += weight * kernel[spatial_index];
            }
            *component = sum as f32;
            if !component.is_finite() {
                return Err(SrmdConditioningError::NonFinite);
            }
        }
        Ok(coefficients)
    }
}

/// Construct the normalized 15-by-15 Gaussian used by KAIR SRMD inference.
///
/// Values are row-major FP64. The coefficient cutoff follows upstream
/// `utils_deblur.fspecial_gaussian`: values below epsilon times the maximum
/// are zeroed before normalization. Sigma `0.01` gives the default delta kernel.
///
/// # Errors
/// Rejects nonpositive/nonfinite sigma or variance underflow.
pub fn srmd_gaussian_kernel(sigma: f64) -> Result<[f64; 225], SrmdConditioningError> {
    let denominator = 2.0 * sigma * sigma;
    if !sigma.is_finite() || sigma <= 0.0 || denominator == 0.0 {
        return Err(SrmdConditioningError::Sigma);
    }
    let mut kernel = [0.0_f64; 225];
    for row in -7_i32..=7 {
        for column in -7_i32..=7 {
            let distance = f64::from(row * row + column * column);
            let value = (-distance / denominator).exp();
            let index = usize::try_from((row + 7) * 15 + column + 7)
                .map_err(|_| SrmdConditioningError::Sigma)?;
            kernel[index] = if value < f64::EPSILON { 0.0 } else { value };
        }
    }
    let total: f64 = kernel.iter().sum();
    for value in &mut kernel {
        *value /= total;
    }
    Ok(kernel)
}
