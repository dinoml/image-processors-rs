//! FP64 landmark fitting and homogeneous camera geometry.

use super::{FaceGeometryError, ImageSize};

/// Fit a full 3D affine map using normalized homogeneous total least squares.
///
/// Source and destination coordinates are centered and scaled to RMS distance
/// sqrt(3), as in scikit-image's affine estimator. A one-sided Jacobi SVD keeps
/// the least right-singular vector without squaring the condition number.
/// The returned row-major matrix maps source homogeneous column vectors.
///
/// # Errors
/// Rejects mismatched, nonfinite, insufficient or degenerate correspondences.
pub fn estimate_affine_3d(
    from: &[[f64; 3]],
    to: &[[f64; 3]],
) -> Result<[[f64; 4]; 4], FaceGeometryError> {
    if from.len() != to.len()
        || from.len() < 4
        || from.iter().chain(to).flatten().any(|v| !v.is_finite())
    {
        return Err(FaceGeometryError::Landmarks);
    }
    let count = from.len() as f64;
    let mut centers = [[0.0; 3]; 2];
    let mut scales = [0.0; 2];
    for (index, points) in [from, to].into_iter().enumerate() {
        for point in points {
            for axis in 0..3 {
                centers[index][axis] += point[axis];
            }
        }
        for value in &mut centers[index] {
            *value /= count;
        }
        let energy: f64 = points
            .iter()
            .flat_map(|p| (0..3).map(move |a| (p[a] - centers[index][a]).powi(2)))
            .sum();
        if energy == 0.0 || !energy.is_finite() {
            return Err(FaceGeometryError::Landmarks);
        }
        scales[index] = 3.0_f64.sqrt() / (energy / count).sqrt();
    }
    let mut equations = Vec::<[f64; 13]>::with_capacity(from.len() * 3);
    for (source, destination) in from.iter().zip(to) {
        for axis in 0..3 {
            let mut row = [0.0; 13];
            for column in 0..3 {
                row[axis * 4 + column] = (source[column] - centers[0][column]) * scales[0];
            }
            row[axis * 4 + 3] = 1.0;
            row[12] = -(destination[axis] - centers[1][axis]) * scales[1];
            equations.push(row);
        }
    }
    let mut right = [[0.0; 13]; 13];
    for (index, row) in right.iter_mut().enumerate() {
        row[index] = 1.0;
    }
    for _ in 0..128 {
        let mut changed = false;
        for p in 0..13 {
            for q in p + 1..13 {
                let (mut alpha, mut beta, mut gamma) = (0.0_f64, 0.0_f64, 0.0_f64);
                for row in &equations {
                    alpha += row[p] * row[p];
                    beta += row[q] * row[q];
                    gamma += row[p] * row[q];
                }
                if gamma.abs() <= f64::EPSILON * (alpha * beta).sqrt() * 8.0 {
                    continue;
                }
                let zeta = (beta - alpha) / (2.0 * gamma);
                let tangent = zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                let tangent = if zeta == 0.0 { 1.0 } else { tangent };
                let cosine = 1.0 / (1.0 + tangent * tangent).sqrt();
                let sine = cosine * tangent;
                for row in equations.iter_mut().chain(right.iter_mut()) {
                    let (a, b) = (row[p], row[q]);
                    row[p] = cosine * a - sine * b;
                    row[q] = sine * a + cosine * b;
                }
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    let mut norms = [0.0; 13];
    for row in &equations {
        for i in 0..13 {
            norms[i] += row[i] * row[i];
        }
    }
    let mut order: Vec<usize> = (0..13).collect();
    order.sort_by(|a, b| norms[*a].total_cmp(&norms[*b]));
    let smallest = order[0];
    if norms[order[1]] <= norms[order[12]] * f64::EPSILON.powi(2)
        || right[12][smallest].abs() <= f64::EPSILON
    {
        return Err(FaceGeometryError::Landmarks);
    }
    let mut matrix = [[0.0; 4]; 4];
    matrix[3][3] = 1.0;
    for axis in 0..3 {
        let divisor = right[12][smallest];
        matrix[axis][3] = centers[1][axis] + right[axis * 4 + 3][smallest] / divisor / scales[1];
        for column in 0..3 {
            matrix[axis][column] =
                right[axis * 4 + column][smallest] / divisor * scales[0] / scales[1];
            matrix[axis][3] -= matrix[axis][column] * centers[0][column];
        }
    }
    if matrix.iter().flatten().any(|v| !v.is_finite()) {
        return Err(FaceGeometryError::Singular);
    }
    Ok(matrix)
}

/// Invert a finite row-major homogeneous 3D transform with partial pivoting.
/// # Errors
/// Rejects nonfinite or singular matrices.
pub fn invert_transform_3d(matrix: [[f64; 4]; 4]) -> Result<[[f64; 4]; 4], FaceGeometryError> {
    if matrix.iter().flatten().any(|v| !v.is_finite()) {
        return Err(FaceGeometryError::Geometry);
    }
    let mut augmented = [[0.0; 8]; 4];
    for i in 0..4 {
        augmented[i][..4].copy_from_slice(&matrix[i]);
        augmented[i][4 + i] = 1.0;
    }
    for column in 0..4 {
        let pivot = (column..4)
            .max_by(|a, b| {
                augmented[*a][column]
                    .abs()
                    .total_cmp(&augmented[*b][column].abs())
            })
            .ok_or(FaceGeometryError::Singular)?;
        if augmented[pivot][column] == 0.0 {
            return Err(FaceGeometryError::Singular);
        }
        augmented.swap(column, pivot);
        let scale = augmented[column][column];
        for value in &mut augmented[column] {
            *value /= scale;
        }
        let row = augmented[column];
        for (index, target) in augmented.iter_mut().enumerate() {
            if index == column {
                continue;
            }
            let factor = target[column];
            for j in 0..8 {
                target[j] -= factor * row[j];
            }
        }
    }
    let mut inverse = [[0.0; 4]; 4];
    for i in 0..4 {
        inverse[i].copy_from_slice(&augmented[i][4..]);
    }
    if inverse.iter().flatten().any(|v| !v.is_finite()) {
        return Err(FaceGeometryError::Singular);
    }
    Ok(inverse)
}

/// Transform 3D points, retaining FP64 coordinates and homogeneous division.
/// # Errors
/// Rejects nonfinite geometry and points whose homogeneous divisor is zero.
pub fn transform_points_3d(
    points: &[[f64; 3]],
    matrix: [[f64; 4]; 4],
) -> Result<Vec<[f64; 3]>, FaceGeometryError> {
    if matrix.iter().flatten().any(|v| !v.is_finite()) {
        return Err(FaceGeometryError::Geometry);
    }
    points
        .iter()
        .map(|point| {
            if point.iter().any(|v| !v.is_finite()) {
                return Err(FaceGeometryError::Landmarks);
            }
            let p = [point[0], point[1], point[2], 1.0];
            let value = matrix.map(|row| row.iter().zip(p).map(|(a, b)| a * b).sum::<f64>());
            if value[3] == 0.0 {
                return Err(FaceGeometryError::Singular);
            }
            let result = [
                value[0] / value[3],
                value[1] / value[3],
                value[2] / value[3],
            ];
            if result.iter().any(|v| !v.is_finite()) {
                return Err(FaceGeometryError::Geometry);
            }
            Ok(result)
        })
        .collect()
}

/// Project homogeneous 3D points from normalized device space into pixel space.
/// The caller supplies the composed camera matrix; coordinates remain unclipped.
/// # Errors
/// Rejects empty image geometry, invalid transforms and points at infinity.
pub fn project_points_3d(
    points: &[[f64; 3]],
    matrix: [[f64; 4]; 4],
    size: ImageSize,
) -> Result<Vec<[f64; 2]>, FaceGeometryError> {
    if size.width == 0 || size.height == 0 {
        return Err(FaceGeometryError::Geometry);
    }
    Ok(transform_points_3d(points, matrix)?
        .into_iter()
        .map(|point| {
            [
                (point[0] + 1.0) * 0.5 * size.width as f64,
                (point[1] + 1.0) * 0.5 * size.height as f64,
            ]
        })
        .collect())
}
