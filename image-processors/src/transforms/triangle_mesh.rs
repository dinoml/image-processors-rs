//! Dependency-free FP64 ray/triangle intersections for image-space depth output.

use thiserror::Error;

/// Invalid mesh geometry supplied to the ray intersector.
#[derive(Debug, Error)]
pub enum MeshIntersectionError {
    /// Mesh vertices must be finite.
    #[error("mesh vertices must be finite")]
    NonfiniteVertex,
    /// A triangle referenced a vertex outside the supplied vertex array.
    #[error("triangle {triangle} references missing vertex {vertex}")]
    InvalidIndex {
        /// Triangle index.
        triangle: usize,
        /// Invalid vertex index.
        vertex: usize,
    },
}

/// The nearest forward intersection. Interior-facing hits remain distinguishable
/// so callers can reject them *after* nearest-hit selection.
#[derive(Clone, Copy, Debug)]
pub struct RayHit {
    /// Original triangle index.
    pub triangle: usize,
    /// Ray parameter, in units of the supplied direction vector.
    pub distance: f64,
    /// Intersection in the mesh's coordinate system.
    pub position: [f64; 3],
    /// Whether the oriented triangle normal opposes the ray direction.
    pub front_facing: bool,
}

/// A borrowed, validated triangle mesh. No weights or external geometry library
/// are needed; degenerate triangles do not intersect rays.
pub struct TriangleMesh<'a> {
    vertices: &'a [[f64; 3]],
    faces: &'a [[usize; 3]],
}

impl<'a> TriangleMesh<'a> {
    /// Borrow mesh arrays and validate indices and coordinates.
    ///
    /// # Errors
    /// Reports nonfinite vertices or out-of-range triangle indices.
    pub fn new(
        vertices: &'a [[f64; 3]],
        faces: &'a [[usize; 3]],
    ) -> Result<Self, MeshIntersectionError> {
        if vertices.iter().flatten().any(|value| !value.is_finite()) {
            return Err(MeshIntersectionError::NonfiniteVertex);
        }
        for (triangle, face) in faces.iter().enumerate() {
            for &vertex in face {
                if vertex >= vertices.len() {
                    return Err(MeshIntersectionError::InvalidIndex { triangle, vertex });
                }
            }
        }
        Ok(Self { vertices, faces })
    }

    /// Find the nearest forward ray/triangle intersection in FP64.
    ///
    /// Uses trimesh's FP64 Cramer barycentrics (strict 1e-13 edge tolerance),
    /// 1e-5 plane-direction threshold and -1e-6 forward-projection tolerance.
    /// Both face orientations participate in nearest-hit selection.
    /// Nonfinite and zero-length rays have no intersection. Directions need not
    /// be normalized; returned distance is the multiplier of that direction.
    pub fn intersect_first(&self, origin: [f64; 3], direction: [f64; 3]) -> Option<RayHit> {
        if origin
            .iter()
            .chain(direction.iter())
            .any(|value| !value.is_finite())
            || dot(direction, direction) == 0.0
        {
            return None;
        }
        let mut nearest: Option<RayHit> = None;
        for (triangle, face) in self.faces.iter().enumerate() {
            let first = self.vertices[face[0]];
            let edge1 = subtract(self.vertices[face[1]], first);
            let edge2 = subtract(self.vertices[face[2]], first);
            let normal = cross(edge1, edge2);
            let normal_length = dot(normal, normal).sqrt();
            if normal_length == 0.0 {
                continue;
            }
            let normal = normal.map(|value| value / normal_length);
            let denominator = dot(normal, direction);
            if denominator.abs() <= 1e-5 {
                continue;
            }
            let distance = dot(normal, subtract(first, origin)) / denominator;
            if nearest.is_some_and(|hit| distance >= hit.distance) {
                continue;
            }
            let position = std::array::from_fn(|axis| origin[axis] + distance * direction[axis]);
            if dot(subtract(position, origin), direction) <= -1e-6 {
                continue;
            }
            let offset = subtract(position, first);
            let dot11 = dot(edge1, edge1);
            let dot12 = dot(edge1, edge2);
            let dot22 = dot(edge2, edge2);
            let determinant = dot11 * dot22 - dot12 * dot12;
            if determinant <= 0.0 {
                continue;
            }
            let dot1p = dot(edge1, offset);
            let dot2p = dot(edge2, offset);
            let inverse = 1.0 / determinant;
            let second = (dot22 * dot1p - dot12 * dot2p) * inverse;
            let third = (dot11 * dot2p - dot12 * dot1p) * inverse;
            let first = 1.0 - second - third;
            if [first, second, third]
                .iter()
                .all(|&value| value > -1e-13 && value < 1.0 + 1e-13)
            {
                nearest = Some(RayHit {
                    triangle,
                    distance,
                    position,
                    front_facing: denominator <= 0.0,
                });
            }
        }
        nearest
    }
}

fn subtract(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|axis| left[axis] - right[axis])
}

fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn cross(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}
