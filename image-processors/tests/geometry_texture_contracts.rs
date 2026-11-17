use image_processors::transforms::geometry_texture::{
    bake_pbr_texture, bake_pbr_texture_with_normals, cleanup_mesh, dual_grid_subdivision,
    flexible_dual_grid_to_mesh, mesh_to_flexible_dual_grid, remesh_narrow_band_dc, simplify_mesh,
    unify_face_orientations, vertex_normals, ChartOptions, FlexibleDualGrid, NormalSource,
    RemeshOptions, SparsePbrVolume, TriangleSurface,
};

fn quad_grid() -> FlexibleDualGrid {
    FlexibleDualGrid {
        coordinates: vec![[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],
        vertices: vec![[0.5; 3]; 4],
        intersected: vec![[true, false, false], [false; 3], [false; 3], [false; 3]],
    }
}

fn triangle() -> TriangleSurface {
    TriangleSurface {
        vertices: vec![[-0.25, -0.25, 0.0], [0.25, -0.25, 0.0], [-0.25, 0.25, 0.0]],
        faces: vec![[0, 1, 2]],
    }
}

fn cube() -> TriangleSurface {
    TriangleSurface {
        vertices: vec![
            [-0.25, -0.25, -0.25],
            [0.25, -0.25, -0.25],
            [0.25, 0.25, -0.25],
            [-0.25, 0.25, -0.25],
            [-0.25, -0.25, 0.25],
            [0.25, -0.25, 0.25],
            [0.25, 0.25, 0.25],
            [-0.25, 0.25, 0.25],
        ],
        faces: vec![
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [3, 7, 6],
            [3, 6, 2],
            [0, 4, 7],
            [0, 7, 3],
            [1, 2, 6],
            [1, 6, 5],
        ],
    }
}

#[test]
fn dual_grid_split_weights_choose_the_documented_diagonal() {
    let first = flexible_dual_grid_to_mesh(&quad_grid(), &[2.0, 1.0, 2.0, 1.0], 2)
        .expect("valid quad grid");
    assert_eq!(first.faces, vec![[0, 1, 2], [0, 2, 3]]);
    assert_eq!(first.vertices[0], [-0.25; 3]);
    let second = flexible_dual_grid_to_mesh(&quad_grid(), &[1.0, 2.0, 1.0, 2.0], 2)
        .expect("valid alternate diagonal");
    assert_eq!(second.faces, vec![[0, 1, 3], [3, 1, 2]]);
}

#[test]
fn dual_grid_rejects_invalid_public_buffers() {
    let mut duplicate = quad_grid();
    duplicate.coordinates[1] = duplicate.coordinates[0];
    assert!(flexible_dual_grid_to_mesh(&duplicate, &[1.0; 4], 2).is_err());
    let mut overflow = quad_grid();
    overflow.coordinates[0][0] = i64::MAX;
    assert!(flexible_dual_grid_to_mesh(&overflow, &[1.0; 4], 2).is_err());
    let mut nonfinite = quad_grid();
    nonfinite.vertices[0][0] = f32::NAN;
    assert!(flexible_dual_grid_to_mesh(&nonfinite, &[1.0; 4], 2).is_err());
    assert!(flexible_dual_grid_to_mesh(&quad_grid(), &[0.0; 4], 2).is_err());
}

#[test]
fn subdivision_uses_x_fast_bits_and_rejects_coordinate_overflow() {
    let occupancy = dual_grid_subdivision(&[[1, -1, 0]], &[[2, -2, 0], [3, -1, 1]])
        .expect("negative coordinates are valid for subdivision");
    assert_eq!(occupancy, vec![[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]);
    assert!(dual_grid_subdivision(&[[i64::MAX, 0, 0]], &[]).is_err());
}

#[test]
fn voxelization_preserves_a_known_planar_qef_solution() {
    let surface = TriangleSurface {
        vertices: vec![
            [-0.4, -0.4, 0.125],
            [0.4, -0.4, 0.125],
            [0.4, 0.4, 0.125],
            [-0.4, 0.4, 0.125],
        ],
        faces: vec![[0, 1, 2], [0, 2, 3]],
    };
    let grid = mesh_to_flexible_dual_grid(&surface, 8, 1.0, 1.0, 0.01)
        .expect("bounded plane supports scanline voxelization and constrained QEFs");
    assert!(!grid.coordinates.is_empty());
    for (coordinate, local) in grid.coordinates.iter().zip(&grid.vertices) {
        let z = (coordinate[2] as f32 + local[2]) / 8.0 - 0.5;
        assert!((z - 0.125).abs() < 2e-6);
    }
}

#[test]
fn hole_fill_preserves_the_boundary_and_adds_a_center_fan() {
    let mut surface = triangle();
    let original_vertices = surface.vertices.clone();
    surface
        .fill_holes(2.0)
        .expect("triangle perimeter is below two");
    assert_eq!(&surface.vertices[..3], original_vertices.as_slice());
    assert_eq!(surface.vertices.len(), 4);
    assert_eq!(surface.faces.len(), 4);
    surface.validate().expect("filled surface remains indexed");
}

#[test]
fn cleanup_removes_duplicate_faces_without_changing_the_surface() {
    let mut surface = triangle();
    surface.faces.push([2, 1, 0]);
    cleanup_mesh(&mut surface, 0.0, 0.0).expect("valid duplicate cleanup");
    assert_eq!(surface.faces.len(), 1);
    assert_eq!(surface.vertices.len(), 3);
}

#[test]
fn orientation_repair_reverses_a_same_direction_shared_edge() {
    let mut surface = TriangleSurface {
        vertices: vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        faces: vec![[0, 1, 2], [1, 2, 3]],
    };
    unify_face_orientations(&mut surface).expect("two manifold-connected triangles");
    assert_eq!(surface.faces, vec![[0, 1, 2], [1, 3, 2]]);
}

#[test]
fn simplification_reduces_a_closed_surface_to_the_requested_capacity() {
    let mut surface = cube();
    simplify_mesh(&mut surface, 8).expect("cube supports nonflipping edge collapses");
    assert!(!surface.faces.is_empty() && surface.faces.len() <= 8);
    surface
        .validate()
        .expect("simplified surface remains indexed");
}

#[test]
fn narrow_band_remeshing_projects_vertices_back_to_the_original_cube() {
    let surface = remesh_narrow_band_dc(
        &cube(),
        &RemeshOptions {
            resolution: 8,
            center: [0.0; 3],
            scale: 1.2,
            band: 0.5,
            project_back: 1.0,
        },
    )
    .expect("small closed cube can be remeshed");
    assert!(!surface.faces.is_empty());
    assert!(surface
        .vertices
        .iter()
        .all(|v| { (v.iter().map(|x| x.abs()).fold(0.0_f32, f32::max) - 0.25).abs() < 1e-5 }));
    surface
        .validate()
        .expect("remeshed surface remains indexed");
}

#[test]
fn pbr_baking_and_glb_preserve_material_channel_order() {
    let coordinates = (0..2)
        .flat_map(|x| (0..2).flat_map(move |y| (0..2).map(move |z| [x, y, z])))
        .collect::<Vec<_>>();
    let attributes = vec![[1.0, 0.0, 0.5, 0.25, 0.75, 1.0]; 8];
    let volume = SparsePbrVolume {
        coordinates: &coordinates,
        attributes: &attributes,
        resolution: 2,
    };
    let baked = bake_pbr_texture(
        &triangle(),
        Some(&[[0.1, 0.1], [0.9, 0.1], [0.1, 0.9]]),
        &volume,
        8,
    )
    .expect("explicit triangle UVs and complete constant PBR volume");
    let pixel = 9;
    for (&actual, expected) in baked.base_color[pixel * 4..pixel * 4 + 4]
        .iter()
        .zip([255_u8, 0, 127, 255])
    {
        assert!(actual.abs_diff(expected) <= 1);
    }
    for (&actual, expected) in baked.metallic_roughness[pixel * 3..pixel * 3 + 3]
        .iter()
        .zip([0_u8, 191, 63])
    {
        assert!(actual.abs_diff(expected) <= 1);
    }
    let glb = baked.to_glb().expect("PBR asset serializes as GLB");
    assert_eq!(&glb[..4], b"glTF");
    assert_eq!(
        u32::from_le_bytes(glb[8..12].try_into().expect("GLB length")) as usize,
        glb.len()
    );
    let json_length =
        u32::from_le_bytes(glb[12..16].try_into().expect("JSON chunk length")) as usize;
    let document: serde_json::Value =
        serde_json::from_slice(&glb[20..20 + json_length]).expect("GLB JSON chunk is valid");
    assert_eq!(document["asset"]["version"], "2.0");
    assert_eq!(
        document["images"]
            .as_array()
            .expect("embedded textures")
            .len(),
        2
    );
}

#[test]
fn conformal_chart_baking_produces_finite_packed_uvs() {
    let coordinates = (0..2)
        .flat_map(|x| (0..2).flat_map(move |y| (0..2).map(move |z| [x, y, z])))
        .collect::<Vec<_>>();
    let attributes = vec![[0.5; 6]; 8];
    let volume = SparsePbrVolume {
        coordinates: &coordinates,
        attributes: &attributes,
        resolution: 2,
    };
    let baked = bake_pbr_texture(&triangle(), None, &volume, 32)
        .expect("single triangle supports conformal parameterization and chart packing");
    assert_eq!(baked.surface.faces.len(), 1);
    assert_eq!(baked.uvs.len(), baked.surface.vertices.len());
    assert!(baked
        .uvs
        .iter()
        .flatten()
        .all(|v| v.is_finite() && (0.0..=1.0).contains(v)));
    assert_eq!(baked.base_color.len(), 32 * 32 * 4);
}

fn unequal_triangles() -> TriangleSurface {
    TriangleSurface {
        vertices: vec![[0.0; 3], [0.4, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.05]],
        faces: vec![[0, 1, 2], [0, 3, 1]],
    }
}

#[test]
fn existing_mesh_angle_normals_differ_from_generated_area_normals() {
    let surface = unequal_triangles();
    let angle = vertex_normals(&surface, NormalSource::CornerAngleWeighted)
        .expect("nondegenerate right angles");
    let area =
        vertex_normals(&surface, NormalSource::AreaWeighted).expect("nondegenerate triangle areas");
    let diagonal = std::f32::consts::FRAC_1_SQRT_2;
    assert!((angle[0][1] - diagonal).abs() < 1e-6);
    assert!((angle[0][2] - diagonal).abs() < 1e-6);
    assert!((area[0][1] - 0.25 / 1.0625_f32.sqrt()).abs() < 1e-6);
    assert!((area[0][2] - 1.0 / 1.0625_f32.sqrt()).abs() < 1e-6);
}

#[test]
fn supplied_normals_survive_chart_expansion_and_unused_vertex_cleanup() {
    let mut surface = unequal_triangles();
    surface.vertices.insert(0, [0.49; 3]);
    for face in &mut surface.faces {
        for index in face {
            *index += 1;
        }
    }
    let normals = [
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
    ];
    let coordinates = (0..2)
        .flat_map(|x| (0..2).flat_map(move |y| (0..2).map(move |z| [x, y, z])))
        .collect::<Vec<_>>();
    let attributes = vec![[0.5; 6]; 8];
    let volume = SparsePbrVolume {
        coordinates: &coordinates,
        attributes: &attributes,
        resolution: 2,
    };
    let baked = bake_pbr_texture_with_normals(
        &surface,
        None,
        &volume,
        32,
        &ChartOptions::default(),
        NormalSource::Supplied(&normals),
    )
    .expect("supplied original-index normals survive UV seam expansion");
    assert_eq!(baked.surface.faces.len(), 2);
    for (position, normal) in baked.surface.vertices.iter().zip(&baked.normals) {
        let original = surface
            .vertices
            .iter()
            .position(|p| p == position)
            .expect("UV expansion retains original vertex positions");
        assert_ne!(original, 0, "unreferenced vertex is not part of the chart");
        assert_eq!(*normal, normals[original]);
    }
}

#[test]
fn invalid_supplied_normal_buffers_are_rejected() {
    let surface = unequal_triangles();
    assert!(vertex_normals(&surface, NormalSource::Supplied(&[])).is_err());
    let mut normals = [[0.0, 0.0, 1.0]; 4];
    normals[2][1] = f32::NAN;
    assert!(vertex_normals(&surface, NormalSource::Supplied(&normals)).is_err());
}
