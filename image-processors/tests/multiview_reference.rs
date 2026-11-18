use image_processors::transforms::geometry_texture::multiview::{
    blend, poisson_blend, Camera, Projection,
};
use serde_json::Value;

fn fixtures() -> Value {
    serde_json::from_str(include_str!("fixtures/multiview_cpu_reference.json")).unwrap()
}
fn f(v: &Value) -> f32 {
    v.as_f64().unwrap() as f32
}
fn triples(v: &Value) -> Vec<[f32; 3]> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|p| [f(&p[0]), f(&p[1]), f(&p[2])])
        .collect()
}
fn close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= tolerance,
            "index {i}: actual={a}, reference={e}"
        );
    }
}
#[test]
fn orbit_and_conditioning_match_original_cpu_torch() {
    for case in fixtures()["cameras"].as_array().unwrap() {
        let camera = Camera::orbit(
            f(&case["elevation"]),
            f(&case["azimuth"]),
            f(&case["distance"]),
            1.1,
            None,
        )
        .unwrap();
        for (column, actual) in [camera.right, camera.up, camera.back, camera.position]
            .iter()
            .enumerate()
        {
            let expected: Vec<_> = (0..3).map(|row| f(&case["c2w"][row][column])).collect();
            close(actual, &expected, 2e-6);
        }
        let expected: Vec<_> = case["conditioning"]
            .as_array()
            .unwrap()
            .iter()
            .map(f)
            .collect();
        close(&camera.plucker(2).unwrap(), &expected, 2e-6);
    }
    assert!(Camera::orbit(0., 0., 0., 1., None).is_err());
    assert!(Camera::orbit(0., 0., 1., 1., Some(180.)).is_err());
}
#[test]
fn cosine_blend_matches_original_torch_including_unseen_texels() {
    for case in fixtures()["blends"].as_array().unwrap() {
        let views: Vec<_> = (0..2)
            .map(|v| Projection {
                colors: triples(&case["colors"][v]),
                valid: case["valid"][v]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_bool().unwrap())
                    .collect(),
                incidence: case["incidence"][v]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(f)
                    .collect(),
            })
            .collect();
        let (actual, known) = blend(&views, f(&case["alpha"]), false).unwrap();
        let expected = triples(&case["expected"]);
        close(actual.as_flattened(), expected.as_flattened(), 2e-6);
        assert_eq!(known, vec![true, true, false]);
    }
}
#[test]
fn finite_iteration_poisson_matches_original_torch_native() {
    for case in fixtures()["poisson"].as_array().unwrap() {
        let source = triples(&case["source"]);
        let target = triples(&case["target"]);
        let mask: Vec<_> = case["mask"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_bool().unwrap())
            .collect();
        let expected = triples(&case["expected"]);
        let actual = poisson_blend(
            &source,
            &target,
            &mask,
            case["size"].as_u64().unwrap() as usize,
            case["iterations"].as_u64().unwrap() as usize,
        )
        .unwrap();
        close(actual.as_flattened(), expected.as_flattened(), 2e-6);
    }
}

#[test]
fn rasterization_depth_uv_and_reprojection_preserve_coordinate_contracts() {
    use image_processors::transforms::geometry_texture::multiview::{
        atlas, project, render, ProjectionOptions,
    };
    use image_processors::transforms::geometry_texture::{
        ChartOptions, NormalSource, TriangleSurface,
    };
    // Analytic two-layer triangle: the center pixel is world origin in the far
    // layer and [0,0,1] in the near layer, independent of submission order.
    let mut surface = TriangleSurface {
        vertices: vec![
            [-0.5, -0.5, 0.],
            [0.5, -0.5, 0.],
            [0., 0.5, 0.],
            [-0.5, -0.5, 1.],
            [0.5, -0.5, 1.],
            [0., 0.5, 1.],
        ],
        faces: vec![[0, 1, 2], [3, 4, 5]],
    };
    let camera = Camera {
        position: [0., 0., 2.],
        right: [1., 0., 0.],
        up: [0., 1., 0.],
        back: [0., 0., 1.],
        scale: 2.,
        fov_degrees: None,
    };
    for _ in 0..2 {
        let view = render(&surface, &camera, 3, NormalSource::AreaWeighted).unwrap();
        assert!(view.mask[4]);
        close(&view.positions[4], &[0., 0., 1.], 1e-6);
        close(&view.normals[4], &[0., 0., 1.], 1e-6);
        assert_eq!(view.depth[4], 1.);
        assert!(!view.mask[0]);
        assert!(view.depth[0].is_infinite());
        surface.faces.reverse();
    }
    surface.vertices.truncate(3);
    surface.faces = vec![[0, 1, 2]];
    let atlas = atlas(
        &surface,
        Some(&[[0., 0.], [1., 0.], [0.5, 1.]]),
        3,
        &ChartOptions::default(),
    )
    .unwrap();
    close(&atlas.positions[4], &[0., 0., 0.], 1e-6);
    assert!(atlas.mask[4]);
    let view = render(&surface, &camera, 3, NormalSource::AreaWeighted).unwrap();
    let colors = vec![[0.2, 0.4, 0.6]; 9];
    let projection = project(
        &atlas,
        &camera,
        &view,
        &colors,
        &ProjectionOptions {
            position_tolerance: 1e-4,
            incidence_threshold: 0.1,
            depth_gradient_threshold: None,
            depth_gradient_dilation: 1,
        },
    )
    .unwrap();
    assert!(projection.valid[4]);
    close(&projection.colors[4], &[0.2, 0.4, 0.6], 1e-6);
    assert_eq!(projection.incidence[4], 1.);
}
#[test]
fn canonical_mesh_roundtrip_and_invalid_geometry_are_explicit() {
    use image_processors::transforms::geometry_texture::multiview::canonical_mesh;
    let vertices = [[1., 2., 3.], [5., 2., 3.], [1., 6., 3.]];
    for center in [false, true] {
        for rotate in [false, true] {
            let (mut surface, transform) =
                canonical_mesh(&vertices, &[[0, 1, 2]], center, 0.5, rotate).unwrap();
            let max = surface
                .vertices
                .iter()
                .flatten()
                .map(|v| v.abs())
                .fold(0_f32, f32::max);
            assert!((max - 0.5).abs() < 1e-6);
            transform.restore(&mut surface);
            for (a, e) in surface.vertices.iter().zip(vertices) {
                close(a, &e.map(|v| v as f32), 1e-6);
            }
        }
    }
    assert!(canonical_mesh(&vertices, &[[0, 1, 3]], true, 0.5, false).is_err());
    assert!(canonical_mesh(&[[0.; 3]; 3], &[[0, 1, 2]], true, 0.5, false).is_err());
    assert!(canonical_mesh(&vertices, &[[0, 1, 2]], true, f64::NAN, false).is_err());
}
