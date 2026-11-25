use super::super::*;

#[test]
fn capped_resize_even_matches_pinned_original_dimensions() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("capped_resize_even_original.json"))
            .expect("original DINO geometry fixture must parse");
    let cases = fixture["cases"].as_array().expect("fixture cases");
    assert_eq!(cases.len(), 311);
    for case in cases {
        let dimension = |key: &str| {
            usize::try_from(case[key].as_u64().expect("dimension")).expect("usize dimension")
        };
        let input =
            ImageSize::new(dimension("height"), dimension("width")).expect("valid source size");
        let cap = case["cap"]
            .as_u64()
            .map(|n| usize::try_from(n).expect("usize cap"));
        let expected = case["expected"]
            .as_array()
            .expect("expected height and width");
        let height = usize::try_from(expected[0].as_u64().expect("height")).expect("usize height");
        let width = usize::try_from(expected[1].as_u64().expect("width")).expect("usize width");
        let actual = shortest_edge_resize_size_capped_ties_even(input, dimension("shortest"), cap);
        if height == 0 || width == 0 {
            assert!(
                actual.is_err(),
                "zero-sized original resampling must be rejected: {case}"
            );
        } else {
            assert_eq!(
                actual.expect("valid capped resize"),
                ImageSize::new(height, width).expect("valid expected size"),
                "{case}"
            );
        }
    }
}

#[test]
fn capped_resize_even_preserves_existing_half_up_profile() {
    let input = ImageSize::new(5, 8).expect("valid size");
    assert_eq!(
        shortest_edge_resize_size_capped_ties_even(input, 8, Some(4)).expect("new geometry"),
        ImageSize::new(2, 3).expect("expected size")
    );
    assert_eq!(
        shortest_edge_resize_size(input, 8, Some(4)).expect("existing geometry"),
        ImageSize::new(3, 4).expect("existing expected size")
    );
    assert!(shortest_edge_resize_size_capped_ties_even(input, 0, None).is_err());
    assert!(shortest_edge_resize_size_capped_ties_even(input, 8, Some(0)).is_err());
    assert!(shortest_edge_resize_size_capped_ties_even(
        ImageSize::new(1, 3).expect("valid size"),
        usize::MAX,
        None
    )
    .is_err());
}
