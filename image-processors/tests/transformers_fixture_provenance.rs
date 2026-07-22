use std::collections::BTreeSet;

use image_processors::{CompatibilityStatus, ProcessorCatalog, UpstreamLibrary};
use serde::Deserialize;
use serde_json::{Map, Value};

const FIXTURE_SCHEMA: &str = "image-processors.transformers-parity.v1";
const GENERATOR: &str = "scripts/parity/transformers_image_parity.py";
const UPSTREAM_LIBRARY: &str = "transformers";
const UPSTREAM_SOURCE: &str = "transformers.image_processing";
const MAX_ABSOLUTE_TOLERANCE: f64 = 0.06;
const MAX_RELATIVE_TOLERANCE: f64 = 0.0;

type FixtureSpec = (&'static str, &'static str);

const FIXTURES: &[FixtureSpec] = &[
    (
        "transformers_clip_vit.json",
        include_str!("fixtures/transformers_clip_vit.json"),
    ),
    (
        "transformers_detr.json",
        include_str!("fixtures/transformers_detr.json"),
    ),
    (
        "transformers_donut.json",
        include_str!("fixtures/transformers_donut.json"),
    ),
    (
        "transformers_gemma3.json",
        include_str!("fixtures/transformers_gemma3.json"),
    ),
    (
        "transformers_idefics3.json",
        include_str!("fixtures/transformers_idefics3.json"),
    ),
    (
        "transformers_llava_next.json",
        include_str!("fixtures/transformers_llava_next.json"),
    ),
    (
        "transformers_llava_next_nhwc.json",
        include_str!("fixtures/transformers_llava_next_nhwc.json"),
    ),
    (
        "transformers_mllama.json",
        include_str!("fixtures/transformers_mllama.json"),
    ),
    (
        "transformers_pixtral.json",
        include_str!("fixtures/transformers_pixtral.json"),
    ),
    (
        "transformers_qwen_vl.json",
        include_str!("fixtures/transformers_qwen_vl.json"),
    ),
    (
        "transformers_qwen_vl_multi_image.json",
        include_str!("fixtures/transformers_qwen_vl_multi_image.json"),
    ),
    (
        "transformers_qwen_vl_video.json",
        include_str!("fixtures/transformers_qwen_vl_video.json"),
    ),
    (
        "transformers_sam.json",
        include_str!("fixtures/transformers_sam.json"),
    ),
    (
        "transformers_videomae.json",
        include_str!("fixtures/transformers_videomae.json"),
    ),
    (
        "transformers_videomae_batch.json",
        include_str!("fixtures/transformers_videomae_batch.json"),
    ),
    (
        "transformers_vivit.json",
        include_str!("fixtures/transformers_vivit.json"),
    ),
    (
        "transformers_vivit_batch.json",
        include_str!("fixtures/transformers_vivit_batch.json"),
    ),
];

#[derive(Debug, Deserialize)]
struct Fixture {
    schema: String,
    generator: String,
    upstream: Upstream,
    input: FixtureInput,
    cases: Vec<ParityCase>,
}

#[derive(Debug, Deserialize)]
struct FixtureInput {
    kind: String,
    description: String,
    parameters: Map<String, Value>,
}

#[derive(Debug, Deserialize)]
struct Upstream {
    library: String,
    source: String,
    version: String,
    commit: String,
}

#[derive(Debug, Deserialize)]
struct ParityCase {
    family: String,
    backend: String,
    class_name: String,
    model_id: String,
    processor_config: Map<String, Value>,
    comparison: ComparisonPolicy,
}

#[derive(Debug, Deserialize)]
struct ComparisonPolicy {
    float_full_values: FloatTolerance,
    float_statistics: FloatTolerance,
    integer_values: String,
    shape: String,
    dtype: String,
    metadata: String,
}

#[derive(Debug, Deserialize)]
struct FloatTolerance {
    absolute_tolerance: f64,
    relative_tolerance: f64,
}

#[test]
fn dedicated_transformers_fixtures_match_catalog_evidence_contract() {
    let catalog = ProcessorCatalog::new();
    let known_fixtures = FIXTURES
        .iter()
        .map(|(fixture_name, _)| *fixture_name)
        .collect::<BTreeSet<_>>();
    let mut observed_links = BTreeSet::new();
    let mut observed_names = BTreeSet::new();

    for (fixture_name, contents) in FIXTURES {
        let fixture: Fixture = serde_json::from_str(contents)
            .unwrap_or_else(|err| panic!("{fixture_name} should parse: {err}"));

        assert_eq!(fixture.schema, FIXTURE_SCHEMA, "{fixture_name}");
        assert_eq!(fixture.generator, GENERATOR, "{fixture_name}");
        assert_eq!(fixture.input.kind, "deterministic_rgb", "{fixture_name}");
        assert!(
            fixture.input.description.contains("i * 37 + 17 + n * 53"),
            "{fixture_name} should record the deterministic RGB formula"
        );
        for parameter in [
            "width",
            "height",
            "image_count",
            "width_step",
            "height_step",
        ] {
            assert!(
                fixture.input.parameters.contains_key(parameter),
                "{fixture_name} should record input parameter {parameter}"
            );
        }
        assert_eq!(fixture.upstream.library, UPSTREAM_LIBRARY, "{fixture_name}");
        assert_eq!(fixture.upstream.source, UPSTREAM_SOURCE, "{fixture_name}");
        assert_eq!(fixture.upstream.commit.len(), 40, "{fixture_name}");
        assert!(
            fixture
                .upstream
                .commit
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase()),
            "{fixture_name} should record a lowercase full Git commit"
        );
        assert!(
            !fixture.upstream.version.trim().is_empty(),
            "{fixture_name} should record the upstream Transformers version"
        );
        assert!(!fixture.cases.is_empty(), "{fixture_name}");

        for case in &fixture.cases {
            let context = format!("{fixture_name}:{}", case.family);
            assert!(!case.model_id.trim().is_empty(), "{context}");
            assert!(
                matches!(case.backend.as_str(), "pil" | "torchvision"),
                "{context}:{} has unsupported backend {}",
                case.class_name,
                case.backend
            );
            assert!(
                !case.processor_config.is_empty(),
                "{context} should record the effective processor configuration"
            );
            validate_comparison_policy(&case.comparison, &context);

            let is_auxiliary_video = *fixture_name == "transformers_qwen_vl_video.json";
            let catalog_class_name = if is_auxiliary_video {
                assert_eq!(case.class_name, "Qwen2VLVideoProcessor", "{context}");
                "Qwen2VLImageProcessor"
            } else {
                &case.class_name
            };

            let entry = catalog
                .find_by_class_name(UpstreamLibrary::Transformers, catalog_class_name)
                .unwrap_or_else(|| panic!("{context} should resolve through ProcessorCatalog"));
            assert_eq!(
                entry.compatibility(),
                CompatibilityStatus::FixtureParity,
                "{context}"
            );

            let evidence = entry
                .fixture_evidence()
                .unwrap_or_else(|| panic!("{context} should declare dedicated fixture evidence"));
            assert_eq!(
                evidence.source_commit(),
                fixture.upstream.commit,
                "{context} source commit"
            );
            assert!(
                evidence.fixture_files().contains(fixture_name),
                "{context} catalog evidence should name {fixture_name}"
            );
            observed_links.insert((entry.class_name(), *fixture_name));
            if !is_auxiliary_video {
                observed_names.insert((
                    entry.class_name().to_owned(),
                    (*fixture_name).to_owned(),
                    case.class_name.clone(),
                ));
            }
        }
    }

    for entry in catalog.entries().iter().filter(|entry| {
        entry.library() == UpstreamLibrary::Transformers && entry.fixture_evidence().is_some()
    }) {
        let evidence = entry.fixture_evidence().expect("filtered above");
        assert!(
            !evidence.fixture_files().is_empty(),
            "{}",
            entry.class_name()
        );
        for fixture_name in evidence.fixture_files() {
            assert!(
                known_fixtures.contains(fixture_name),
                "{} references unknown fixture {fixture_name}",
                entry.class_name()
            );
            assert!(
                observed_links.contains(&(entry.class_name(), *fixture_name)),
                "{} is not linked back from {fixture_name}",
                entry.class_name()
            );
        }
        for class_name in
            std::iter::once(entry.class_name()).chain(entry.class_aliases().iter().copied())
        {
            assert!(
                evidence.fixture_files().iter().any(|fixture_name| {
                    observed_names.contains(&(
                        entry.class_name().to_owned(),
                        (*fixture_name).to_owned(),
                        class_name.to_owned(),
                    ))
                }),
                "{} dedicated evidence does not cover {class_name}",
                entry.class_name()
            );
        }
    }
}

fn validate_comparison_policy(policy: &ComparisonPolicy, context: &str) {
    for (label, tolerance) in [
        ("float_full_values", &policy.float_full_values),
        ("float_statistics", &policy.float_statistics),
    ] {
        assert!(
            tolerance.absolute_tolerance.is_finite() && tolerance.absolute_tolerance >= 0.0,
            "{context}:{label} should have a finite non-negative absolute tolerance"
        );
        assert!(
            tolerance.absolute_tolerance <= MAX_ABSOLUTE_TOLERANCE,
            "{context}:{label} absolute tolerance exceeds reviewed maximum"
        );
        assert!(
            tolerance.relative_tolerance.is_finite() && tolerance.relative_tolerance >= 0.0,
            "{context}:{label} should have a finite non-negative relative tolerance"
        );
        assert!(
            tolerance.relative_tolerance <= MAX_RELATIVE_TOLERANCE,
            "{context}:{label} relative tolerance exceeds reviewed maximum"
        );
    }

    for (label, value) in [
        ("integer_values", policy.integer_values.as_str()),
        ("shape", policy.shape.as_str()),
        ("dtype", policy.dtype.as_str()),
        ("metadata", policy.metadata.as_str()),
    ] {
        assert_eq!(value, "exact", "{context}:{label}");
    }
}
