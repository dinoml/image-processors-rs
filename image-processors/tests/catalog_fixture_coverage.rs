use std::collections::BTreeSet;

use image_processors::{CatalogEntry, CompatibilityStatus, ProcessorCatalog, UpstreamLibrary};
use serde_json::Value;

const TRANSFORMERS_AGGREGATES: &[AggregateFixture] = &[
    AggregateFixture {
        name: "fixtures/transformers/catalog_encoder_restoration.json",
        contents: include_str!("fixtures/transformers/catalog_encoder_restoration.json"),
        commit_path: &["upstream", "commit"],
        case_paths: &[["cases", "class_name"]],
    },
    AggregateFixture {
        name: "fixtures/transformers/catalog_task_vision.json",
        contents: include_str!("fixtures/transformers/catalog_task_vision.json"),
        commit_path: &["upstream", "commit"],
        case_paths: &[["cases", "class_name"], ["alias_cases", "alias_class_name"]],
    },
    AggregateFixture {
        name: "fixtures/transformers/catalog_multimodal.json",
        contents: include_str!("fixtures/transformers/catalog_multimodal.json"),
        commit_path: &["upstream", "commit"],
        case_paths: &[["cases", "class_name"], ["aliases", "alias"]],
    },
];

const DEDICATED_TRANSFORMERS_FIXTURES: &[(&str, &str)] = &[
    (
        "fixtures/transformers_clip_vit.json",
        include_str!("fixtures/transformers_clip_vit.json"),
    ),
    (
        "fixtures/transformers_detr.json",
        include_str!("fixtures/transformers_detr.json"),
    ),
    (
        "fixtures/transformers_donut.json",
        include_str!("fixtures/transformers_donut.json"),
    ),
    (
        "fixtures/transformers_gemma3.json",
        include_str!("fixtures/transformers_gemma3.json"),
    ),
    (
        "fixtures/transformers_idefics3.json",
        include_str!("fixtures/transformers_idefics3.json"),
    ),
    (
        "fixtures/transformers_llava_next.json",
        include_str!("fixtures/transformers_llava_next.json"),
    ),
    (
        "fixtures/transformers_llava_next_nhwc.json",
        include_str!("fixtures/transformers_llava_next_nhwc.json"),
    ),
    (
        "fixtures/transformers_mllama.json",
        include_str!("fixtures/transformers_mllama.json"),
    ),
    (
        "fixtures/transformers_pixtral.json",
        include_str!("fixtures/transformers_pixtral.json"),
    ),
    (
        "fixtures/transformers_qwen_vl.json",
        include_str!("fixtures/transformers_qwen_vl.json"),
    ),
    (
        "fixtures/transformers_qwen_vl_multi_image.json",
        include_str!("fixtures/transformers_qwen_vl_multi_image.json"),
    ),
    (
        "fixtures/transformers_qwen_vl_video.json",
        include_str!("fixtures/transformers_qwen_vl_video.json"),
    ),
    (
        "fixtures/transformers_sam.json",
        include_str!("fixtures/transformers_sam.json"),
    ),
    (
        "fixtures/transformers_videomae.json",
        include_str!("fixtures/transformers_videomae.json"),
    ),
    (
        "fixtures/transformers_videomae_batch.json",
        include_str!("fixtures/transformers_videomae_batch.json"),
    ),
    (
        "fixtures/transformers_vivit.json",
        include_str!("fixtures/transformers_vivit.json"),
    ),
    (
        "fixtures/transformers_vivit_batch.json",
        include_str!("fixtures/transformers_vivit_batch.json"),
    ),
];

const DIFFUSERS_FIXTURE: (&str, &str) = (
    "fixtures/diffusers/image_processors.json",
    include_str!("fixtures/diffusers/image_processors.json"),
);

struct AggregateFixture {
    name: &'static str,
    contents: &'static str,
    commit_path: &'static [&'static str],
    case_paths: &'static [[&'static str; 2]],
}

struct Evidence {
    library: UpstreamLibrary,
    commit: String,
    file: &'static str,
    class_names: BTreeSet<String>,
}

#[test]
fn every_fixture_parity_catalog_entry_and_alias_has_source_controlled_evidence() {
    let evidence = load_evidence();
    let catalog = ProcessorCatalog::new();

    for entry in catalog
        .entries()
        .iter()
        .filter(|entry| entry.compatibility() == CompatibilityStatus::FixtureParity)
    {
        assert_catalog_name_is_covered(entry, entry.class_name(), &evidence);
        for alias in entry.class_aliases() {
            assert_catalog_name_is_covered(entry, alias, &evidence);
        }
    }

    for source in &evidence {
        for class_name in &source.class_names {
            let catalog_class_name = if class_name == "Qwen2VLVideoProcessor" {
                assert_eq!(source.file, "fixtures/transformers_qwen_vl_video.json");
                "Qwen2VLImageProcessor"
            } else {
                class_name
            };
            let entry = catalog
                .find_by_class_name(source.library, catalog_class_name)
                .unwrap_or_else(|| {
                    panic!(
                        "{} records unknown {} class {class_name}",
                        source.file,
                        source.library.as_str()
                    )
                });
            assert_eq!(
                entry.audit_commit(),
                source.commit,
                "{}:{class_name} revision",
                source.file
            );
            if class_name == "Qwen2VLVideoProcessor" {
                let evidence = entry
                    .fixture_evidence()
                    .expect("Qwen2-VL should declare dedicated fixture evidence");
                assert!(
                    evidence
                        .fixture_files()
                        .contains(&"transformers_qwen_vl_video.json"),
                    "Qwen2-VL video fixture should be linked to the catalog family"
                );
            }
        }
    }
}

fn assert_catalog_name_is_covered(entry: &CatalogEntry, class_name: &str, evidence: &[Evidence]) {
    let is_covered = evidence.iter().any(|source| {
        source.library == entry.library()
            && source.commit == entry.audit_commit()
            && source.class_names.contains(class_name)
    });

    assert!(
        is_covered,
        "{} {} has no fixture evidence at audit commit {}",
        entry.library().as_str(),
        class_name,
        entry.audit_commit()
    );
}

fn load_evidence() -> Vec<Evidence> {
    let mut evidence = TRANSFORMERS_AGGREGATES
        .iter()
        .map(|fixture| {
            let json = parse_fixture(fixture.name, fixture.contents);
            Evidence {
                library: UpstreamLibrary::Transformers,
                commit: string_at(&json, fixture.commit_path, fixture.name).to_owned(),
                file: fixture.name,
                class_names: fixture
                    .case_paths
                    .iter()
                    .flat_map(|path| names_at(&json, path, fixture.name))
                    .collect(),
            }
        })
        .collect::<Vec<_>>();

    evidence.extend(
        DEDICATED_TRANSFORMERS_FIXTURES
            .iter()
            .map(|(fixture_name, contents)| {
                let json = parse_fixture(fixture_name, contents);
                Evidence {
                    library: UpstreamLibrary::Transformers,
                    commit: string_at(&json, &["upstream", "commit"], fixture_name).to_owned(),
                    file: fixture_name,
                    class_names: names_at(&json, &["cases", "class_name"], fixture_name)
                        .into_iter()
                        .collect(),
                }
            }),
    );

    let json = parse_fixture(DIFFUSERS_FIXTURE.0, DIFFUSERS_FIXTURE.1);
    evidence.push(Evidence {
        library: UpstreamLibrary::Diffusers,
        commit: string_at(&json, &["upstream", "commit"], DIFFUSERS_FIXTURE.0).to_owned(),
        file: DIFFUSERS_FIXTURE.0,
        class_names: names_at(&json, &["cases", "class_name"], DIFFUSERS_FIXTURE.0)
            .into_iter()
            .collect(),
    });

    for source in &evidence {
        assert_eq!(source.commit.len(), 40, "{}", source.file);
        assert!(!source.class_names.is_empty(), "{}", source.file);
    }
    evidence
}

fn parse_fixture(name: &str, contents: &str) -> Value {
    serde_json::from_str(contents).unwrap_or_else(|error| panic!("{name} should parse: {error}"))
}

fn names_at(json: &Value, path: &[&str; 2], fixture_name: &str) -> Vec<String> {
    json[path[0]]
        .as_array()
        .unwrap_or_else(|| panic!("{fixture_name}:{} should be an array", path[0]))
        .iter()
        .map(|case| {
            case[path[1]]
                .as_str()
                .unwrap_or_else(|| panic!("{fixture_name}:{}.{} should be text", path[0], path[1]))
                .to_owned()
        })
        .collect()
}

fn string_at<'a>(json: &'a Value, path: &[&str], fixture_name: &str) -> &'a str {
    let value = path.iter().fold(json, |value, segment| &value[*segment]);
    value
        .as_str()
        .unwrap_or_else(|| panic!("{fixture_name}:{} should be text", path.join(".")))
}
