//! Immutable upstream processor compatibility catalog.
//!
//! The catalog maps reviewed Diffusers and Transformers processor identities to
//! Rust-native recipes and family wrappers. It is source-controlled static data:
//! lookups never download model files, query the Hub, or require Python.

#[path = "catalog_data.rs"]
mod catalog_data;

/// Upstream library that defines a processor identity.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum UpstreamLibrary {
    /// Hugging Face Transformers.
    Transformers,
    /// Hugging Face Diffusers.
    Diffusers,
}

impl UpstreamLibrary {
    /// Returns the stable lowercase catalog name for this library.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Transformers => "transformers",
            Self::Diffusers => "diffusers",
        }
    }
}

/// Rust family wrapper, preset collection, or utility for a catalog entry.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum ProcessorFamilyKind {
    /// Diffusers VAE image processor family.
    Vae,
    /// Diffusers LDM3D RGB-plus-depth VAE image processor family.
    VaeLdm3d,
    /// Attention mask downsampling utility family.
    AttentionMask,
    /// Aspect-ratio bucket selection and resize/crop planning utility family.
    AspectRatioBucket,
    /// Pipeline-local image conditioning processor bucket.
    ImageConditioning,
    /// Pipeline-local video conditioning processor bucket.
    VideoConditioning,
    /// Dense depth, normal, intrinsic-image, and uncertainty prediction bucket.
    DensePrediction,
    /// Encoder and classifier preset family.
    EncoderClassifier,
    /// Detection and grounding preset family.
    DetectionGrounding,
    /// Segmentation, mask, and matting preset family.
    Segmentation,
    /// Depth and geometry preset family.
    DepthGeometry,
    /// Vision-language and multimodal input preset family.
    VisionLanguage,
    /// Video processor preset family.
    Video,
    /// OCR, document, and table preset family.
    DocumentUnderstanding,
    /// Keypoint, matching, and pose preset family.
    KeypointMatchingPose,
    /// Image restoration preset family.
    ImageRestoration,
    /// CLIP encoder image processor family.
    Clip,
    /// ViT-style classifier image processor family.
    Vit,
    /// VideoMAE video image processor family.
    VideoMae,
    /// ViViT video image processor family.
    Vivit,
    /// Qwen/VLM smart-resize and patch-flatten image processor family.
    QwenVl,
    /// LLaVA-NeXT AnyRes image processor family.
    LlavaNext,
    /// Pixtral patch-aligned image processor family.
    Pixtral,
    /// Idefics3 split-image processor family.
    Idefics3,
    /// Gemma3 pan-and-scan image processor family.
    Gemma3,
    /// Mllama tiled-image processor family.
    Mllama,
    /// DETR-style detection image processor family.
    Detr,
    /// SAM segmentation image processor family.
    Sam,
    /// Document and OCR image processor family.
    DocumentOcr,
}

impl ProcessorFamilyKind {
    /// Returns the stable lowercase catalog name for this family.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Vae => "vae",
            Self::VaeLdm3d => "vae_ldm3d",
            Self::AttentionMask => "attention_mask",
            Self::AspectRatioBucket => "aspect_ratio_bucket",
            Self::ImageConditioning => "image_conditioning",
            Self::VideoConditioning => "video_conditioning",
            Self::DensePrediction => "dense_prediction",
            Self::EncoderClassifier => "encoder_classifier",
            Self::DetectionGrounding => "detection_grounding",
            Self::Segmentation => "segmentation",
            Self::DepthGeometry => "depth_geometry",
            Self::VisionLanguage => "vision_language",
            Self::Video => "video",
            Self::DocumentUnderstanding => "document_understanding",
            Self::KeypointMatchingPose => "keypoint_matching_pose",
            Self::ImageRestoration => "image_restoration",
            Self::Clip => "clip",
            Self::Vit => "vit",
            Self::VideoMae => "videomae",
            Self::Vivit => "vivit",
            Self::QwenVl => "qwen_vl",
            Self::LlavaNext => "llava_next",
            Self::Pixtral => "pixtral",
            Self::Idefics3 => "idefics3",
            Self::Gemma3 => "gemma3",
            Self::Mllama => "mllama",
            Self::Detr => "detr",
            Self::Sam => "sam",
            Self::DocumentOcr => "document_ocr",
        }
    }
}

/// Audited compatibility status for a catalog entry.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum CompatibilityStatus {
    /// Behavior is covered by full-value source-controlled upstream parity fixtures.
    FixtureParity,
    /// Behavior is covered by summary, sample, statistics, or geometry fixtures.
    SummaryFixture,
    /// Rust behavior is implemented, but full upstream value parity is pending.
    Implemented,
    /// Upstream behavior has been audited and classified, but implementation is pending.
    Planned,
}

impl CompatibilityStatus {
    /// Returns the stable lowercase catalog name for this status.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::FixtureParity => "fixture_parity",
            Self::SummaryFixture => "summary_fixture",
            Self::Implemented => "implemented",
            Self::Planned => "planned",
        }
    }
}

/// A single audited upstream processor mapping.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CatalogEntry {
    library: UpstreamLibrary,
    class_name: &'static str,
    class_aliases: &'static [&'static str],
    model_type: Option<&'static str>,
    processor_ids: &'static [&'static str],
    recipe_id: Option<&'static str>,
    family: ProcessorFamilyKind,
    audit_commit: &'static str,
    compatibility: CompatibilityStatus,
}

/// Source revision and files for dedicated full-value parity evidence.
///
/// For [`CompatibilityStatus::FixtureParity`], the evidence revision must
/// match [`CatalogEntry::audit_commit`] so the catalog identity and numeric
/// payload were audited against the same upstream source.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FixtureEvidence {
    source_commit: &'static str,
    fixture_files: &'static [&'static str],
}

impl FixtureEvidence {
    /// Returns the exact upstream source commit used to generate the fixtures.
    pub const fn source_commit(self) -> &'static str {
        self.source_commit
    }

    /// Returns the dedicated fixture file names covering the catalog entry.
    pub const fn fixture_files(self) -> &'static [&'static str] {
        self.fixture_files
    }
}

impl CatalogEntry {
    /// Returns the upstream library for this entry.
    pub const fn library(&self) -> UpstreamLibrary {
        self.library
    }

    /// Returns the canonical upstream image processor class name.
    pub const fn class_name(&self) -> &'static str {
        self.class_name
    }

    /// Returns alternate upstream class names that resolve to this entry.
    pub const fn class_aliases(&self) -> &'static [&'static str] {
        self.class_aliases
    }

    /// Returns the upstream model type, when the processor has one.
    pub const fn model_type(&self) -> Option<&'static str> {
        self.model_type
    }

    /// Returns known pretrained processor ids associated with this entry.
    pub const fn processor_ids(&self) -> &'static [&'static str] {
        self.processor_ids
    }

    /// Returns the stable Rust recipe id for this entry, when one exists.
    pub const fn recipe_id(&self) -> Option<&'static str> {
        self.recipe_id
    }

    /// Returns the Rust family wrapper for this entry.
    pub const fn family(&self) -> ProcessorFamilyKind {
        self.family
    }

    /// Returns the upstream source commit used for the compatibility audit.
    pub const fn audit_commit(&self) -> &'static str {
        self.audit_commit
    }

    /// Returns the audited compatibility status for this entry.
    pub const fn compatibility(&self) -> CompatibilityStatus {
        self.compatibility
    }

    /// Returns dedicated full-value parity evidence for this entry, when used.
    ///
    /// Catalog entries covered by aggregated catalog fixtures return `None`.
    pub fn fixture_evidence(&self) -> Option<FixtureEvidence> {
        catalog_data::fixture_evidence(self.class_name)
    }
}

/// Typed catalog query used by [`ProcessorCatalog::find`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CatalogQuery<'a> {
    library: Option<UpstreamLibrary>,
    class_name: Option<&'a str>,
    model_type: Option<&'a str>,
    processor_id: Option<&'a str>,
}

impl<'a> CatalogQuery<'a> {
    /// Creates an empty query.
    ///
    /// Empty queries intentionally do not match an arbitrary catalog entry.
    pub const fn new() -> Self {
        Self {
            library: None,
            class_name: None,
            model_type: None,
            processor_id: None,
        }
    }

    /// Creates a query scoped to one upstream library.
    pub const fn for_library(library: UpstreamLibrary) -> Self {
        Self {
            library: Some(library),
            class_name: None,
            model_type: None,
            processor_id: None,
        }
    }

    /// Adds an upstream library filter.
    pub const fn with_library(mut self, library: UpstreamLibrary) -> Self {
        self.library = Some(library);
        self
    }

    /// Adds an upstream class-name selector.
    pub const fn with_class_name(mut self, class_name: &'a str) -> Self {
        self.class_name = Some(class_name);
        self
    }

    /// Adds an upstream model-type selector.
    pub const fn with_model_type(mut self, model_type: &'a str) -> Self {
        self.model_type = Some(model_type);
        self
    }

    /// Adds a pretrained processor-id selector.
    pub const fn with_processor_id(mut self, processor_id: &'a str) -> Self {
        self.processor_id = Some(processor_id);
        self
    }

    fn has_selector(self) -> bool {
        self.class_name.is_some() || self.model_type.is_some() || self.processor_id.is_some()
    }
}

/// Static compatibility catalog for upstream processor lookup.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ProcessorCatalog;

impl ProcessorCatalog {
    /// Creates a catalog backed by the source-controlled static tables.
    pub const fn new() -> Self {
        Self
    }

    /// Returns all audited catalog entries.
    pub fn entries(&self) -> &'static [CatalogEntry] {
        catalog_data::CATALOG_ENTRIES
    }

    /// Finds the first entry matching every selector in `query`.
    ///
    /// An empty query returns `None` instead of selecting an arbitrary entry.
    pub fn find(&self, query: CatalogQuery<'_>) -> Option<&'static CatalogEntry> {
        if !query.has_selector() {
            return None;
        }

        self.entries()
            .iter()
            .find(|entry| entry_matches_query(entry, query))
    }

    /// Finds an entry by upstream library and processor class name.
    pub fn find_by_class_name(
        &self,
        library: UpstreamLibrary,
        class_name: &str,
    ) -> Option<&'static CatalogEntry> {
        self.find(CatalogQuery::for_library(library).with_class_name(class_name))
    }

    /// Finds an entry by upstream library and model type.
    pub fn find_by_model_type(
        &self,
        library: UpstreamLibrary,
        model_type: &str,
    ) -> Option<&'static CatalogEntry> {
        self.find(CatalogQuery::for_library(library).with_model_type(model_type))
    }

    /// Finds an entry by known pretrained processor id.
    pub fn find_by_processor_id(&self, processor_id: &str) -> Option<&'static CatalogEntry> {
        self.find(CatalogQuery::new().with_processor_id(processor_id))
    }
}

fn entry_matches_query(entry: &CatalogEntry, query: CatalogQuery<'_>) -> bool {
    query.library.is_none_or(|library| entry.library == library)
        && query.class_name.is_none_or(|class_name| {
            entry.class_name == class_name || entry.class_aliases.contains(&class_name)
        })
        && query
            .model_type
            .is_none_or(|model_type| entry.model_type == Some(model_type))
        && query
            .processor_id
            .is_none_or(|processor_id| entry.processor_ids.contains(&processor_id))
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeSet, HashSet};

    use super::*;
    use crate::processors::{
        ClipImageProcessorConfig, DetrImageProcessorConfig, DocumentOcrImageProcessorConfig,
        EncoderImageProcessorConfig, EncoderImageProcessorPreset, Gemma3ImageProcessorConfig,
        Idefics3ImageProcessorConfig, LlavaNextImageProcessorConfig, MllamaImageProcessorConfig,
        PixtralImageProcessorConfig, QwenVlImageProcessorConfig, SamImageProcessorConfig,
        Swin2SrImageProcessorConfig, TaskVisionImageProcessorConfig, TaskVisionProcessorPreset,
        VaeImageProcessorConfig, VideoMaeImageProcessorConfig, VitImageProcessorConfig,
        VivitImageProcessorConfig, MULTIMODAL_PRESETS,
    };
    use crate::recipe::{ProcessorRecipe, ProcessorRecipePostprocess};

    #[test]
    fn lookup_by_class_resolves_clip_recipe() {
        let catalog = ProcessorCatalog::new();

        let entry = catalog
            .find_by_class_name(UpstreamLibrary::Transformers, "CLIPImageProcessor")
            .expect("CLIP catalog entry should be present");

        assert_eq!(entry.recipe_id(), Some("transformers.clip_image_processor"));
        assert_eq!(entry.family(), ProcessorFamilyKind::Clip);
        assert_eq!(entry.compatibility(), CompatibilityStatus::FixtureParity);
    }

    #[test]
    fn lookup_by_class_alias_resolves_canonical_entry() {
        let catalog = ProcessorCatalog::new();

        let entry = catalog
            .find_by_class_name(UpstreamLibrary::Transformers, "CLIPImageProcessorPil")
            .expect("CLIP PIL alias should resolve");

        assert_eq!(entry.class_name(), "CLIPImageProcessor");
        assert_eq!(entry.recipe_id(), Some("transformers.clip_image_processor"));
        assert!(entry.class_aliases().contains(&"CLIPImageProcessorPil"));
    }

    #[test]
    fn lookup_by_model_type_resolves_vit_recipe() {
        let catalog = ProcessorCatalog::new();

        let entry = catalog
            .find_by_model_type(UpstreamLibrary::Transformers, "vit")
            .expect("ViT model-type catalog entry should be present");

        assert_eq!(entry.class_name(), "ViTImageProcessor");
        assert_eq!(entry.recipe_id(), Some("transformers.vit_image_processor"));
    }

    #[test]
    fn lookup_by_processor_id_is_static_and_network_free() {
        let catalog = ProcessorCatalog::new();

        let entry = catalog
            .find_by_processor_id("openai/clip-vit-base-patch32")
            .expect("known CLIP processor id should resolve");

        assert_eq!(entry.library(), UpstreamLibrary::Transformers);
        assert!(entry
            .processor_ids()
            .contains(&"openai/clip-vit-base-patch32"));
    }

    #[test]
    fn query_filters_by_library_and_requires_selector() {
        let catalog = ProcessorCatalog::new();

        assert!(catalog
            .find_by_class_name(UpstreamLibrary::Transformers, "VaeImageProcessor")
            .is_none());
        assert!(catalog
            .find_by_class_name(UpstreamLibrary::Diffusers, "VaeImageProcessor")
            .is_some());
        assert!(catalog.find(CatalogQuery::new()).is_none());
    }

    #[test]
    fn entries_track_audit_commit_and_status() {
        let catalog = ProcessorCatalog::new();

        for entry in catalog.entries() {
            assert!(!entry.audit_commit().is_empty());
            assert_ne!(entry.compatibility().as_str(), "");
            for alias in entry.class_aliases() {
                assert_ne!(*alias, "");
                assert_ne!(*alias, entry.class_name());
            }
        }
    }

    #[test]
    fn every_catalog_entry_has_full_fixture_parity() {
        let catalog = ProcessorCatalog::new();
        let incomplete = catalog
            .entries()
            .iter()
            .filter(|entry| entry.compatibility() != CompatibilityStatus::FixtureParity)
            .map(CatalogEntry::class_name)
            .collect::<Vec<_>>();

        assert!(
            incomplete.is_empty(),
            "catalog entries without full fixture parity: {incomplete:?}"
        );
    }

    #[test]
    fn catalog_recipe_ids_match_family_configs() {
        let catalog = ProcessorCatalog::new();
        let expected = [
            (
                ProcessorFamilyKind::Vae,
                VaeImageProcessorConfig::default()
                    .processor_recipe()
                    .expect("VAE recipe should be valid"),
            ),
            (
                ProcessorFamilyKind::VaeLdm3d,
                VaeImageProcessorConfig::default()
                    .processor_recipe()
                    .expect("LDM3D VAE recipe should be valid"),
            ),
            (
                ProcessorFamilyKind::Clip,
                ClipImageProcessorConfig::default()
                    .processor_recipe()
                    .expect("CLIP recipe should be valid"),
            ),
            (
                ProcessorFamilyKind::Vit,
                VitImageProcessorConfig::default()
                    .processor_recipe()
                    .expect("ViT recipe should be valid"),
            ),
            (
                ProcessorFamilyKind::VideoMae,
                VideoMaeImageProcessorConfig::default()
                    .processor_recipe()
                    .expect("VideoMAE recipe should be valid"),
            ),
            (
                ProcessorFamilyKind::Vivit,
                VivitImageProcessorConfig::default()
                    .processor_recipe()
                    .expect("ViViT recipe should be valid"),
            ),
            (
                ProcessorFamilyKind::QwenVl,
                QwenVlImageProcessorConfig::default()
                    .processor_recipe()
                    .expect("Qwen/VLM recipe should be valid"),
            ),
            (
                ProcessorFamilyKind::LlavaNext,
                LlavaNextImageProcessorConfig::default()
                    .processor_recipe()
                    .expect("LLaVA-NeXT recipe should be valid"),
            ),
            (
                ProcessorFamilyKind::Pixtral,
                PixtralImageProcessorConfig::default()
                    .processor_recipe()
                    .expect("Pixtral recipe should be valid"),
            ),
            (
                ProcessorFamilyKind::Idefics3,
                Idefics3ImageProcessorConfig::default()
                    .processor_recipe()
                    .expect("Idefics3 recipe should be valid"),
            ),
            (
                ProcessorFamilyKind::Gemma3,
                Gemma3ImageProcessorConfig::default()
                    .processor_recipe()
                    .expect("Gemma3 recipe should be valid"),
            ),
            (
                ProcessorFamilyKind::Mllama,
                MllamaImageProcessorConfig::default()
                    .processor_recipe()
                    .expect("Mllama recipe should be valid"),
            ),
            (
                ProcessorFamilyKind::Detr,
                DetrImageProcessorConfig::default()
                    .processor_recipe()
                    .expect("DETR recipe should be valid"),
            ),
            (
                ProcessorFamilyKind::Sam,
                SamImageProcessorConfig::default()
                    .processor_recipe()
                    .expect("SAM recipe should be valid"),
            ),
            (
                ProcessorFamilyKind::DocumentOcr,
                DocumentOcrImageProcessorConfig::default()
                    .processor_recipe()
                    .expect("document/OCR recipe should be valid"),
            ),
        ];

        for (family, recipe) in expected {
            for entry in catalog
                .entries()
                .iter()
                .filter(|entry| entry.family() == family)
            {
                assert_eq!(entry.recipe_id(), Some(recipe.id()));
            }
        }
    }

    #[test]
    fn utility_entries_can_be_cataloged_without_recipe() {
        let catalog = ProcessorCatalog::new();

        let entry = catalog
            .find_by_class_name(UpstreamLibrary::Diffusers, "IPAdapterMaskProcessor")
            .expect("IP-Adapter mask catalog entry should be present");

        assert_eq!(entry.recipe_id(), None);
        assert_eq!(entry.family(), ProcessorFamilyKind::AttentionMask);
        assert_eq!(entry.compatibility(), CompatibilityStatus::FixtureParity);

        let entry = catalog
            .find_by_class_name(UpstreamLibrary::Diffusers, "PixArtImageProcessor")
            .expect("PixArt catalog entry should be present");

        assert_eq!(entry.recipe_id(), None);
        assert_eq!(entry.family(), ProcessorFamilyKind::AspectRatioBucket);
        assert_eq!(entry.compatibility(), CompatibilityStatus::FixtureParity);
    }

    #[test]
    fn diffusers_entries_have_full_fixture_parity() {
        let catalog = ProcessorCatalog::new();

        for class_name in [
            "VaeImageProcessor",
            "VaeImageProcessorLDM3D",
            "BlipImageProcessor",
            "IPAdapterMaskProcessor",
            "Flux2ImageProcessor",
            "HunyuanVideo15ImageProcessor",
            "JoyImageEditImageProcessor",
            "LTX2VideoHDRProcessor",
            "MarigoldImageProcessor",
            "PixArtImageProcessor",
            "VisualClozeProcessor",
            "WanAnimateImageProcessor",
        ] {
            let entry = catalog
                .find_by_class_name(UpstreamLibrary::Diffusers, class_name)
                .unwrap_or_else(|| panic!("missing Diffusers catalog entry for {class_name}"));

            assert_eq!(entry.compatibility(), CompatibilityStatus::FixtureParity);
        }
    }

    #[test]
    fn diffusers_pipeline_local_processors_track_recipe_statuses() {
        let catalog = ProcessorCatalog::new();
        let expected = [
            (
                "BlipImageProcessor",
                ProcessorFamilyKind::ImageConditioning,
                CompatibilityStatus::FixtureParity,
                None,
            ),
            (
                "Flux2ImageProcessor",
                ProcessorFamilyKind::ImageConditioning,
                CompatibilityStatus::FixtureParity,
                None,
            ),
            (
                "JoyImageEditImageProcessor",
                ProcessorFamilyKind::ImageConditioning,
                CompatibilityStatus::FixtureParity,
                None,
            ),
            (
                "VisualClozeProcessor",
                ProcessorFamilyKind::ImageConditioning,
                CompatibilityStatus::FixtureParity,
                None,
            ),
            (
                "WanAnimateImageProcessor",
                ProcessorFamilyKind::ImageConditioning,
                CompatibilityStatus::FixtureParity,
                None,
            ),
            (
                "HunyuanVideo15ImageProcessor",
                ProcessorFamilyKind::VideoConditioning,
                CompatibilityStatus::FixtureParity,
                None,
            ),
            (
                "LTX2VideoHDRProcessor",
                ProcessorFamilyKind::VideoConditioning,
                CompatibilityStatus::FixtureParity,
                Some("diffusers.ltx2_video_hdr_postprocess"),
            ),
            (
                "MarigoldImageProcessor",
                ProcessorFamilyKind::DensePrediction,
                CompatibilityStatus::FixtureParity,
                None,
            ),
        ];

        for (class_name, family, compatibility, recipe_id) in expected {
            let entry = catalog
                .find_by_class_name(UpstreamLibrary::Diffusers, class_name)
                .unwrap_or_else(|| {
                    panic!("missing pipeline-local Diffusers catalog entry for {class_name}")
                });

            assert_eq!(entry.recipe_id(), recipe_id);
            assert_eq!(entry.family(), family);
            assert_eq!(entry.compatibility(), compatibility);
        }
    }

    #[test]
    fn ltx2_catalog_entry_points_at_common_logc3_postprocess_recipe() {
        let catalog = ProcessorCatalog::new();
        let entry = catalog
            .find_by_class_name(UpstreamLibrary::Diffusers, "LTX2VideoHDRProcessor")
            .expect("LTX2 catalog entry should be present");
        let recipe = ProcessorRecipe::logc3_hdr_video_postprocess(
            "diffusers.ltx2_video_hdr_postprocess",
            "decoded_video",
        )
        .expect("LogC3 HDR postprocess recipe should be valid");

        assert_eq!(entry.recipe_id(), Some(recipe.id()));
        assert!(recipe.is_postprocess_only());
        assert!(
            matches!(
                recipe.postprocess(),
                [ProcessorRecipePostprocess::VideoTensor(_)]
            ),
            "expected a generic video tensor descriptor, got {:?}",
            recipe.postprocess()
        );
        assert_eq!(entry.compatibility(), CompatibilityStatus::FixtureParity);
    }

    #[test]
    fn transformers_fixture_presets_match_catalog_recipes() {
        let catalog = ProcessorCatalog::new();

        for preset in EncoderImageProcessorPreset::ALL {
            let entry = catalog
                .find_by_class_name(UpstreamLibrary::Transformers, preset.class_name())
                .unwrap_or_else(|| panic!("missing encoder preset {}", preset.class_name()));
            let recipe = EncoderImageProcessorConfig::for_preset(preset)
                .processor_recipe()
                .unwrap_or_else(|error| panic!("{} recipe failed: {error}", preset.class_name()));
            assert_eq!(entry.recipe_id(), Some(recipe.id()));
            assert_eq!(entry.compatibility(), CompatibilityStatus::FixtureParity);
        }

        let swin = catalog
            .find_by_class_name(UpstreamLibrary::Transformers, "Swin2SRImageProcessor")
            .expect("missing Swin2SR preset");
        assert_eq!(
            swin.recipe_id(),
            Some(Swin2SrImageProcessorConfig::RECIPE_ID)
        );

        for preset in TaskVisionProcessorPreset::ALL {
            let entry = catalog
                .find_by_class_name(UpstreamLibrary::Transformers, preset.class_name())
                .unwrap_or_else(|| panic!("missing task-vision preset {}", preset.class_name()));
            let recipe = TaskVisionImageProcessorConfig::for_preset(preset)
                .processor_recipe()
                .unwrap_or_else(|error| panic!("{} recipe failed: {error}", preset.class_name()));
            assert_eq!(entry.recipe_id(), Some(recipe.id()));
        }

        for preset in MULTIMODAL_PRESETS {
            let entry = catalog
                .find_by_class_name(UpstreamLibrary::Transformers, preset.class_name())
                .unwrap_or_else(|| panic!("missing multimodal preset {}", preset.class_name()));
            let recipe = preset
                .processor_recipe()
                .unwrap_or_else(|error| panic!("{} recipe failed: {error}", preset.class_name()));
            let processor = recipe
                .build()
                .unwrap_or_else(|error| panic!("{} recipe failed to build: {error}", recipe.id()));
            assert_eq!(entry.recipe_id(), Some(recipe.id()));
            assert_eq!(processor.preset(), preset);
            assert_eq!(entry.compatibility(), CompatibilityStatus::FixtureParity);
        }
    }

    #[test]
    fn former_planned_catalog_entries_are_exhaustively_backed_by_presets() {
        let mut preset_classes = BTreeSet::new();
        preset_classes.extend(
            EncoderImageProcessorPreset::ALL
                .into_iter()
                .map(EncoderImageProcessorPreset::class_name),
        );
        preset_classes.insert("Swin2SRImageProcessor");
        preset_classes.extend(
            TaskVisionProcessorPreset::ALL
                .into_iter()
                .map(TaskVisionProcessorPreset::class_name),
        );
        preset_classes.extend(
            MULTIMODAL_PRESETS
                .iter()
                .copied()
                .map(|preset| preset.class_name()),
        );

        let catalog_classes = ProcessorCatalog::new()
            .entries()
            .iter()
            .filter(|entry| {
                matches!(
                    entry.family(),
                    ProcessorFamilyKind::EncoderClassifier
                        | ProcessorFamilyKind::DetectionGrounding
                        | ProcessorFamilyKind::Segmentation
                        | ProcessorFamilyKind::DepthGeometry
                        | ProcessorFamilyKind::VisionLanguage
                        | ProcessorFamilyKind::Video
                        | ProcessorFamilyKind::DocumentUnderstanding
                        | ProcessorFamilyKind::KeypointMatchingPose
                        | ProcessorFamilyKind::ImageRestoration
                )
            })
            .map(CatalogEntry::class_name)
            .collect::<BTreeSet<_>>();

        assert_eq!(preset_classes.len(), 106);
        assert_eq!(preset_classes, catalog_classes);
    }

    #[test]
    fn transformers_class_aliases_cover_current_pil_variants() {
        let catalog = ProcessorCatalog::new();
        let expected = [
            ("BeitImageProcessorPil", "BeitImageProcessor"),
            (
                "Ernie4_5_VL_MoeImageProcessor",
                "Ernie4_5_VLMoeImageProcessor",
            ),
            (
                "Ernie4_5_VL_MoeImageProcessorPil",
                "Ernie4_5_VLMoeImageProcessor",
            ),
            ("Qwen2VLImageProcessorPil", "Qwen2VLImageProcessor"),
        ];

        for (alias, canonical) in expected {
            let entry = catalog
                .find_by_class_name(UpstreamLibrary::Transformers, alias)
                .unwrap_or_else(|| panic!("missing Transformers alias lookup for {alias}"));

            assert_eq!(entry.class_name(), canonical);
        }
    }

    #[test]
    fn transformers_catalog_matches_audited_class_and_entry_counts() {
        let catalog = ProcessorCatalog::new();
        let mut class_names = HashSet::new();
        let mut entry_count = 0;

        for entry in catalog
            .entries()
            .iter()
            .filter(|entry| entry.library() == UpstreamLibrary::Transformers)
        {
            entry_count += 1;
            assert!(class_names.insert(entry.class_name()));
            for alias in entry.class_aliases() {
                assert!(class_names.insert(*alias));
            }
        }

        assert_eq!(entry_count, 119);
        assert_eq!(class_names.len(), 209);
    }

    #[test]
    fn catalog_keys_are_unique() {
        let catalog = ProcessorCatalog::new();
        let mut class_keys = HashSet::new();
        let mut model_type_keys = HashSet::new();
        let mut processor_id_keys = HashSet::new();

        for entry in catalog.entries() {
            assert!(class_keys.insert((entry.library(), entry.class_name())));
            for alias in entry.class_aliases() {
                assert!(class_keys.insert((entry.library(), *alias)));
            }

            if let Some(model_type) = entry.model_type() {
                assert!(model_type_keys.insert((entry.library(), model_type)));
            }

            for processor_id in entry.processor_ids() {
                assert!(processor_id_keys.insert(*processor_id));
            }
        }
    }
}
