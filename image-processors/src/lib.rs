#![doc = include_str!("../../README.md")]
#![doc = include_str!("../../docs/USAGE.md")]
#![warn(missing_docs)]
//! Core image-processing primitives.
//!
//! This crate owns media loading, decoded frame types, deterministic numeric
//! transforms, image resizing, layout conversion, and shape computations that
//! are shared across processor families.

/// Image processor configuration and preprocessing entrypoints.
pub mod image;

/// Immutable upstream processor compatibility catalog.
pub mod catalog;

/// Media source loading, decoded frames, and frame sampling.
pub mod media;

/// Typed processor outputs for model-family preprocessing.
pub mod output;

/// Task-specific output restoration helpers.
pub mod postprocess;

/// Model-family image processor configurations and wrappers.
pub mod processors;

/// Reusable processor recipe descriptions.
pub mod recipe;

/// Tensor storage, shape, dtype, and layout conversion primitives.
pub mod tensor;

/// Resize, crop, fill, pixel-format conversion, and geometry helpers.
pub mod transforms;

#[doc(inline)]
pub use catalog::{
    CatalogEntry, CatalogQuery, CompatibilityStatus, FixtureEvidence, ProcessorCatalog,
    ProcessorFamilyKind, UpstreamLibrary,
};
#[doc(inline)]
pub use image::{
    default_batch_execution, BatchExecution, ImageProcessor, ImageProcessorConfig,
    ImageProcessorError, ImageProcessorOptions, ImageProcessorWorkspace,
    DEFAULT_PARALLEL_BATCH_THRESHOLD,
};
#[doc(inline)]
pub use media::{
    decode_image_bytes, decode_image_bytes_with_backend, decode_image_sequence_bytes,
    decode_image_sequence_bytes_with_backend, load_image_from_path,
    load_image_from_path_with_backend, load_image_sequence_from_path,
    load_image_sequence_from_path_with_backend, load_image_sequence_from_url,
    load_image_sequence_from_url_with_mode, load_image_sequence_from_url_with_remote_options,
    load_image_sequence_from_url_with_remote_options_and_image_decode_backend,
    load_video_from_path, load_video_from_path_with_options, load_video_from_url_with_options,
    load_video_from_url_with_remote_options, DefaultMediaLoader, FrameSampling, FrameTiming,
    ImageDecodeBackend, ImageFrame, ImageSequence, LoadedMedia, LoopBehavior, MediaError,
    MediaHint, MediaLoader, MediaLocation, MediaSource, MediaType, PixelFormat, RemoteLoadOptions,
    RemoteReadMode, RemoteRedirectPolicy, VideoClip, VideoDecodeOptions, VideoFrame,
};
#[doc(inline)]
pub use output::{
    ProcessorMetadataName, ProcessorMetadataOutput, ProcessorMetadataValue, ProcessorOutput,
    ProcessorTensorName, ProcessorTensorOutput,
};
#[doc(inline)]
pub use postprocess::{
    binarize_mask, binary_mask_to_box, binary_mask_to_rle, binary_rle_to_mask, box_near_crop_edge,
    detection_box_iou, filter_generated_masks, generate_layered_crop_boxes, logc3_to_linear,
    mask_stability_score, non_max_suppression, normalized_point_grid, pad_crop_mask,
    post_process_binary_mask, post_process_coordinates, post_process_dense_map,
    post_process_depth_map, post_process_generated_masks, post_process_image_tensor,
    post_process_instance_segmentation, post_process_logc3_hdr_video_tensor,
    post_process_object_detection, post_process_panoptic_segmentation, post_process_recipe_outputs,
    post_process_semantic_segmentation, post_process_video_tensor, resize_padded_mask_logits,
    scale_detection_box, scale_image_point, BinaryMaskPostprocessOutput, BinaryRleMask,
    CoordinatePoint, CoordinatePostprocessOutput, CropGenerationOptions, DenseMapPostprocessOutput,
    DepthMapPostprocessOutput, DetectionBoundingBox, DetectionCenterBox, FilteredMask,
    GeneratedMaskPrediction, ImagePoint, LayeredCropBox, MaskFilterOptions,
    ObjectDetectionPrediction, OutputHookPostprocessOutput, RecipePostprocessContext,
    RecipePostprocessError, RecipePostprocessOutput, SegmentationPostProcessOptions,
    SegmentationPostProcessPrediction, SegmentationSegmentInfo, SemanticSegmentationPrediction,
    TensorPostprocessError, TokenSequencePostprocessOutput,
};
#[doc(inline)]
pub use processors::{
    BlipImageProcessor, BlipImageProcessorConfig, ClipImageProcessor, ClipImageProcessorConfig,
    DepthMapU16, DetrImageProcessor, DetrImageProcessorConfig, DocumentOcrImageProcessor,
    DocumentOcrImageProcessorConfig, EncoderGeometry, EncoderImageProcessor,
    EncoderImageProcessorConfig, EncoderImageProcessorPreset, Flux2ImageProcessor,
    Flux2ImageProcessorConfig, Gemma3ImageProcessor, Gemma3ImageProcessorConfig,
    HunyuanVideo15ImageProcessor, HunyuanVideo15ImageProcessorConfig, Idefics3ImageProcessor,
    Idefics3ImageProcessorConfig, InpaintOverlayContext, InpaintPreprocessOutput,
    JoyImageEditImageProcessor, JoyImageEditImageProcessorConfig, Ldm3dPostprocessOutput,
    Ldm3dPreprocessOutput, LlavaNextImageProcessor, LlavaNextImageProcessorConfig,
    Ltx2VideoHdrProcessor, Ltx2VideoHdrProcessorConfig, MarigoldImageProcessor,
    MarigoldImageProcessorConfig, MarigoldPreprocessOutput, MllamaImageProcessor,
    MllamaImageProcessorConfig, PaddleDetectionLimit, PixtralImageProcessor,
    PixtralImageProcessorConfig, PresetProcessorError, ProcessorConfigError, ProcessorFamilyConfig,
    QwenVlImageProcessor, QwenVlImageProcessorConfig, SamImageProcessor, SamImageProcessorConfig,
    ShortestEdgeResizeConfig, Swin2SrImageProcessor, Swin2SrImageProcessorConfig,
    TaskVisionColorMode, TaskVisionImageProcessor, TaskVisionImageProcessorConfig,
    TaskVisionKeypointMatchingOutput, TaskVisionPadding, TaskVisionPoseBox,
    TaskVisionProcessorFamily, TaskVisionProcessorPreset, TaskVisionResize,
    TaskVisionSegmentationRequest, VaeImageProcessor, VaeImageProcessorConfig,
    VaeImageProcessorLdm3d, VaeOutputType, VaePostprocessOutput, VideoMaeImageProcessor,
    VideoMaeImageProcessorConfig, VisualClozePreprocessOutput, VisualClozeProcessor,
    VisualClozeProcessorConfig, VitImageProcessor, VitImageProcessorConfig, VivitImageProcessor,
    VivitImageProcessorConfig, WanAnimateImageProcessor, WanAnimateImageProcessorConfig,
};
#[doc(inline)]
pub use recipe::{
    ProcessorRecipe, ProcessorRecipeInput, ProcessorRecipeOutput, ProcessorRecipePostprocess,
    ProcessorRecipeStage, RecipeAspectRatioCropStage, RecipeBinaryMaskPostprocess,
    RecipeCoordinatePostprocess, RecipeCoordinateTask, RecipeCropStage, RecipeDenseMapPostprocess,
    RecipeDenseMapTask, RecipeDepthPostprocess, RecipeDepthUnit, RecipeDetectionScoreMode,
    RecipeDocumentGeometryStage, RecipeError, RecipeImageSizeSource, RecipeImageSplitStage,
    RecipeObjectDetectionPostprocess, RecipeOutputHookPostprocess, RecipeOutputHookTask,
    RecipeOverlayStage, RecipePadStage, RecipePatchError, RecipePatchGridStage, RecipePatchStage,
    RecipeResizeStage, RecipeResizeTarget, RecipeSegmentationPostprocess,
    RecipeSegmentationPostprocessOptions, RecipeSegmentationStrategy, RecipeSegmentationTask,
    RecipeTiledCanvasStage, RecipeTokenSequenceDecoder, RecipeTokenSequencePostprocess,
    RecipeTokenSequenceTask, RecipeTokenVocabularySource, RecipeVideoTensorPostprocess,
    RecipeVideoTransferFunction,
};
#[doc(inline)]
pub use tensor::{
    DType, ImageLayout, Layout, QuantizationParams, Tensor, TensorData, TensorDataView,
    TensorError, TensorLeadingAxis, TensorView, VideoLayout,
};
#[doc(inline)]
pub use transforms::{
    area_resize_plan, aspect_ratio_crop_batch_metadata, aspect_ratio_crop_plan,
    binarize_mask_to_unit_f32, bottom_right_resize_pad_plan, center_crop_box, center_crop_frame,
    centered_padding, composite_mask_frame, concatenate_frames_horizontally_rgb,
    convert_frame_pixel_format, crop_frame, crop_region, denormalize_signed_to_unit,
    downsample_attention_mask, fit_inside_size, inpaint_overlay, is_latent_channel_count,
    longest_edge_resize_size, multiple_of_resize_plan, nested_frame_batch_padding_plan,
    nested_image_grid_metadata, normalize_detection_annotation, normalize_unit_to_signed,
    overlay_frame, pad_frame, pad_frame_symmetric_to_next_multiple,
    pad_frame_to_multiple_with_canvas_fill, pad_frame_with_canvas_fill,
    pad_image_sequence_frames_with_canvas_fill, pad_normalized_detection_annotation,
    pad_to_multiple_plan, pad_video_clip_frames_with_canvas_fill, padded_image_size,
    patch_aligned_resize_plan, patch_aligned_resize_size, patch_grid_batch_plan,
    patch_grid_image_patches, patch_grid_output_size, patch_grid_plan,
    post_process_sigmoid_best_object_detection, post_process_sigmoid_top_k_object_detection,
    prepare_coco_detection_annotation, repeat_last_frame_batch_to_multiple,
    repeat_last_image_sequence_to_multiple, repeat_last_video_clip_to_multiple,
    resize_center_crop_frame, resize_center_crop_plan, resize_detection_annotation,
    resize_fill_plan, resize_frame, resize_frame_to_area_limit,
    resize_frame_to_bottom_right_padded_frame, resize_frame_to_fill_frame,
    resize_frame_with_decision, resize_rgb_to_fill_frame, scale_factor_resize_plan,
    select_aspect_ratio_bucket, select_patch_grid_resolution, shortest_edge_resize_size,
    should_rotate_to_match_orientation, spatial_batch_padding_plan, split_image_batch_metadata,
    split_image_encoder_size, split_image_plan, split_image_resize_size,
    supported_tiled_canvas_grids, temporal_repeat_last_plan, tiled_canvas_aspect_ratio_id,
    tiled_canvas_aspect_ratio_mask, tiled_canvas_batch_metadata, tiled_canvas_plan,
    unpad_frame_to_size, validate_image_crop_box, validate_image_size_constraints,
    validate_padding_fill, video_clip_size_bucket_plan, video_size_bucket_plan, AreaResizePlan,
    AspectRatioCrop, AspectRatioCropBatchMetadata, AspectRatioCropOptions, AspectRatioCropPlan,
    AttentionMaskDownsample, BottomRightResizePadPlan, CanvasFill, CocoObjectAnnotation,
    DetectionAnnotation, ImageCropBox, ImageSize, ImageSizeConstraints, MultipleOfResizePlan,
    NestedFrameBatchPaddingPlan, NestedImageGridMetadata, NestedImageGridTarget,
    NormalizedDetectionAnnotation, OverlayPosition, PadToMultiplePlan, Padding,
    PatchAlignedResizePlan, PatchGridBatchPlan, PatchGridFrames, PatchGridPlan,
    ResizeCenterCropPlan, ResizeDecision, ResizeFillPlan, ResizeFilter, ResizeKernel, ResizeLimits,
    ResizeMode, ResizeParity, ResizeRounding, RgbCanvasFill, ScaleFactorResizePlan,
    SpatialBatchPaddingPlan, SplitImageBatchMetadata, SplitImagePlan, TemporalRepeatLastPlan,
    TiledCanvasBatchMetadata, TiledCanvasGrid, TiledCanvasPlan, TransformError, VideoSizeBucket,
    VideoSizeBucketPlan,
};
