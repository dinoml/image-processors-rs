//! Image transform and geometry helpers.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::media::{ImageFrame, ImageSequence, MediaError, PixelFormat, VideoClip, VideoFrame};
use image_resize_kernels::{
    ImageSize as KernelImageSize, ResizeCrop as KernelResizeCrop, ResizeError as KernelResizeError,
    ResizeFilter as KernelResizeFilter, ResizeProfile, ResizeWorkspace as KernelResizeWorkspace,
};

mod detection;
pub(crate) use detection::post_process_oneformer_panoptic_segmentation;
pub use detection::{
    detection_box_iou, non_max_suppression, normalize_detection_annotation,
    pad_normalized_detection_annotation, post_process_instance_segmentation,
    post_process_object_detection, post_process_panoptic_segmentation,
    post_process_semantic_segmentation, post_process_sigmoid_best_object_detection,
    post_process_sigmoid_top_k_object_detection, prepare_coco_detection_annotation,
    resize_detection_annotation, scale_detection_box, CocoObjectAnnotation, DetectionAnnotation,
    DetectionBoundingBox, DetectionCenterBox, NormalizedDetectionAnnotation,
    ObjectDetectionPrediction, SegmentationPostProcessOptions, SegmentationPostProcessPrediction,
    SegmentationSegmentInfo, SemanticSegmentationPrediction,
};
mod frame_ops;
#[cfg(test)]
use frame_ops::convert_to_rgb;
pub use frame_ops::{
    bottom_right_resize_pad_plan, center_crop_box, center_crop_frame, composite_mask_frame,
    concatenate_frames_horizontally_rgb, convert_frame_pixel_format, crop_frame, crop_region,
    multiple_of_resize_plan, overlay_frame, pad_frame, pad_frame_symmetric_to_next_multiple,
    pad_frame_to_multiple_with_canvas_fill, pad_frame_with_canvas_fill,
    pad_image_sequence_frames_with_canvas_fill, pad_to_multiple_plan,
    pad_video_clip_frames_with_canvas_fill, padded_image_size, repeat_last_frame_batch_to_multiple,
    repeat_last_image_sequence_to_multiple, repeat_last_video_clip_to_multiple,
    resize_center_crop_frame, resize_center_crop_frame_with_decision, resize_center_crop_plan,
    resize_fill_plan, resize_frame, resize_frame_to_area_limit,
    resize_frame_to_bottom_right_padded_frame, resize_frame_to_fill_frame,
    resize_frame_with_decision, resize_rgb_to_fill_frame, temporal_repeat_last_plan,
    unpad_frame_to_size, validate_padding_fill,
};
use frame_ops::{crop_image_data, frame_size, new_frame};
pub(crate) use frame_ops::{
    resize_frame_owned_with_decision, resize_frame_owned_with_decision_workspace,
    resize_frame_to_f32_torchvision,
};
mod layout_plans;
use layout_plans::compare_ratios_less;
pub use layout_plans::{
    aspect_ratio_crop_batch_metadata, aspect_ratio_crop_plan, downsample_attention_mask,
    nested_frame_batch_padding_plan, nested_image_grid_metadata, patch_aligned_resize_plan,
    patch_aligned_resize_size, spatial_batch_padding_plan, split_image_batch_metadata,
    split_image_encoder_size, split_image_plan, split_image_resize_size, AspectRatioCrop,
    AspectRatioCropBatchMetadata, AspectRatioCropOptions, AspectRatioCropPlan,
    AttentionMaskDownsample, NestedFrameBatchPaddingPlan, NestedImageGridMetadata,
    NestedImageGridTarget, OverlayPosition, PatchAlignedResizePlan, ResizeLimits,
    SpatialBatchPaddingPlan, SplitImageBatchMetadata, SplitImagePlan,
};
mod masks;
mod patch_grid;
pub use masks::{
    binarize_mask, binarize_mask_to_unit_f32, binary_mask_to_box, binary_mask_to_rle,
    binary_rle_to_mask, box_near_crop_edge, filter_generated_masks, generate_layered_crop_boxes,
    mask_stability_score, normalized_point_grid, pad_crop_mask, post_process_binary_mask,
    post_process_generated_masks, resize_padded_mask_logits, scale_image_point, BinaryRleMask,
    CropGenerationOptions, FilteredMask, GeneratedMaskPrediction, ImagePoint, LayeredCropBox,
    MaskFilterOptions,
};
pub use patch_grid::{
    patch_grid_batch_plan, patch_grid_image_patches, patch_grid_image_patches_with_decision,
    patch_grid_output_size, patch_grid_plan, select_patch_grid_resolution, PatchGridBatchPlan,
    PatchGridFrames, PatchGridPlan,
};
mod tiled_canvas;
pub use tiled_canvas::{
    supported_tiled_canvas_grids, tiled_canvas_aspect_ratio_id, tiled_canvas_aspect_ratio_mask,
    tiled_canvas_batch_metadata, tiled_canvas_plan, TiledCanvasBatchMetadata, TiledCanvasGrid,
    TiledCanvasPlan,
};

const IDEFICS3_MAX_IMAGE_EDGE: usize = 4096;
const LOGC3_A: f32 = 5.555556;
const LOGC3_B: f32 = 0.052272;
const LOGC3_C: f32 = 0.247190;
const LOGC3_D: f32 = 0.385537;
const LOGC3_E: f32 = 5.367655;
const LOGC3_F: f32 = 0.092809;
const LOGC3_CUT: f32 = 0.010591;

/// Image dimensions in height-width order.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ImageSize {
    /// Image height in pixels.
    pub height: usize,
    /// Image width in pixels.
    pub width: usize,
}

impl ImageSize {
    /// Creates an image size with positive dimensions.
    ///
    /// # Errors
    ///
    /// Returns an error when either dimension is zero.
    pub fn new(height: usize, width: usize) -> Result<Self, TransformError> {
        if height == 0 || width == 0 {
            return Err(TransformError::InvalidSize { height, width });
        }
        Ok(Self { height, width })
    }

    /// Rounds both dimensions down to a multiple of `factor`.
    ///
    /// # Errors
    ///
    /// Returns an error when `factor` is zero or rounding produces a zero dimension.
    pub fn round_down_to_multiple(self, factor: usize) -> Result<Self, TransformError> {
        if factor == 0 {
            return Err(TransformError::InvalidScaleFactor(factor));
        }
        Self::new(
            self.height - self.height % factor,
            self.width - self.width % factor,
        )
    }

    /// Qwen-VL style resize shape computation.
    ///
    /// Dimensions are rounded to a multiple of `factor`, constrained by total
    /// pixel bounds, and kept close to the source aspect ratio.
    ///
    /// # Errors
    ///
    /// Returns an error when limits are invalid, unachievable, or arithmetic overflows.
    pub fn smart_resize(self, limits: ResizeLimits) -> Result<Self, TransformError> {
        if limits.factor == 0
            || limits.min_pixels == 0
            || limits.max_pixels == 0
            || limits.min_pixels > limits.max_pixels
        {
            return Err(TransformError::InvalidSmartResizeParams);
        }
        let minimum_factor_pixels = limits
            .factor
            .checked_mul(limits.factor)
            .ok_or(TransformError::ImageSizeOverflow)?;
        if limits.max_pixels < minimum_factor_pixels {
            return Err(TransformError::InvalidSmartResizeParams);
        }

        let source_pixels = checked_pixels(self)?;
        let longer = self.height.max(self.width) as f64;
        let shorter = self.height.min(self.width) as f64;
        let aspect_ratio = longer / shorter;
        if aspect_ratio > 200.0 {
            return Err(TransformError::InvalidAspectRatio(aspect_ratio));
        }

        let factor_f = limits.factor as f64;
        let mut resized_height = multiple_from_units(
            (self.height as f64 / factor_f).round().max(1.0),
            limits.factor,
        )?;
        let mut resized_width = multiple_from_units(
            (self.width as f64 / factor_f).round().max(1.0),
            limits.factor,
        )?;
        let pixel_count = checked_area(resized_height, resized_width)?;

        if pixel_count > limits.max_pixels {
            let beta = (source_pixels as f64 / limits.max_pixels as f64).sqrt();
            resized_height = limits.factor.max(multiple_from_units(
                (self.height as f64 / beta / factor_f).floor(),
                limits.factor,
            )?);
            resized_width = limits.factor.max(multiple_from_units(
                (self.width as f64 / beta / factor_f).floor(),
                limits.factor,
            )?);
        } else if pixel_count < limits.min_pixels {
            let beta = (limits.min_pixels as f64 / source_pixels as f64).sqrt();
            resized_height =
                multiple_from_units((self.height as f64 * beta / factor_f).ceil(), limits.factor)?;
            resized_width =
                multiple_from_units((self.width as f64 * beta / factor_f).ceil(), limits.factor)?;
        }

        let resized_pixels = checked_area(resized_height, resized_width)?;
        if resized_pixels < limits.min_pixels || resized_pixels > limits.max_pixels {
            return Err(TransformError::InvalidSmartResizeParams);
        }

        Self::new(resized_height, resized_width)
    }
}

/// Video bucket dimensions in frame-height-width order.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct VideoSizeBucket {
    /// Number of frames in the bucket.
    pub frame_count: usize,
    /// Spatial frame size in height-width order.
    pub size: ImageSize,
}

impl VideoSizeBucket {
    /// Creates a video size bucket with positive frame and spatial dimensions.
    ///
    /// # Errors
    ///
    /// Returns an error when `frame_count` is zero or either spatial dimension
    /// is zero.
    pub fn new(frame_count: usize, size: ImageSize) -> Result<Self, TransformError> {
        validate_frame_count(frame_count)?;
        validate_size(size)?;
        Ok(Self { frame_count, size })
    }
}

/// Pixel padding in top-right-bottom-left order.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct Padding {
    /// Pixels added above the image.
    pub top: usize,
    /// Pixels added to the right of the image.
    pub right: usize,
    /// Pixels added below the image.
    pub bottom: usize,
    /// Pixels added to the left of the image.
    pub left: usize,
}

impl Padding {
    /// Creates padding in top-right-bottom-left order.
    pub fn new(top: usize, right: usize, bottom: usize, left: usize) -> Self {
        Self {
            top,
            right,
            bottom,
            left,
        }
    }

    /// Creates equal padding on every edge.
    pub fn all(value: usize) -> Self {
        Self::new(value, value, value, value)
    }

    /// Creates symmetric vertical and horizontal padding.
    pub fn symmetric(vertical: usize, horizontal: usize) -> Self {
        Self::new(vertical, horizontal, vertical, horizontal)
    }

    /// Returns true when no padding is requested.
    pub fn is_empty(self) -> bool {
        self.top == 0 && self.right == 0 && self.bottom == 0 && self.left == 0
    }
}

/// Resize plan that rounds requested dimensions down to a scale-factor multiple.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ScaleFactorResizePlan {
    /// Original input image dimensions.
    pub original_size: ImageSize,
    /// Requested image dimensions before scale-factor rounding.
    pub requested_size: ImageSize,
    /// Scale factor used for spatial rounding.
    pub scale_factor: usize,
    /// Dimensions rounded down to multiples of `scale_factor`.
    pub resized_size: ImageSize,
    /// Spatial dimensions after dividing by `scale_factor`.
    pub latent_size: ImageSize,
}

/// Minimum and maximum bounds for validating image dimensions.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct ImageSizeConstraints {
    /// Maximum allowed width-height or height-width ratio.
    pub max_aspect_ratio: f64,
    /// Minimum allowed width and height in pixels.
    pub min_side_length: usize,
}

/// Selected video bucket for frame-count and spatial resizing.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VideoSizeBucketPlan {
    /// Source video frame count and spatial dimensions.
    pub source: VideoSizeBucket,
    /// Selected bucket dimensions.
    pub selected: VideoSizeBucket,
}

impl VideoSizeBucketPlan {
    /// Returns true when the selected bucket matches the source exactly.
    pub const fn is_identity(self) -> bool {
        self.source.frame_count == self.selected.frame_count
            && self.source.size.height == self.selected.size.height
            && self.source.size.width == self.selected.size.width
    }

    /// Returns the absolute frame-count difference between source and bucket.
    pub const fn frame_delta(self) -> usize {
        self.source.frame_count.abs_diff(self.selected.frame_count)
    }
}

/// Area-limiting resize plan for an image.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AreaResizePlan {
    /// Original input image dimensions.
    pub original_size: ImageSize,
    /// Maximum pixel area requested by the caller.
    pub target_area: usize,
    /// Dimensions after area limiting.
    pub resized_size: ImageSize,
}

impl AreaResizePlan {
    /// Returns true when the image should be resized.
    pub fn should_resize(self) -> bool {
        self.original_size != self.resized_size
    }
}

/// Resize-to-cover and center-crop plan.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResizeCenterCropPlan {
    /// Original input image dimensions.
    pub original_size: ImageSize,
    /// Requested output dimensions.
    pub target_size: ImageSize,
    /// Intermediate resized dimensions before cropping.
    pub resized_size: ImageSize,
    /// Center crop box in the intermediate resized image.
    pub crop_box: ImageCropBox,
}

/// Rounding mode for intermediate resize dimensions.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ResizeRounding {
    /// Round intermediate dimensions down.
    Floor,
    /// Round intermediate dimensions up.
    Ceil,
}

/// Resize-size plan that rounds dimensions down to per-axis multiples.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MultipleOfResizePlan {
    /// Original input image dimensions.
    pub original_size: ImageSize,
    /// Requested image dimensions before multiple rounding.
    pub requested_size: ImageSize,
    /// Required height and width multiples.
    pub multiples: ImageSize,
    /// Dimensions rounded down to the requested multiples.
    pub resized_size: ImageSize,
}

/// Padding plan that rounds dimensions up to per-axis multiples.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PadToMultiplePlan {
    /// Original input image dimensions.
    pub source_size: ImageSize,
    /// Required height and width multiples.
    pub multiples: ImageSize,
    /// Dimensions after bottom/right padding.
    pub padded_size: ImageSize,
    /// Bottom/right padding that expands `source_size` to `padded_size`.
    pub padding: Padding,
}

impl PadToMultiplePlan {
    /// Returns true when the source already satisfies the requested multiples.
    pub fn is_identity(self) -> bool {
        self.padding.is_empty()
    }

    /// Returns the crop box that removes bottom/right padding.
    pub fn content_crop_box(self) -> ImageCropBox {
        ImageCropBox::new(0, 0, self.source_size.width, self.source_size.height)
    }
}

/// Temporal padding plan that repeats the final frame up to a multiple.
///
/// This describes the common video/VLM behavior where incomplete temporal
/// patch groups are filled by reusing the last decoded frame.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TemporalRepeatLastPlan {
    /// Number of decoded source frames.
    pub source_frame_count: usize,
    /// Requested temporal multiple.
    pub multiple: usize,
    /// Number of frames after repeat-last padding.
    pub target_frame_count: usize,
    /// Number of repeated final frames appended to the source frame batch.
    pub padding_frames: usize,
}

impl TemporalRepeatLastPlan {
    /// Returns true when no repeat-last padding is needed.
    pub const fn is_identity(self) -> bool {
        self.padding_frames == 0
    }

    /// Returns the number of temporal groups implied by this plan.
    pub fn group_count(self) -> usize {
        self.target_frame_count
            .checked_div(self.multiple)
            .unwrap_or(0)
    }

    /// Maps a padded output frame index back to a source frame index.
    ///
    /// Returns `None` when `output_index` is outside the padded frame range.
    pub fn source_index(self, output_index: usize) -> Option<usize> {
        if self.source_frame_count != 0 && output_index < self.target_frame_count {
            Some(output_index.min(self.source_frame_count - 1))
        } else {
            None
        }
    }

    /// Iterates over source frame indices for every padded output position.
    pub fn source_indices(self) -> impl Iterator<Item = usize> {
        let target_frame_count = if self.source_frame_count == 0 {
            0
        } else {
            self.target_frame_count
        };
        let last_source_index = self.source_frame_count.saturating_sub(1);
        (0..target_frame_count).map(move |index| index.min(last_source_index))
    }
}

/// Resize-and-fill plan.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResizeFillPlan {
    /// Source image dimensions.
    pub source_size: ImageSize,
    /// Requested output dimensions.
    pub target_size: ImageSize,
    /// Aspect-preserving resized dimensions before centering.
    pub resized_size: ImageSize,
    /// Centering padding around the resized image.
    pub padding: Padding,
}

/// Aspect-preserving resize followed by bottom/right canvas padding.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BottomRightResizePadPlan {
    /// Source image dimensions.
    pub source_size: ImageSize,
    /// Requested output dimensions.
    pub target_size: ImageSize,
    /// Aspect-preserving resized dimensions before padding.
    pub resized_size: ImageSize,
    /// Bottom/right padding that expands the resized image to `target_size`.
    pub padding: Padding,
}

/// Canvas fill behavior for RGB resize-and-fill preprocessing.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum RgbCanvasFill {
    /// Fill empty regions with an RGB color.
    ConstantRgb([u8; 3]),
    /// Fill empty regions by extending the resized image edge pixels.
    ImageEdges,
}

impl Default for RgbCanvasFill {
    fn default() -> Self {
        Self::ConstantRgb([0, 0, 0])
    }
}

/// Canvas fill behavior for resize-and-fill preprocessing.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum CanvasFill {
    /// Fill empty regions with a scalar value or one byte per frame channel.
    Constant(Vec<u8>),
    /// Fill empty regions by extending the resized image edge pixels.
    ImageEdges,
    /// Fill empty regions by reflecting image pixels at the canvas boundary.
    ReflectImage,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BorderFillMode {
    Replicate,
    Reflect,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PastedRegion {
    left: usize,
    top: usize,
    right: usize,
    bottom: usize,
}

impl CanvasFill {
    /// Creates a constant canvas-fill descriptor.
    pub fn constant(fill: impl Into<Vec<u8>>) -> Self {
        Self::Constant(fill.into())
    }
}

impl Default for CanvasFill {
    fn default() -> Self {
        Self::Constant(vec![0])
    }
}

impl From<RgbCanvasFill> for CanvasFill {
    fn from(fill: RgbCanvasFill) -> Self {
        match fill {
            RgbCanvasFill::ConstantRgb(rgb) => Self::Constant(rgb.to_vec()),
            RgbCanvasFill::ImageEdges => Self::ImageEdges,
        }
    }
}

/// Absolute pixel crop box in corner coordinates.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ImageCropBox {
    /// Left crop coordinate in pixels.
    pub x_min: usize,
    /// Top crop coordinate in pixels.
    pub y_min: usize,
    /// Right crop coordinate in pixels.
    pub x_max: usize,
    /// Bottom crop coordinate in pixels.
    pub y_max: usize,
}

impl ImageCropBox {
    /// Creates an image crop box from absolute corner coordinates.
    pub fn new(x_min: usize, y_min: usize, x_max: usize, y_max: usize) -> Self {
        Self {
            x_min,
            y_min,
            x_max,
            y_max,
        }
    }
}

/// Resize resampling filter.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum ResizeFilter {
    /// Nearest-neighbor filtering.
    Nearest,
    /// Bilinear filtering.
    Bilinear,
    /// Bicubic filtering.
    Bicubic,
    /// Lanczos filtering.
    Lanczos,
}

impl ResizeFilter {
    /// Returns the explicit resize decision for this filter.
    pub fn decision(self) -> ResizeDecision {
        ResizeDecision::resampling(self)
    }

    /// Returns the concrete kernel selected by this filter.
    pub fn kernel(self) -> ResizeKernel {
        self.decision().kernel()
    }
}

/// Concrete resize kernel used by the resize path.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum ResizeKernel {
    /// Nearest-neighbor sampling.
    Nearest,
    /// Triangle filter used for bilinear-style resizing.
    Triangle,
    /// Catmull-Rom cubic filter used for bicubic-style resizing.
    CatmullRom,
    /// Lanczos3 filter used for high-quality resizing.
    Lanczos3,
}

/// Resize parity policy for model preprocessing.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub enum ResizeParity {
    /// Use the configured resampling filter.
    #[default]
    Resampling,
    /// Use a compatibility path for matching upstream processor fixtures.
    Compatibility,
    /// Use Torchvision's antialiased tensor-resize rounding behavior.
    Torchvision,
    /// Preserve pixel identities with nearest-neighbor sampling.
    PixelExact,
}

/// Explicit resize implementation decision.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResizeDecision {
    filter: ResizeFilter,
    kernel: ResizeKernel,
    parity: ResizeParity,
}

impl ResizeDecision {
    /// Creates a resampling resize decision.
    pub fn resampling(filter: ResizeFilter) -> Self {
        Self {
            filter,
            kernel: kernel_for_filter(filter),
            parity: ResizeParity::Resampling,
        }
    }

    /// Creates a pixel-exact resize decision.
    pub fn pixel_exact() -> Self {
        Self {
            filter: ResizeFilter::Nearest,
            kernel: ResizeKernel::Nearest,
            parity: ResizeParity::PixelExact,
        }
    }

    /// Creates a compatibility resize decision.
    pub fn compatibility(filter: ResizeFilter) -> Self {
        Self {
            filter,
            kernel: kernel_for_filter(filter),
            parity: ResizeParity::Compatibility,
        }
    }

    /// Creates a Torchvision-compatible resize decision.
    pub fn torchvision(filter: ResizeFilter) -> Self {
        Self {
            filter,
            kernel: kernel_for_filter(filter),
            parity: ResizeParity::Torchvision,
        }
    }

    /// Creates a resize decision from a filter and parity policy.
    ///
    /// # Errors
    ///
    /// Returns an error when `PixelExact` is requested with a blending filter.
    pub fn new(filter: ResizeFilter, parity: ResizeParity) -> Result<Self, TransformError> {
        match parity {
            ResizeParity::Resampling => Ok(Self::resampling(filter)),
            ResizeParity::Compatibility => Ok(Self::compatibility(filter)),
            ResizeParity::Torchvision => Ok(Self::torchvision(filter)),
            ResizeParity::PixelExact if filter == ResizeFilter::Nearest => Ok(Self::pixel_exact()),
            ResizeParity::PixelExact => Err(TransformError::InvalidResizeParity { filter, parity }),
        }
    }

    /// Returns the requested public filter.
    pub fn filter(self) -> ResizeFilter {
        self.filter
    }

    /// Returns the concrete backend kernel.
    pub fn kernel(self) -> ResizeKernel {
        self.kernel
    }

    /// Returns the resize parity policy.
    pub fn parity(self) -> ResizeParity {
        self.parity
    }
}

fn kernel_for_filter(filter: ResizeFilter) -> ResizeKernel {
    match filter {
        ResizeFilter::Nearest => ResizeKernel::Nearest,
        ResizeFilter::Bilinear => ResizeKernel::Triangle,
        ResizeFilter::Bicubic => ResizeKernel::CatmullRom,
        ResizeFilter::Lanczos => ResizeKernel::Lanczos3,
    }
}

/// Resize sizing mode.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum ResizeMode {
    /// Resize directly to the target dimensions.
    Default,
    /// Preserve aspect ratio and fill remaining space from edge pixels.
    Fill,
    /// Preserve aspect ratio and crop overflow around the center.
    Crop,
}

/// Computes an aspect-preserving resize size with a fixed longest edge.
///
/// The source aspect ratio is preserved, and the longer output edge is set to
/// `longest_edge`.
///
/// # Errors
///
/// Returns an error when dimensions are invalid, `longest_edge` is zero, or
/// arithmetic overflows.
pub fn longest_edge_resize_size(
    original_size: ImageSize,
    longest_edge: usize,
) -> Result<ImageSize, TransformError> {
    validate_size(original_size)?;
    if longest_edge == 0 {
        return Err(TransformError::InvalidScaleFactor(longest_edge));
    }
    let max_edge = original_size.height.max(original_size.width);
    ImageSize::new(
        round_ratio_dimension(original_size.height, longest_edge, max_edge)?,
        round_ratio_dimension(original_size.width, longest_edge, max_edge)?,
    )
}

/// Computes an aspect-preserving resize size with a fixed shortest edge.
///
/// When `longest_edge` is set, the requested shortest edge is capped so the
/// resized longest edge does not exceed that maximum.
///
/// # Errors
///
/// Returns an error when dimensions are invalid, `shortest_edge` is zero,
/// `longest_edge` is zero, or arithmetic overflows.
pub fn shortest_edge_resize_size(
    original_size: ImageSize,
    shortest_edge: usize,
    longest_edge: Option<usize>,
) -> Result<ImageSize, TransformError> {
    validate_size(original_size)?;
    if shortest_edge == 0 {
        return Err(TransformError::InvalidScaleFactor(shortest_edge));
    }
    if longest_edge == Some(0) {
        return Err(TransformError::InvalidScaleFactor(0));
    }

    let source_shortest = original_size.height.min(original_size.width);
    let source_longest = original_size.height.max(original_size.width);
    let mut target_shortest = shortest_edge;
    if let Some(longest_edge) = longest_edge {
        let capped_shortest = round_ratio_dimension(source_shortest, longest_edge, source_longest)?;
        if capped_shortest <= target_shortest {
            target_shortest = capped_shortest;
        }
    }

    if (original_size.width <= original_size.height && original_size.width == target_shortest)
        || (original_size.height <= original_size.width && original_size.height == target_shortest)
    {
        return Ok(original_size);
    }

    let (height, width) = if original_size.width < original_size.height {
        (
            checked_ratio_dimension(original_size.height, target_shortest, original_size.width)?,
            target_shortest,
        )
    } else {
        (
            target_shortest,
            checked_ratio_dimension(original_size.width, target_shortest, original_size.height)?,
        )
    };

    ImageSize::new(height.max(1), width.max(1))
}

/// Returns whether a source should rotate to match target orientation.
///
/// # Errors
///
/// Returns an error when source or target dimensions are invalid.
pub fn should_rotate_to_match_orientation(
    source_size: ImageSize,
    target_size: ImageSize,
) -> Result<bool, TransformError> {
    validate_size(source_size)?;
    validate_size(target_size)?;
    Ok(
        (target_size.width < target_size.height && source_size.width > source_size.height)
            || (target_size.width > target_size.height && source_size.width < source_size.height),
    )
}

/// Computes the largest aspect-preserving size that fits inside a canvas.
///
/// The output fits inside `target_size`, preserving aspect ratio with integer
/// truncation. Images already within the target are left unchanged.
///
/// # Errors
///
/// Returns an error when dimensions are invalid or arithmetic overflows.
pub fn fit_inside_size(
    source_size: ImageSize,
    target_size: ImageSize,
) -> Result<ImageSize, TransformError> {
    validate_size(source_size)?;
    validate_size(target_size)?;
    if source_size.height <= target_size.height && source_size.width <= target_size.width {
        return Ok(source_size);
    }

    let height_limited = (target_size.height as u128)
        .checked_mul(source_size.width as u128)
        .ok_or(TransformError::ImageSizeOverflow)?
        <= (target_size.width as u128)
            .checked_mul(source_size.height as u128)
            .ok_or(TransformError::ImageSizeOverflow)?;

    let (height, width) = if height_limited {
        (
            target_size.height,
            checked_ratio_dimension(source_size.width, target_size.height, source_size.height)?,
        )
    } else {
        (
            checked_ratio_dimension(source_size.height, target_size.width, source_size.width)?,
            target_size.width,
        )
    };

    ImageSize::new(height.max(1), width.max(1))
}

/// Computes centered padding to place an image on a larger canvas.
///
/// # Errors
///
/// Returns an error when dimensions are invalid or the source image is larger
/// than the target canvas.
pub fn centered_padding(source: ImageSize, target: ImageSize) -> Result<Padding, TransformError> {
    validate_size(source)?;
    validate_size(target)?;
    if source.height > target.height || source.width > target.width {
        return Err(TransformError::CropTooLarge {
            source_size: target,
            target_size: source,
        });
    }

    let horizontal = target.width - source.width;
    let vertical = target.height - source.height;
    let left = horizontal / 2;
    let top = vertical / 2;
    Ok(Padding::new(top, horizontal - left, vertical - top, left))
}

/// Builds a scale-factor resize plan.
///
/// # Errors
///
/// Returns an error when dimensions are invalid, `scale_factor` is zero, or
/// rounding would produce zero-sized output.
pub fn scale_factor_resize_plan(
    original_size: ImageSize,
    requested_size: Option<ImageSize>,
    scale_factor: usize,
) -> Result<ScaleFactorResizePlan, TransformError> {
    validate_size(original_size)?;
    if scale_factor == 0 {
        return Err(TransformError::InvalidScaleFactor(scale_factor));
    }
    let requested_size = requested_size.unwrap_or(original_size);
    validate_size(requested_size)?;
    let resized_size = requested_size.round_down_to_multiple(scale_factor)?;
    let latent_size = ImageSize::new(
        resized_size.height / scale_factor,
        resized_size.width / scale_factor,
    )?;

    Ok(ScaleFactorResizePlan {
        original_size,
        requested_size,
        scale_factor,
        resized_size,
        latent_size,
    })
}

/// Validates an image size against common minimum-side and aspect-ratio limits.
///
/// # Errors
///
/// Returns an error when dimensions are invalid, constraints are invalid, the
/// image is too small, or the aspect ratio exceeds the configured maximum.
pub fn validate_image_size_constraints(
    image_size: ImageSize,
    constraints: ImageSizeConstraints,
) -> Result<(), TransformError> {
    validate_size(image_size)?;
    validate_image_constraints(constraints)?;

    if image_size.width < constraints.min_side_length
        || image_size.height < constraints.min_side_length
    {
        return Err(TransformError::ImageTooSmall {
            size: image_size,
            min_side_length: constraints.min_side_length,
        });
    }

    let aspect_ratio = (image_size.width as f64 / image_size.height as f64)
        .max(image_size.height as f64 / image_size.width as f64);
    if aspect_ratio > constraints.max_aspect_ratio {
        return Err(TransformError::AspectRatioTooLarge {
            aspect_ratio,
            max_aspect_ratio: constraints.max_aspect_ratio,
        });
    }

    Ok(())
}

/// Selects the candidate size with the closest height-width aspect ratio.
///
/// # Errors
///
/// Returns an error when `original_size` is invalid, `candidate_sizes` is
/// empty, or any candidate dimensions are invalid.
pub fn select_aspect_ratio_bucket(
    original_size: ImageSize,
    candidate_sizes: &[ImageSize],
) -> Result<ImageSize, TransformError> {
    validate_size(original_size)?;
    if candidate_sizes.is_empty() {
        return Err(TransformError::EmptyResolutionCandidates);
    }

    let source_ratio = original_size.height as f64 / original_size.width as f64;
    let mut best_size = None;
    let mut best_delta = f64::INFINITY;
    for candidate in candidate_sizes.iter().copied() {
        validate_size(candidate)?;
        let candidate_ratio = candidate.height as f64 / candidate.width as f64;
        let delta = (candidate_ratio - source_ratio).abs();
        if delta < best_delta {
            best_delta = delta;
            best_size = Some(candidate);
        }
    }

    best_size.ok_or(TransformError::EmptyResolutionCandidates)
}

/// Selects the closest video size bucket for a source frame count and size.
///
/// Candidate buckets are compared by closest frame count first, then closest
/// height-width aspect ratio, then closest pixel area. Candidate order is used
/// as the final tie-breaker.
///
/// # Errors
///
/// Returns an error when the source bucket is invalid, no candidates are
/// provided, any candidate is invalid, or area arithmetic overflows.
pub fn video_size_bucket_plan(
    source: VideoSizeBucket,
    candidate_buckets: &[VideoSizeBucket],
) -> Result<VideoSizeBucketPlan, TransformError> {
    validate_video_size_bucket(source)?;
    if candidate_buckets.is_empty() {
        return Err(TransformError::EmptyResolutionCandidates);
    }

    let source_ratio = source.size.height as f64 / source.size.width as f64;
    let source_area = checked_pixels(source.size)?;
    let mut best = None;
    for bucket in candidate_buckets.iter().copied() {
        validate_video_size_bucket(bucket)?;
        let candidate = VideoSizeBucketCandidate {
            bucket,
            frame_delta: source.frame_count.abs_diff(bucket.frame_count),
            aspect_ratio_delta: (bucket.size.height as f64 / bucket.size.width as f64
                - source_ratio)
                .abs(),
            area_delta: source_area.abs_diff(checked_pixels(bucket.size)?),
        };
        if best.is_none_or(|best| is_better_video_size_bucket(candidate, best)) {
            best = Some(candidate);
        }
    }

    best.map(|candidate| VideoSizeBucketPlan {
        source,
        selected: candidate.bucket,
    })
    .ok_or(TransformError::EmptyResolutionCandidates)
}

/// Selects the closest video size bucket for a decoded clip.
///
/// The clip must contain at least one frame and all frames must have the same
/// spatial dimensions. The returned plan uses the clip length and frame size as
/// the source bucket.
///
/// # Errors
///
/// Returns an error when the clip is empty, frames have mixed spatial sizes,
/// candidates are invalid, no candidates are provided, or area arithmetic
/// overflows.
pub fn video_clip_size_bucket_plan(
    clip: &VideoClip,
    candidate_buckets: &[VideoSizeBucket],
) -> Result<VideoSizeBucketPlan, TransformError> {
    let first = clip
        .frames()
        .first()
        .ok_or(TransformError::EmptyFrameBatch)?;
    let source_size = frame_size(first.image());
    for frame in clip.frames() {
        let actual = frame_size(frame.image());
        if actual != source_size {
            return Err(TransformError::IncompatibleFrameSize {
                frame: "video",
                expected: source_size,
                actual,
            });
        }
    }

    video_size_bucket_plan(
        VideoSizeBucket {
            frame_count: clip.len(),
            size: source_size,
        },
        candidate_buckets,
    )
}

#[derive(Clone, Copy, Debug)]
struct VideoSizeBucketCandidate {
    bucket: VideoSizeBucket,
    frame_delta: usize,
    aspect_ratio_delta: f64,
    area_delta: usize,
}

fn is_better_video_size_bucket(
    candidate: VideoSizeBucketCandidate,
    best: VideoSizeBucketCandidate,
) -> bool {
    if candidate.frame_delta != best.frame_delta {
        return candidate.frame_delta < best.frame_delta;
    }
    if candidate.aspect_ratio_delta < best.aspect_ratio_delta {
        return true;
    }
    if candidate.aspect_ratio_delta > best.aspect_ratio_delta {
        return false;
    }
    candidate.area_delta < best.area_delta
}

/// Builds an area-limiting resize plan.
///
/// Images whose area is at or below `target_area` keep their original size.
/// Larger images are scaled by `sqrt(target_area / original_area)` and each
/// positive dimension is truncated to an integer.
///
/// # Errors
///
/// Returns an error when dimensions are invalid, `target_area` is zero, or
/// arithmetic overflows.
pub fn area_resize_plan(
    original_size: ImageSize,
    target_area: usize,
) -> Result<AreaResizePlan, TransformError> {
    validate_size(original_size)?;
    if target_area == 0 {
        return Err(TransformError::InvalidTargetArea(target_area));
    }

    let original_area = checked_pixels(original_size)?;
    let resized_size = if original_area <= target_area {
        original_size
    } else {
        let scale = (target_area as f64 / original_area as f64).sqrt();
        ImageSize::new(
            floor_scaled_dimension(original_size.height, scale)?,
            floor_scaled_dimension(original_size.width, scale)?,
        )?
    };

    Ok(AreaResizePlan {
        original_size,
        target_area,
        resized_size,
    })
}

/// Returns true when a tensor channel count matches a latent channel count.
///
/// # Errors
///
/// Returns an error when `latent_channels` is zero.
pub fn is_latent_channel_count(
    channel_count: usize,
    latent_channels: usize,
) -> Result<bool, TransformError> {
    if latent_channels == 0 {
        return Err(TransformError::InvalidScaleFactor(latent_channels));
    }
    Ok(channel_count == latent_channels)
}

/// Normalizes image values from `[0, 1]` to `[-1, 1]`.
///
/// # Errors
///
/// Returns an error when any value is non-finite.
pub fn normalize_unit_to_signed(values: &[f32]) -> Result<Vec<f32>, TransformError> {
    values
        .iter()
        .copied()
        .map(|value| {
            validate_image_value(value)?;
            Ok(2.0 * value - 1.0)
        })
        .collect()
}

/// Denormalizes image values from `[-1, 1]` to clamped `[0, 1]`.
///
/// # Errors
///
/// Returns an error when any value is non-finite.
pub fn denormalize_signed_to_unit(values: &[f32]) -> Result<Vec<f32>, TransformError> {
    values
        .iter()
        .copied()
        .map(|value| {
            validate_image_value(value)?;
            Ok((value * 0.5 + 0.5).clamp(0.0, 1.0))
        })
        .collect()
}

/// Applies the ARRI LogC3 EI 800 inverse transform.
///
/// Input values are clamped to `[0, 1]` before applying the piecewise LogC3
/// inverse curve.
///
/// # Errors
///
/// Returns an error when any input value is non-finite.
pub fn logc3_to_linear(values: &[f32]) -> Result<Vec<f32>, TransformError> {
    values.iter().copied().map(logc3_value_to_linear).collect()
}

fn logc3_value_to_linear(value: f32) -> Result<f32, TransformError> {
    validate_image_value(value)?;
    let logc = value.clamp(0.0, 1.0);
    let cut_log = LOGC3_E * LOGC3_CUT + LOGC3_F;
    if logc >= cut_log {
        Ok((10.0_f32.powf((logc - LOGC3_D) / LOGC3_C) - LOGC3_B) / LOGC3_A)
    } else {
        Ok((logc - LOGC3_F) / LOGC3_E)
    }
}

/// Composites a generated image over an original image using a mask.
///
/// When `crop_box` is provided, the generated image is resized to that crop
/// and pasted onto a blank canvas before mask compositing.
///
/// # Errors
///
/// Returns an error when frame sizes or formats are invalid, crop geometry is
/// invalid, resizing fails, or output frame invariants fail.
pub fn inpaint_overlay(
    original_image: &ImageFrame,
    generated_image: &ImageFrame,
    mask: &ImageFrame,
    crop_box: Option<ImageCropBox>,
) -> Result<ImageFrame, TransformError> {
    let original_size = frame_size(original_image);
    validate_frame_size("mask", original_size, frame_size(mask))?;

    let original_rgb = convert_frame_pixel_format(original_image, PixelFormat::Rgb8)?;
    let generated_canvas = match crop_box {
        Some(crop_box) => {
            validate_image_crop_box(crop_box, original_size)?;
            let crop_size = image_crop_box_size(crop_box)?;
            let generated_rgb = convert_frame_pixel_format(generated_image, PixelFormat::Rgb8)?;
            let resized_generated = resize_frame(
                &generated_rgb,
                crop_size,
                ResizeFilter::Lanczos,
                ResizeMode::Crop,
            )?;
            let blank = vec![0; image_len(original_size, PixelFormat::Rgb8.channels())?];
            let canvas = new_frame(
                original_size.width,
                original_size.height,
                PixelFormat::Rgb8,
                blank,
                original_image.timing(),
            )?;
            overlay_frame(
                &canvas,
                &resized_generated,
                OverlayPosition::new(
                    usize_to_isize(crop_box.x_min, "crop x")?,
                    usize_to_isize(crop_box.y_min, "crop y")?,
                ),
            )?
        }
        None => {
            validate_frame_size(
                "generated image",
                original_size,
                frame_size(generated_image),
            )?;
            convert_frame_pixel_format(generated_image, PixelFormat::Rgb8)?
        }
    };

    composite_mask_frame(&original_rgb, &generated_canvas, mask)
}

fn checked_pixels(size: ImageSize) -> Result<usize, TransformError> {
    checked_area(size.height, size.width)
}

fn checked_area(height: usize, width: usize) -> Result<usize, TransformError> {
    height
        .checked_mul(width)
        .ok_or(TransformError::ImageSizeOverflow)
}

fn multiple_from_units(units: f64, factor: usize) -> Result<usize, TransformError> {
    if !units.is_finite() || units < 0.0 {
        return Err(TransformError::ImageSizeOverflow);
    }

    let units = units.max(1.0);
    if units > (usize::MAX / factor) as f64 {
        return Err(TransformError::ImageSizeOverflow);
    }

    (units as usize)
        .checked_mul(factor)
        .ok_or(TransformError::ImageSizeOverflow)
}

fn checked_ratio_dimension(
    numerator: usize,
    multiplier: usize,
    divisor: usize,
) -> Result<usize, TransformError> {
    debug_assert!(divisor > 0);
    let value = (numerator as u128) * (multiplier as u128) / (divisor as u128);
    if value > usize::MAX as u128 {
        return Err(TransformError::ImageSizeOverflow);
    }
    Ok(value as usize)
}

fn round_ratio_dimension(
    numerator: usize,
    multiplier: usize,
    divisor: usize,
) -> Result<usize, TransformError> {
    debug_assert!(divisor > 0);
    let value = (numerator as u128)
        .checked_mul(multiplier as u128)
        .and_then(|product| product.checked_mul(2))
        .and_then(|product| product.checked_add(divisor as u128))
        .ok_or(TransformError::ImageSizeOverflow)?
        / ((divisor as u128)
            .checked_mul(2)
            .ok_or(TransformError::ImageSizeOverflow)?);
    if value > usize::MAX as u128 {
        return Err(TransformError::ImageSizeOverflow);
    }
    Ok(value as usize)
}

fn round_odd_up(value: usize) -> Result<usize, TransformError> {
    if value.is_multiple_of(2) {
        Ok(value)
    } else {
        value
            .checked_add(1)
            .ok_or(TransformError::ImageSizeOverflow)
    }
}

fn round_up_to_multiple_transform(value: usize, multiple: usize) -> Result<usize, TransformError> {
    if multiple == 0 {
        return Err(TransformError::InvalidScaleFactor(multiple));
    }
    let remainder = value % multiple;
    if remainder == 0 {
        return Ok(value);
    }
    value
        .checked_add(multiple - remainder)
        .ok_or(TransformError::ImageSizeOverflow)
}

fn scale_size_below_upper_bound(
    size: ImageSize,
    max_edge: usize,
) -> Result<ImageSize, TransformError> {
    if max_edge == 0 {
        return Err(TransformError::InvalidScaleFactor(max_edge));
    }

    if size.width >= size.height && size.width > max_edge {
        Ok(ImageSize {
            height: checked_ratio_dimension(size.height, max_edge, size.width)?.max(1),
            width: max_edge,
        })
    } else if size.height > size.width && size.height > max_edge {
        Ok(ImageSize {
            height: max_edge,
            width: checked_ratio_dimension(size.width, max_edge, size.height)?.max(1),
        })
    } else {
        Ok(size)
    }
}

fn ceil_ratio_dimension(
    numerator: usize,
    multiplier: usize,
    divisor: usize,
) -> Result<usize, TransformError> {
    debug_assert!(divisor > 0);
    let product = (numerator as u128) * (multiplier as u128);
    let divisor = divisor as u128;
    let value = product.div_ceil(divisor);
    if value > usize::MAX as u128 {
        return Err(TransformError::ImageSizeOverflow);
    }
    Ok(value as usize)
}

fn floor_scaled_dimension(value: usize, scale: f64) -> Result<usize, TransformError> {
    if !scale.is_finite() || scale < 0.0 {
        return Err(TransformError::ImageSizeOverflow);
    }
    let scaled = (value as f64) * scale;
    if !scaled.is_finite() || scaled > usize::MAX as f64 {
        return Err(TransformError::ImageSizeOverflow);
    }
    Ok(scaled as usize)
}

fn floor_divided_dimension(value: usize, divisor: f64) -> Result<usize, TransformError> {
    if !divisor.is_finite() || divisor <= 0.0 {
        return Err(TransformError::ImageSizeOverflow);
    }
    let scaled = (value as f64 / divisor).floor();
    if !scaled.is_finite() || scaled > usize::MAX as f64 {
        return Err(TransformError::ImageSizeOverflow);
    }
    Ok(scaled as usize)
}

fn crop_frame_region(
    frame: &ImageFrame,
    origin_x: usize,
    origin_y: usize,
    target: ImageSize,
) -> Result<ImageFrame, TransformError> {
    let source = frame_size(frame);
    let end_x = origin_x
        .checked_add(target.width)
        .ok_or(TransformError::ImageSizeOverflow)?;
    let end_y = origin_y
        .checked_add(target.height)
        .ok_or(TransformError::ImageSizeOverflow)?;
    if end_x > source.width || end_y > source.height {
        return Err(TransformError::CropTooLarge {
            source_size: source,
            target_size: target,
        });
    }

    let data = crop_image_data(
        frame.data(),
        source,
        frame.channels(),
        origin_x,
        origin_y,
        target,
    )?;
    new_frame(
        target.width,
        target.height,
        frame.pixel_format(),
        data,
        frame.timing(),
    )
}

fn validate_size(size: ImageSize) -> Result<(), TransformError> {
    ImageSize::new(size.height, size.width).map(|_| ())
}

fn validate_frame_count(frame_count: usize) -> Result<(), TransformError> {
    if frame_count == 0 {
        Err(TransformError::InvalidFrameCount(frame_count))
    } else {
        Ok(())
    }
}

fn validate_video_size_bucket(bucket: VideoSizeBucket) -> Result<(), TransformError> {
    validate_frame_count(bucket.frame_count)?;
    validate_size(bucket.size)
}

fn validate_image_constraints(constraints: ImageSizeConstraints) -> Result<(), TransformError> {
    if constraints.max_aspect_ratio.is_finite()
        && constraints.max_aspect_ratio >= 1.0
        && constraints.min_side_length > 0
    {
        Ok(())
    } else {
        Err(TransformError::InvalidImageSizeConstraints {
            max_aspect_ratio: constraints.max_aspect_ratio,
            min_side_length: constraints.min_side_length,
        })
    }
}

fn validate_bbox_values(bbox: [f32; 4]) -> Result<(), TransformError> {
    if bbox.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(TransformError::InvalidBoundingBox { bbox })
    }
}

fn validate_point_values(point: [f32; 2]) -> Result<(), TransformError> {
    if point.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(TransformError::InvalidPoint { point })
    }
}

fn validate_scale_factor_f32(value: f32) -> Result<(), TransformError> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        Err(TransformError::InvalidFloatScaleFactor(value))
    }
}

fn validate_overlap_ratio(value: f32) -> Result<(), TransformError> {
    if value.is_finite() && (0.0..1.0).contains(&value) {
        Ok(())
    } else {
        Err(TransformError::InvalidOverlapRatio(value))
    }
}

fn validate_mask_value(value: f32) -> Result<(), TransformError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(TransformError::InvalidMaskValue(value))
    }
}

fn validate_image_value(value: f32) -> Result<(), TransformError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(TransformError::InvalidImageValue(value))
    }
}

fn validate_score_value(value: f32) -> Result<(), TransformError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(TransformError::InvalidScoreValue(value))
    }
}

fn validate_iou_threshold(value: f32) -> Result<(), TransformError> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(TransformError::InvalidIouThreshold(value))
    }
}

fn validate_detection_box_extents(bbox: DetectionBoundingBox) -> Result<(), TransformError> {
    validate_bbox_values([bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max])?;
    if bbox.width() >= 0.0 && bbox.height() >= 0.0 {
        Ok(())
    } else {
        Err(TransformError::InvalidBoundingBox {
            bbox: [bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max],
        })
    }
}

/// Validates that an absolute crop box is non-empty and inside an image.
///
/// # Errors
///
/// Returns an error when `original_size` is invalid, or when `crop_box` is
/// empty or extends outside `original_size`.
pub fn validate_image_crop_box(
    crop_box: ImageCropBox,
    original_size: ImageSize,
) -> Result<(), TransformError> {
    validate_size(original_size)?;
    if crop_box.x_min < crop_box.x_max
        && crop_box.y_min < crop_box.y_max
        && crop_box.x_max <= original_size.width
        && crop_box.y_max <= original_size.height
    {
        Ok(())
    } else {
        Err(TransformError::InvalidImageCropBox {
            crop_box,
            image_size: original_size,
        })
    }
}

fn image_crop_box_size(crop_box: ImageCropBox) -> Result<ImageSize, TransformError> {
    ImageSize::new(
        crop_box.y_max - crop_box.y_min,
        crop_box.x_max - crop_box.x_min,
    )
}

pub(crate) fn resize_f32_image_bilinear(
    values: &[f32],
    source_size: ImageSize,
    target_size: ImageSize,
) -> Result<Vec<f32>, TransformError> {
    resize_f32_bilinear_with(values, source_size, target_size, validate_image_value)
}

pub(crate) fn resize_f32_image_torchvision_bilinear(
    values: &[f32],
    source_size: ImageSize,
    target_size: ImageSize,
) -> Result<Vec<f32>, TransformError> {
    image_resize_kernels::resize_f32_torchvision(
        values,
        KernelImageSize {
            height: source_size.height,
            width: source_size.width,
        },
        1,
        KernelImageSize {
            height: target_size.height,
            width: target_size.width,
        },
        KernelResizeFilter::Bilinear,
    )
    .map_err(|error| TransformError::ResizeKernel {
        message: error.to_string(),
    })
}

/// Resizes and center-crops contiguous floating-point image planes.
///
/// Planes are stored consecutively in row-major order. Resizing uses
/// half-pixel bilinear interpolation and floor-rounded cover geometry, which
/// matches tensor image processors such as Diffusers PixArt.
///
/// # Errors
///
/// Returns an error when `plane_count` is zero, dimensions overflow, the
/// input length differs from `plane_count * source_size.area()`, a value is
/// non-finite, or resize/crop geometry is invalid.
pub fn resize_center_crop_f32_planes(
    values: &[f32],
    plane_count: usize,
    source_size: ImageSize,
    target_size: ImageSize,
) -> Result<Vec<f32>, TransformError> {
    if plane_count == 0 {
        return Err(TransformError::InvalidFrameCount(plane_count));
    }
    let source_pixels = checked_pixels(source_size)?;
    let expected = source_pixels
        .checked_mul(plane_count)
        .ok_or(TransformError::ImageSizeOverflow)?;
    validate_len(values.len(), expected)?;

    let plan = resize_center_crop_plan(source_size, target_size, ResizeRounding::Floor)?;
    let target_pixels = checked_pixels(target_size)?;
    let output_len = target_pixels
        .checked_mul(plane_count)
        .ok_or(TransformError::ImageSizeOverflow)?;
    let mut output = Vec::with_capacity(output_len);

    for plane in values.chunks_exact(source_pixels) {
        let resized = resize_f32_image_bilinear(plane, source_size, plan.resized_size)?;
        for y in plan.crop_box.y_min..plan.crop_box.y_max {
            let row_start = y * plan.resized_size.width + plan.crop_box.x_min;
            let row_end = row_start + target_size.width;
            output.extend_from_slice(&resized[row_start..row_end]);
        }
    }
    Ok(output)
}

fn resize_f32_bilinear(
    values: &[f32],
    source_size: ImageSize,
    target_size: ImageSize,
) -> Result<Vec<f32>, TransformError> {
    resize_f32_bilinear_with(values, source_size, target_size, validate_mask_value)
}

fn resize_f32_bilinear_with(
    values: &[f32],
    source_size: ImageSize,
    target_size: ImageSize,
    validate_value: fn(f32) -> Result<(), TransformError>,
) -> Result<Vec<f32>, TransformError> {
    validate_size(source_size)?;
    validate_size(target_size)?;
    validate_len(values.len(), checked_pixels(source_size)?)?;
    for value in values.iter().copied() {
        validate_value(value)?;
    }
    if source_size == target_size {
        return Ok(values.to_vec());
    }

    let target_pixels = checked_pixels(target_size)?;
    let mut output = Vec::with_capacity(target_pixels);
    for target_y in 0..target_size.height {
        let (y_low, y_high, y_weight) =
            interpolation_indices(target_y, source_size.height, target_size.height);
        for target_x in 0..target_size.width {
            let (x_low, x_high, x_weight) =
                interpolation_indices(target_x, source_size.width, target_size.width);
            let top_left = values[y_low * source_size.width + x_low];
            let top_right = values[y_low * source_size.width + x_high];
            let bottom_left = values[y_high * source_size.width + x_low];
            let bottom_right = values[y_high * source_size.width + x_high];
            let top = lerp(top_left, top_right, x_weight);
            let bottom = lerp(bottom_left, bottom_right, x_weight);
            output.push(lerp(top, bottom, y_weight));
        }
    }
    Ok(output)
}

fn resize_f32_bicubic(
    values: &[f32],
    source_size: ImageSize,
    target_size: ImageSize,
) -> Result<Vec<f32>, TransformError> {
    validate_size(source_size)?;
    validate_size(target_size)?;
    validate_len(values.len(), checked_pixels(source_size)?)?;
    for value in values.iter().copied() {
        validate_mask_value(value)?;
    }
    if source_size == target_size {
        return Ok(values.to_vec());
    }

    let target_pixels = checked_pixels(target_size)?;
    let mut output = Vec::with_capacity(target_pixels);
    for target_y in 0..target_size.height {
        let source_y =
            interpolation_source_position(target_y, source_size.height, target_size.height);
        let y_base = source_y.floor() as isize;
        for target_x in 0..target_size.width {
            let source_x =
                interpolation_source_position(target_x, source_size.width, target_size.width);
            let x_base = source_x.floor() as isize;
            let mut value = 0.0f32;
            for y_offset in -1..=2 {
                let source_y_index =
                    clamp_interpolation_index(y_base + y_offset, source_size.height);
                let y_weight = cubic_weight(source_y - (y_base + y_offset) as f32);
                for x_offset in -1..=2 {
                    let source_x_index =
                        clamp_interpolation_index(x_base + x_offset, source_size.width);
                    let x_weight = cubic_weight(source_x - (x_base + x_offset) as f32);
                    value += values[source_y_index * source_size.width + source_x_index]
                        * y_weight
                        * x_weight;
                }
            }
            validate_mask_value(value)?;
            output.push(value);
        }
    }
    Ok(output)
}

fn interpolation_indices(
    target_index: usize,
    source_extent: usize,
    target_extent: usize,
) -> (usize, usize, f32) {
    let source_position = interpolation_source_position(target_index, source_extent, target_extent);
    let low = source_position.floor() as isize;
    let high = low + 1;
    let weight = source_position - low as f32;
    (
        clamp_interpolation_index(low, source_extent),
        clamp_interpolation_index(high, source_extent),
        weight,
    )
}

fn interpolation_source_position(
    target_index: usize,
    source_extent: usize,
    target_extent: usize,
) -> f32 {
    (target_index as f32 + 0.5) * source_extent as f32 / target_extent as f32 - 0.5
}

fn clamp_interpolation_index(index: isize, extent: usize) -> usize {
    if index < 0 {
        0
    } else {
        (index as usize).min(extent - 1)
    }
}

fn lerp(first: f32, second: f32, weight: f32) -> f32 {
    first.mul_add(1.0 - weight, second * weight)
}

fn cubic_weight(distance: f32) -> f32 {
    const A: f32 = -0.75;
    let x = distance.abs();
    if x <= 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        ((A * x - 5.0 * A) * x + 8.0 * A) * x - 4.0 * A
    } else {
        0.0
    }
}

fn usize_to_isize(value: usize, dimension: &'static str) -> Result<isize, TransformError> {
    isize::try_from(value).map_err(|_| TransformError::DimensionTooLarge {
        dimension,
        value,
        max: isize::MAX as usize,
    })
}

fn ensure_isize_size(size: ImageSize) -> Result<(), TransformError> {
    usize_to_isize(size.width, "width")?;
    usize_to_isize(size.height, "height")?;
    Ok(())
}

fn validate_pixel_chunks(values: &[u8], channels: usize) -> Result<(), TransformError> {
    if channels == 0 {
        return Err(TransformError::UnsupportedChannels(channels));
    }
    if values.len().is_multiple_of(channels) {
        Ok(())
    } else {
        Err(TransformError::InvalidChannelDataLength {
            channels,
            actual: values.len(),
        })
    }
}

fn validate_pixel_fill(fill: &[u8], channels: usize) -> Result<(), TransformError> {
    if channels == 0 {
        return Err(TransformError::UnsupportedChannels(channels));
    }
    if fill.len() == 1 || fill.len() == channels {
        Ok(())
    } else {
        Err(TransformError::InvalidPaddingFill {
            channels,
            actual: fill.len(),
        })
    }
}

fn validate_frame_size(
    frame: &'static str,
    expected: ImageSize,
    actual: ImageSize,
) -> Result<(), TransformError> {
    if actual == expected {
        Ok(())
    } else {
        Err(TransformError::IncompatibleFrameSize {
            frame,
            expected,
            actual,
        })
    }
}

fn validate_pixel_format(
    frame: &'static str,
    expected: PixelFormat,
    actual: PixelFormat,
) -> Result<(), TransformError> {
    if actual == expected {
        Ok(())
    } else {
        Err(TransformError::IncompatiblePixelFormat {
            frame,
            expected,
            actual,
        })
    }
}

fn image_len(size: ImageSize, channels: usize) -> Result<usize, TransformError> {
    size.height
        .checked_mul(size.width)
        .and_then(|pixels| pixels.checked_mul(channels))
        .ok_or(TransformError::ImageSizeOverflow)
}

fn validate_len(actual: usize, expected: usize) -> Result<(), TransformError> {
    if actual == expected {
        Ok(())
    } else {
        Err(TransformError::InvalidBufferLength { expected, actual })
    }
}

/// Errors returned by image transform helpers.
#[derive(Debug, Error, PartialEq)]
pub enum TransformError {
    /// Image dimensions were zero.
    #[error("height and width must be positive, got height={height}, width={width}")]
    InvalidSize {
        /// Image height.
        height: usize,
        /// Image width.
        width: usize,
    },
    /// Resize scale factor was zero.
    #[error("scale factor must be positive, got {0}")]
    InvalidScaleFactor(usize),
    /// Frame count was zero.
    #[error("frame count must be positive, got {0}")]
    InvalidFrameCount(usize),
    /// Temporal frame multiple was zero.
    #[error("temporal frame multiple must be positive, got {0}")]
    InvalidTemporalMultiple(usize),
    /// Target pixel area was zero.
    #[error("target area must be positive, got {0}")]
    InvalidTargetArea(usize),
    /// Source aspect ratio exceeded supported limits.
    #[error("aspect ratio must be smaller than 200, got {0}")]
    InvalidAspectRatio(f64),
    /// Image size constraints were invalid.
    #[error(
        "image size constraints must be valid, got max_aspect_ratio={max_aspect_ratio}, min_side_length={min_side_length}"
    )]
    InvalidImageSizeConstraints {
        /// Maximum allowed aspect ratio.
        max_aspect_ratio: f64,
        /// Minimum allowed side length.
        min_side_length: usize,
    },
    /// Image dimensions were smaller than a processor-specific minimum.
    #[error("image size {size:?} is smaller than minimum side length {min_side_length}")]
    ImageTooSmall {
        /// Actual image size.
        size: ImageSize,
        /// Minimum allowed side length.
        min_side_length: usize,
    },
    /// Source aspect ratio exceeded a processor-specific limit.
    #[error("aspect ratio {aspect_ratio} exceeds maximum allowed ratio {max_aspect_ratio}")]
    AspectRatioTooLarge {
        /// Actual aspect ratio.
        aspect_ratio: f64,
        /// Maximum allowed aspect ratio.
        max_aspect_ratio: f64,
    },
    /// Aspect-ratio crop activation ratio was not finite and positive.
    #[error("aspect-ratio crop activation ratio must be finite and positive, got {0}")]
    InvalidAspectRatioCropActivationRatio(f64),
    /// Floating-point scale factor was negative or non-finite.
    #[error("floating-point scale factor must be finite and non-negative, got {0}")]
    InvalidFloatScaleFactor(f32),
    /// Bounding box values were non-finite.
    #[error("bounding box values must be finite, got {bbox:?}")]
    InvalidBoundingBox {
        /// Bounding box values.
        bbox: [f32; 4],
    },
    /// Point coordinates were non-finite.
    #[error("point coordinates must be finite, got {point:?}")]
    InvalidPoint {
        /// Point values.
        point: [f32; 2],
    },
    /// Crop overlap ratio was outside `[0, 1)`.
    #[error("crop overlap ratio must be finite and in [0, 1), got {0}")]
    InvalidOverlapRatio(f32),
    /// Layered crop box was empty or outside the original image.
    #[error("layered crop box {crop_box:?} is empty or outside image size {image_size:?}")]
    InvalidCropBox {
        /// Crop box.
        crop_box: LayeredCropBox,
        /// Original image size.
        image_size: ImageSize,
    },
    /// Image crop box was empty or outside the original image.
    #[error("image crop box {crop_box:?} is empty or outside image size {image_size:?}")]
    InvalidImageCropBox {
        /// Crop box.
        crop_box: ImageCropBox,
        /// Original image size.
        image_size: ImageSize,
    },
    /// Mask value or threshold was non-finite.
    #[error("mask values and thresholds must be finite, got {0}")]
    InvalidMaskValue(f32),
    /// Image value was non-finite.
    #[error("image values must be finite, got {0}")]
    InvalidImageValue(f32),
    /// Detection score or logit was non-finite.
    #[error("scores and logits must be finite, got {0}")]
    InvalidScoreValue(f32),
    /// IoU threshold was outside `[0, 1]`.
    #[error("IoU threshold must be finite and in [0, 1], got {0}")]
    InvalidIouThreshold(f32),
    /// Class count did not include at least one foreground class and background.
    #[error("class count must include foreground labels and background, got {0}")]
    InvalidClassCount(usize),
    /// Annotation area was negative or non-finite.
    #[error("annotation area must be finite and non-negative, got {0}")]
    InvalidAnnotationArea(f32),
    /// Detection annotation field lengths were inconsistent.
    #[error(
        "detection annotation lengths differ: class_labels={class_labels}, boxes={boxes}, area={area}, iscrowd={iscrowd}"
    )]
    InconsistentDetectionAnnotationLengths {
        /// Number of class labels.
        class_labels: usize,
        /// Number of boxes.
        boxes: usize,
        /// Number of area values.
        area: usize,
        /// Number of crowd flags.
        iscrowd: usize,
    },
    /// Smart resize parameters were invalid or impossible to satisfy.
    #[error("resize factor and pixel limits are invalid or unachievable")]
    InvalidSmartResizeParams,
    /// A tile grid was empty or exceeded the maximum tile count.
    #[error("tile grid rows={rows}, columns={columns} exceeds max_image_tiles={max_image_tiles}")]
    InvalidTileGrid {
        /// Number of tile rows.
        rows: usize,
        /// Number of tile columns.
        columns: usize,
        /// Maximum allowed tile count.
        max_image_tiles: usize,
    },
    /// No candidate resolutions were provided.
    #[error("resolution candidates cannot be empty")]
    EmptyResolutionCandidates,
    /// No image sizes were provided for batch planning.
    #[error("image batch cannot be empty")]
    EmptyImageBatch,
    /// No rows or columns were provided for nested image-grid metadata.
    #[error("nested image grid cannot be empty")]
    EmptyNestedImageGrid,
    /// A nested image-grid field had the wrong number of rows.
    #[error("nested image grid field `{field}` has {actual} rows, expected {expected}")]
    InconsistentNestedImageGridRows {
        /// Metadata field name.
        field: &'static str,
        /// Expected row count.
        expected: usize,
        /// Actual row count.
        actual: usize,
    },
    /// A nested image-grid row had the wrong number of columns.
    #[error(
        "nested image grid field `{field}` row {row} has {actual} columns, expected {expected}"
    )]
    InconsistentNestedImageGridColumns {
        /// Metadata field name.
        field: &'static str,
        /// Row index.
        row: usize,
        /// Expected column count.
        expected: usize,
        /// Actual column count.
        actual: usize,
    },
    /// No frames were provided for temporal batch planning.
    #[error("frame batch cannot be empty")]
    EmptyFrameBatch,
    /// No mask planes were provided for mask processing.
    #[error("mask batch cannot be empty")]
    EmptyMaskBatch,
    /// No patch plans were provided for batch planning.
    #[error("patch batch cannot be empty")]
    EmptyPatchBatch,
    /// Attention dimensions were not all positive.
    #[error(
        "attention shape must be positive, got batch_size={batch_size}, num_queries={num_queries}, value_embed_dim={value_embed_dim}"
    )]
    InvalidAttentionShape {
        /// Requested batch size.
        batch_size: usize,
        /// Requested attention query count.
        num_queries: usize,
        /// Requested value embedding dimension.
        value_embed_dim: usize,
    },
    /// Image buffer length did not match dimensions and channels.
    #[error("invalid image buffer length: expected {expected} bytes, got {actual}")]
    InvalidBufferLength {
        /// Expected byte length.
        expected: usize,
        /// Actual byte length.
        actual: usize,
    },
    /// Pixel data length was not divisible by the channel count.
    #[error("pixel data length {actual} is not divisible by channel count {channels}")]
    InvalidChannelDataLength {
        /// Channel count.
        channels: usize,
        /// Actual byte length.
        actual: usize,
    },
    /// Image dimensions or buffer length overflowed `usize`.
    #[error("image size overflows usize")]
    ImageSizeOverflow,
    /// A center crop target exceeded the source image size.
    #[error("crop target {target_size:?} exceeds source image size {source_size:?}")]
    CropTooLarge {
        /// Source image dimensions.
        source_size: ImageSize,
        /// Requested crop dimensions.
        target_size: ImageSize,
    },
    /// A dimension exceeded a backend-supported maximum.
    #[error("{dimension} dimension {value} exceeds maximum supported value {max}")]
    DimensionTooLarge {
        /// Dimension name.
        dimension: &'static str,
        /// Actual dimension value.
        value: usize,
        /// Maximum supported value.
        max: usize,
    },
    /// A resize parity policy was incompatible with the selected filter.
    #[error("resize parity {parity:?} is incompatible with filter {filter:?}")]
    InvalidResizeParity {
        /// Requested resize filter.
        filter: ResizeFilter,
        /// Requested parity policy.
        parity: ResizeParity,
    },
    /// The resize kernel backend failed.
    #[error("resize kernel backend failed: {message}")]
    ResizeKernel {
        /// Backend error message.
        message: String,
    },
    /// An internal image frame invariant failed.
    #[error("internal image frame invariant failed: {0}")]
    ImageFrameInvariant(String),
    /// The channel count is unsupported.
    #[error("unsupported channel count {0}")]
    UnsupportedChannels(usize),
    /// Padding fill length was not scalar or equal to channel count.
    #[error("padding fill length must be 1 or channel count {channels}, got {actual}")]
    InvalidPaddingFill {
        /// Frame channel count.
        channels: usize,
        /// Actual fill byte length.
        actual: usize,
    },
    /// A secondary frame did not match the expected dimensions.
    #[error("{frame} frame size {actual:?} does not match expected {expected:?}")]
    IncompatibleFrameSize {
        /// Frame role.
        frame: &'static str,
        /// Expected frame dimensions.
        expected: ImageSize,
        /// Actual frame dimensions.
        actual: ImageSize,
    },
    /// A secondary frame did not match the expected pixel format.
    #[error("{frame} pixel format {actual:?} does not match expected {expected:?}")]
    IncompatiblePixelFormat {
        /// Frame role.
        frame: &'static str,
        /// Expected pixel format.
        expected: PixelFormat,
        /// Actual pixel format.
        actual: PixelFormat,
    },
}

#[cfg(test)]
mod tests;
