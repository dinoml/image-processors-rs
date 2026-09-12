use super::*;
use serde::{Deserialize, Serialize};

/// A preprocessing stage in a processor recipe.
#[non_exhaustive]
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "stage", rename_all = "snake_case")]
pub enum ProcessorRecipeStage {
    /// Convert decoded frames to the requested pixel format.
    ConvertPixelFormat {
        /// Target pixel format.
        format: PixelFormat,
    },
    /// Resize decoded frames.
    Resize {
        /// Resize stage parameters.
        resize: RecipeResizeStage,
    },
    /// Apply Owlv2's rescale-before-pad-and-antialiased-resize pipeline.
    Owlv2AntialiasedResize {
        /// Fixed output dimensions.
        size: ImageSize,
        /// Whether to bottom/right-pad the input to a square before resizing.
        pad_to_square: bool,
        /// Whether to rescale byte values before interpolation.
        do_rescale: bool,
        /// Pre-interpolation rescale factor.
        rescale_factor: f32,
    },
    /// Apply VitPose's bounding-box-driven affine resize.
    VitPoseAffine {
        /// Fixed affine output dimensions.
        size: ImageSize,
        /// Bounding-box scale normalization factor.
        normalize_factor: f32,
        /// Context padding factor applied around each box.
        padding_factor: f32,
    },
    /// Multiply pixel values by a scalar factor.
    Rescale {
        /// Scalar rescale factor.
        factor: f32,
    },
    /// Apply per-channel normalization.
    Normalize {
        /// Per-channel or scalar mean values.
        mean: Vec<f32>,
        /// Per-channel or scalar standard deviation values.
        std: Vec<f32>,
    },
    /// Threshold values into `0.0` or `1.0`.
    Binarize,
    /// Flatten image tensors into patch rows.
    PatchFlatten {
        /// Patch-flattening parameters.
        patch: RecipePatchStage,
    },
    /// Select frames from a decoded image sequence or video clip.
    SampleFrames {
        /// Frame sampling policy.
        sampling: FrameSampling,
    },
    /// Repeat the final frame until the frame batch length is a multiple.
    TemporalRepeatLast {
        /// Required temporal multiple.
        multiple: usize,
    },
    /// Split one decoded frame into a resized base frame plus patch-grid frames.
    PatchGrid {
        /// Patch-grid extraction parameters.
        patch_grid: RecipePatchGridStage,
    },
    /// Split large images into local frames plus an optional global frame.
    ImageSplit {
        /// Split-image geometry parameters.
        split: RecipeImageSplitStage,
    },
    /// Emit aspect-ratio-driven crops for elongated images.
    AspectRatioCrops {
        /// Aspect-ratio crop-selection parameters.
        aspect_ratio_crops: RecipeAspectRatioCropStage,
    },
    /// Resize, pad, and split images into a tiled canvas.
    TiledCanvas {
        /// Tiled-canvas geometry parameters.
        tile: RecipeTiledCanvasStage,
    },
    /// Apply Donut/document OCR resize, thumbnail, rotation, and padding geometry.
    DocumentGeometry {
        /// Document/OCR geometry parameters.
        document: RecipeDocumentGeometryStage,
    },
    /// Validate decoded image dimensions against minimum-side and aspect-ratio limits.
    ValidateImageSize {
        /// Validation constraints.
        constraints: ImageSizeConstraints,
    },
    /// Select the closest aspect-ratio bucket from a candidate list.
    SelectAspectRatioBucket {
        /// Candidate bucket sizes.
        candidates: Vec<ImageSize>,
    },
    /// Select the closest frame-count and spatial bucket for decoded video.
    SelectVideoSizeBucket {
        /// Candidate video buckets.
        candidates: Vec<VideoSizeBucket>,
    },
    /// Resize only when an image exceeds a maximum pixel area.
    AreaResize {
        /// Maximum pixel area.
        target_area: usize,
        /// Resampling filter.
        filter: ResizeFilter,
    },
    /// Crop decoded frames without resizing.
    Crop {
        /// Crop stage parameters.
        crop: RecipeCropStage,
    },
    /// Resize to cover a target size and crop the centered output.
    ResizeCenterCrop {
        /// Output size after center crop.
        size: ImageSize,
        /// Intermediate resize rounding.
        rounding: ResizeRounding,
        /// Resampling filter.
        filter: ResizeFilter,
    },
    /// Pad decoded frames with a constant pixel value.
    Pad {
        /// Padding stage parameters.
        pad: RecipePadStage,
    },
    /// Pad decoded frames with a canvas fill behavior.
    PadCanvas {
        /// Padding to apply in top-right-bottom-left order.
        padding: Padding,
        /// Canvas fill behavior.
        fill: CanvasFill,
    },
    /// Pad decoded frames on the bottom/right to per-axis multiples.
    PadToMultiple {
        /// Required height and width multiples.
        multiples: ImageSize,
        /// Canvas fill behavior.
        fill: CanvasFill,
    },
    /// Pad bottom/right to the next per-axis multiples with edge-inclusive
    /// symmetric reflection.
    PadSymmetricToNextMultiple {
        /// Required height and width multiples. Dimensions already divisible
        /// by these values still advance by one full multiple.
        multiples: ImageSize,
    },
    /// Overlay a secondary frame onto the current frame.
    Overlay {
        /// Overlay stage parameters.
        overlay: RecipeOverlayStage,
    },
    /// Composite a foreground frame over a background frame using a mask.
    MaskComposite,
    /// Round dimensions down to per-axis multiples.
    RoundToMultiple {
        /// Optional requested size before rounding.
        requested_size: Option<ImageSize>,
        /// Required height and width multiples.
        multiples: ImageSize,
    },
    /// Resize to fit on a canvas using the current pixel format.
    ResizeFill {
        /// Output canvas size.
        size: ImageSize,
        /// Canvas fill behavior.
        fill: CanvasFill,
        /// Resampling filter.
        filter: ResizeFilter,
    },
    /// Resize to fit on an RGB canvas.
    ResizeFillRgb {
        /// Output canvas size.
        size: ImageSize,
        /// Canvas fill behavior.
        fill: RgbCanvasFill,
        /// Resampling filter.
        filter: ResizeFilter,
    },
    /// Concatenate a batch of RGB-converted frames horizontally.
    ConcatenateHorizontallyRgb {
        /// Canvas fill color.
        fill: [u8; 3],
    },
}

impl ProcessorRecipeStage {
    pub(super) fn kind(&self) -> &'static str {
        match self {
            Self::ConvertPixelFormat { .. } => "convert_pixel_format",
            Self::Resize { .. } => "resize",
            Self::Owlv2AntialiasedResize { .. } => "owlv2_antialiased_resize",
            Self::VitPoseAffine { .. } => "vit_pose_affine",
            Self::Rescale { .. } => "rescale",
            Self::Normalize { .. } => "normalize",
            Self::Binarize => "binarize",
            Self::PatchFlatten { .. } => "patch_flatten",
            Self::SampleFrames { .. } => "sample_frames",
            Self::TemporalRepeatLast { .. } => "temporal_repeat_last",
            Self::PatchGrid { .. } => "patch_grid",
            Self::ImageSplit { .. } => "image_split",
            Self::AspectRatioCrops { .. } => "aspect_ratio_crops",
            Self::TiledCanvas { .. } => "tiled_canvas",
            Self::DocumentGeometry { .. } => "document_geometry",
            Self::ValidateImageSize { .. } => "validate_image_size",
            Self::SelectAspectRatioBucket { .. } => "select_aspect_ratio_bucket",
            Self::SelectVideoSizeBucket { .. } => "select_video_size_bucket",
            Self::AreaResize { .. } => "area_resize",
            Self::Crop { .. } => "crop",
            Self::ResizeCenterCrop { .. } => "resize_center_crop",
            Self::Pad { .. } => "pad",
            Self::PadCanvas { .. } => "pad_canvas",
            Self::PadToMultiple { .. } => "pad_to_multiple",
            Self::PadSymmetricToNextMultiple { .. } => "pad_symmetric_to_next_multiple",
            Self::Overlay { .. } => "overlay",
            Self::MaskComposite => "mask_composite",
            Self::RoundToMultiple { .. } => "round_to_multiple",
            Self::ResizeFill { .. } => "resize_fill",
            Self::ResizeFillRgb { .. } => "resize_fill_rgb",
            Self::ConcatenateHorizontallyRgb { .. } => "concatenate_horizontally_rgb",
        }
    }
}

/// Crop parameters for a processor recipe.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RecipeCropStage {
    /// Center crop with nearest-even offsets, after symmetric zero padding if needed.
    CenterTiesEven {
        /// Output crop size.
        size: ImageSize,
    },
    /// Crop the center window from the current image.
    Center {
        /// Output crop size.
        size: ImageSize,
    },
    /// Crop an absolute pixel box from the current image.
    Absolute {
        /// Absolute crop box in corner coordinates.
        crop_box: ImageCropBox,
    },
}

impl RecipeCropStage {
    /// Creates a center-crop stage.
    pub fn center(size: ImageSize) -> Self {
        Self::Center { size }
    }

    /// Creates an absolute crop stage.
    pub fn absolute(crop_box: ImageCropBox) -> Self {
        Self::Absolute { crop_box }
    }

    pub(super) fn kind(self) -> &'static str {
        match self {
            Self::Center { .. } => "center_crop",
            Self::CenterTiesEven { .. } => "center_crop_ties_even",
            Self::Absolute { .. } => "absolute_crop",
        }
    }
}

/// Constant padding parameters for a processor recipe.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RecipePadStage {
    /// Padding to apply in top-right-bottom-left order.
    pub padding: Padding,
    /// Fill bytes, either scalar or one byte per channel after pixel conversion.
    pub fill: Vec<u8>,
}

impl RecipePadStage {
    /// Creates a constant-padding stage.
    pub fn constant(padding: Padding, fill: impl Into<Vec<u8>>) -> Self {
        Self {
            padding,
            fill: fill.into(),
        }
    }
}

/// Overlay parameters for a processor recipe.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RecipeOverlayStage {
    /// Top-left foreground position on the background frame.
    pub position: OverlayPosition,
}

impl RecipeOverlayStage {
    /// Creates an overlay stage from a top-left foreground position.
    pub fn new(position: OverlayPosition) -> Self {
        Self { position }
    }
}

/// Patch-grid extraction parameters for a processor recipe.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RecipePatchGridStage {
    /// Candidate high-resolution canvas sizes.
    pub candidate_sizes: Vec<ImageSize>,
    /// Base-frame resize target emitted before high-resolution patches.
    pub base_size: ImageSize,
    /// Square patch edge used when splitting the selected canvas.
    pub patch_size: usize,
}

impl RecipePatchGridStage {
    /// Creates a patch-grid extraction stage.
    pub fn new(candidate_sizes: Vec<ImageSize>, base_size: ImageSize, patch_size: usize) -> Self {
        Self {
            candidate_sizes,
            base_size,
            patch_size,
        }
    }
}

/// Split-image geometry parameters for a processor recipe.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RecipeImageSplitStage {
    /// Longest-edge resize target used before split-frame extraction.
    pub longest_edge: usize,
    /// Maximum square frame edge used for local and global frames.
    pub max_image_size: usize,
    /// Whether a long-edge resize is applied before vision-encoder alignment.
    pub do_resize: bool,
}

impl RecipeImageSplitStage {
    /// Creates an image split stage.
    pub fn new(longest_edge: usize, max_image_size: usize, do_resize: bool) -> Self {
        Self {
            longest_edge,
            max_image_size,
            do_resize,
        }
    }
}

/// Aspect-ratio crop-selection parameters for a processor recipe.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct RecipeAspectRatioCropStage {
    /// Crop-selection options.
    pub options: AspectRatioCropOptions,
}

impl RecipeAspectRatioCropStage {
    /// Creates an aspect-ratio crop stage.
    pub fn new(options: AspectRatioCropOptions) -> Self {
        Self { options }
    }
}

/// Tiled-canvas geometry parameters for a processor recipe.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RecipeTiledCanvasStage {
    /// Square tile edge length.
    pub tile_size: usize,
    /// Maximum number of tiles allowed for one image.
    pub max_image_tiles: usize,
}

impl RecipeTiledCanvasStage {
    /// Creates a tiled-canvas stage.
    pub fn new(tile_size: usize, max_image_tiles: usize) -> Self {
        Self {
            tile_size,
            max_image_tiles,
        }
    }
}

/// Donut/document OCR geometry parameters for a processor recipe.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RecipeDocumentGeometryStage {
    /// Target document canvas size.
    pub target_size: ImageSize,
    /// Whether to apply the initial shortest-edge resize.
    pub do_resize: bool,
    /// Whether to constrain resized images to the target canvas.
    pub do_thumbnail: bool,
    /// Whether to rotate images whose long axis differs from the target canvas.
    pub do_align_long_axis: bool,
    /// Whether to center-pad images to the target canvas.
    pub do_pad: bool,
}

impl RecipeDocumentGeometryStage {
    /// Creates a document/OCR geometry stage.
    pub fn new(
        target_size: ImageSize,
        do_resize: bool,
        do_thumbnail: bool,
        do_align_long_axis: bool,
        do_pad: bool,
    ) -> Self {
        Self {
            target_size,
            do_resize,
            do_thumbnail,
            do_align_long_axis,
            do_pad,
        }
    }
}

/// Resize stage parameters for a processor recipe.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RecipeResizeStage {
    /// Target dimensions or dynamic per-call target behavior.
    pub target: RecipeResizeTarget,
    /// Resize sizing mode.
    pub mode: ResizeMode,
    /// Resampling filter.
    pub filter: ResizeFilter,
    /// Resize parity policy.
    #[serde(default)]
    pub parity: ResizeParity,
}

impl RecipeResizeStage {
    /// Creates a fixed-size resize stage.
    pub fn fixed(
        size: ImageSize,
        mode: ResizeMode,
        filter: ResizeFilter,
        parity: ResizeParity,
    ) -> Self {
        Self {
            target: RecipeResizeTarget::Fixed { size },
            mode,
            filter,
            parity,
        }
    }

    /// Creates a dynamic resize stage controlled by per-call options.
    pub fn dynamic(mode: ResizeMode, filter: ResizeFilter, parity: ResizeParity) -> Self {
        Self {
            target: RecipeResizeTarget::Dynamic,
            mode,
            filter,
            parity,
        }
    }

    /// Creates a smart-resize stage controlled by pixel-count limits.
    pub fn smart(
        limits: ResizeLimits,
        mode: ResizeMode,
        filter: ResizeFilter,
        parity: ResizeParity,
    ) -> Self {
        Self {
            target: RecipeResizeTarget::SmartResize { limits },
            mode,
            filter,
            parity,
        }
    }

    /// Creates an aspect-preserving shortest-edge resize stage.
    pub fn shortest_edge(
        shortest_edge: usize,
        longest_edge: Option<usize>,
        filter: ResizeFilter,
        parity: ResizeParity,
    ) -> Self {
        Self {
            target: RecipeResizeTarget::ShortestEdge {
                shortest_edge,
                longest_edge,
            },
            mode: ResizeMode::Default,
            filter,
            parity,
        }
    }

    /// Creates a single resize after rounding both resolved axes down to a divisor.
    /// Uses the unrounded capped aspect scale before applying the divisor.
    pub fn shortest_edge_round_down(
        shortest_edge: usize,
        longest_edge: Option<usize>,
        multiple: usize,
        filter: ResizeFilter,
        parity: ResizeParity,
    ) -> Self {
        Self {
            target: RecipeResizeTarget::ShortestEdgeRoundDown {
                shortest_edge,
                longest_edge,
                multiple,
            },
            mode: ResizeMode::Default,
            filter,
            parity,
        }
    }

    /// Creates an aspect-preserving longest-edge resize stage.
    pub fn longest_edge(longest_edge: usize, filter: ResizeFilter, parity: ResizeParity) -> Self {
        Self {
            target: RecipeResizeTarget::LongestEdge { longest_edge },
            mode: ResizeMode::Default,
            filter,
            parity,
        }
    }
}

/// Resize target selection for a processor recipe.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RecipeResizeTarget {
    /// Resize to a fixed height and width.
    Fixed {
        /// Fixed resize dimensions.
        size: ImageSize,
    },
    /// Resolve resize size from per-call options or input dimensions.
    Dynamic,
    /// Resolve resize size through smart pixel-count limits.
    SmartResize {
        /// Smart resize limits.
        limits: ResizeLimits,
    },
    /// Resize so the shortest image edge reaches `shortest_edge`.
    ShortestEdge {
        /// Target size for the shorter image edge.
        shortest_edge: usize,
        /// Optional maximum size for the longer image edge.
        #[serde(default)]
        longest_edge: Option<usize>,
    },
    /// Preserve the capped aspect scale, then round both axes down before one resample.
    ShortestEdgeRoundDown {
        /// Target size for the shorter image edge.
        shortest_edge: usize,
        /// Optional maximum size for the longer image edge.
        #[serde(default)]
        longest_edge: Option<usize>,
        /// Positive divisor for both final dimensions.
        multiple: usize,
    },
    /// Resize so the longest image edge reaches `longest_edge`.
    LongestEdge {
        /// Target size for the longer image edge.
        longest_edge: usize,
    },
}

impl RecipeResizeTarget {
    pub(super) fn kind(self) -> &'static str {
        match self {
            Self::Fixed { .. } => "resize_fixed",
            Self::Dynamic => "resize_dynamic",
            Self::SmartResize { .. } => "resize_smart",
            Self::ShortestEdge { .. } => "resize_shortest_edge",
            Self::ShortestEdgeRoundDown { .. } => "resize_shortest_edge_round_down",
            Self::LongestEdge { .. } => "resize_longest_edge",
        }
    }
}
