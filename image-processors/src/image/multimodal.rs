//! Thumbnail-first image tiling and uniformly sampled decoded video frames.

use super::*;
use crate::media::FrameSampling;
use crate::transforms::thumbnail_tiled_canvas_grid;

impl ImageProcessor {
    /// Resizes a thumbnail followed by row-major tiles of a stretched canvas.
    ///
    /// Uses the PerceptionLM canvas ordering/tie policy. Resize, pixel conversion,
    /// rescaling and normalization use this processor's existing configuration.
    /// The FP32 NCHW result is `[1 + tiles, channels, tile_size, tile_size]`.
    /// With resizing disabled, returns the ordinary unmodified-size image path.
    ///
    /// # Errors
    ///
    /// Rejects non-NCHW/non-FP32 results, invalid geometry or shared processor
    /// failures. No decoded-frame or quantized intermediate copies are created.
    pub fn preprocess_thumbnail_tiles(
        &self,
        image: &ImageFrame,
        tile_size: usize,
        max_image_tiles: usize,
    ) -> Result<Tensor, ImageProcessorError> {
        if self.config.output_layout != Layout::NCHW {
            return Err(ImageProcessorError::UnsupportedLayout(
                self.config.output_layout,
            ));
        }
        if !self.config.do_resize {
            return self.preprocess_image(image);
        }
        let grid = thumbnail_tiled_canvas_grid(
            ImageSize::new(image.height(), image.width())?,
            tile_size,
            max_image_tiles,
        )?;
        let height = grid
            .rows
            .checked_mul(tile_size)
            .ok_or(TransformError::ImageSizeOverflow)?;
        let width = grid
            .columns
            .checked_mul(tile_size)
            .ok_or(TransformError::ImageSizeOverflow)?;
        let thumbnail = self.preprocess_image_with_options(
            image,
            ImageProcessorOptions {
                height: Some(tile_size),
                width: Some(tile_size),
                resize_mode: Some(ResizeMode::Default),
            },
        )?;
        let canvas = self.preprocess_image_with_options(
            image,
            ImageProcessorOptions {
                height: Some(height),
                width: Some(width),
                resize_mode: Some(ResizeMode::Default),
            },
        )?;
        let channels = *thumbnail
            .shape()
            .get(1)
            .ok_or(ImageProcessorError::IncompatibleBatchShapes)?;
        if thumbnail.shape() != [1, channels, tile_size, tile_size]
            || canvas.shape() != [1, channels, height, width]
        {
            return Err(ImageProcessorError::IncompatibleBatchShapes);
        }
        let TensorData::F32(canvas) = canvas.data() else {
            return Err(ImageProcessorError::ExpectedDataType("F32"));
        };
        let (data, _, _, _) = thumbnail.into_parts();
        let TensorData::F32(mut values) = data else {
            return Err(ImageProcessorError::ExpectedDataType("F32"));
        };
        let count = grid
            .tile_count()?
            .checked_add(1)
            .ok_or(TransformError::ImageSizeOverflow)?;
        let total = count
            .checked_mul(channels)
            .and_then(|size| size.checked_mul(tile_size))
            .and_then(|size| size.checked_mul(tile_size))
            .ok_or(TransformError::ImageSizeOverflow)?;
        values.reserve(total - values.len());
        for row in 0..grid.rows {
            for column in 0..grid.columns {
                for channel in 0..channels {
                    for y in 0..tile_size {
                        let start = channel * height * width
                            + (row * tile_size + y) * width
                            + column * tile_size;
                        values.extend_from_slice(&canvas[start..start + tile_size]);
                    }
                }
            }
        }
        Ok(Tensor::new(
            TensorData::F32(values),
            [count, channels, tile_size, tile_size],
            Layout::NCHW,
        )?)
    }

    /// Uniformly samples decoded video frames and preprocesses them as a batch.
    ///
    /// Sampling follows [`FrameSampling::uniform_indices`], including FPS-count
    /// truncation and floor-rounded frame indices. Frames are borrowed, and
    /// tensor stacking reuses the existing image preprocessing owner.
    ///
    /// # Errors
    ///
    /// Propagates sampling, image processing and incompatible frame-shape errors.
    pub fn preprocess_uniform_video(
        &self,
        frames: &[ImageFrame],
        num_frames: Option<usize>,
        target_fps: Option<f64>,
        source_fps: Option<f64>,
    ) -> Result<Tensor, ImageProcessorError> {
        let indices =
            FrameSampling::uniform_indices(frames.len(), num_frames, target_fps, source_fps)?;
        let tensors = indices
            .into_iter()
            .map(|index| self.preprocess_image(&frames[index]))
            .collect::<Result<Vec<_>, _>>()?;
        stack_tensors(tensors, TensorLeadingAxis::Frames)
    }
}
