//! Tiled-canvas geometry helpers.

use super::*;

/// Tiled-canvas grid dimensions in row-column order.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TiledCanvasGrid {
    /// Number of tile rows.
    pub rows: usize,
    /// Number of tile columns.
    pub columns: usize,
}

impl TiledCanvasGrid {
    /// Creates tile-grid dimensions with positive row and column counts.
    ///
    /// # Errors
    ///
    /// Returns an error when either dimension is zero.
    pub fn new(rows: usize, columns: usize) -> Result<Self, TransformError> {
        if rows == 0 || columns == 0 {
            return Err(TransformError::InvalidTileGrid {
                rows,
                columns,
                max_image_tiles: 0,
            });
        }
        Ok(Self { rows, columns })
    }

    /// Returns the number of tiles in the grid.
    ///
    /// # Errors
    ///
    /// Returns an error if tile-count arithmetic overflows.
    pub fn tile_count(self) -> Result<usize, TransformError> {
        self.rows
            .checked_mul(self.columns)
            .ok_or(TransformError::ImageSizeOverflow)
    }
}

/// Tiled-canvas resize plan for one image.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TiledCanvasPlan {
    /// Original image dimensions.
    pub original_size: ImageSize,
    /// Square tile edge length.
    pub tile_size: usize,
    /// Maximum number of tiles allowed for one image.
    pub max_image_tiles: usize,
    /// Selected tile grid.
    pub grid: TiledCanvasGrid,
    /// Selected padded canvas dimensions.
    pub canvas_size: ImageSize,
    /// Aspect-preserving resized dimensions before padding.
    pub resized_size: ImageSize,
    /// Right/bottom padding that expands `resized_size` to `canvas_size`.
    pub padding: Padding,
    /// One-based aspect-ratio id for `grid`.
    pub aspect_ratio_id: usize,
}

/// Nested tiled-canvas batch metadata.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TiledCanvasBatchMetadata {
    /// Maximum number of image slots in any batch sample.
    pub max_images_per_sample: usize,
    /// Actual per-image tile counts, unpadded per sample.
    pub num_tiles: Vec<Vec<usize>>,
    /// Padded one-based aspect-ratio ids, with zero for padded image slots.
    pub aspect_ratio_ids: Vec<Vec<usize>>,
    /// Padded tile-validity mask in batch-sample-image-tile order.
    pub aspect_ratio_mask: Vec<Vec<Vec<bool>>>,
}

/// Lists supported tiled-canvas grids in aspect-ratio id order.
///
/// The returned grids use row-column order. The first grid has id `1`, while
/// id `0` is reserved for padded image slots in batched metadata.
///
/// # Errors
///
/// Returns an error when `max_image_tiles` is zero.
pub fn supported_tiled_canvas_grids(
    max_image_tiles: usize,
) -> Result<Vec<TiledCanvasGrid>, TransformError> {
    validate_max_image_tiles(max_image_tiles)?;
    let mut grids = Vec::with_capacity(max_image_tiles);
    for rows in 1..=max_image_tiles {
        for columns in 1..=max_image_tiles {
            let count = rows
                .checked_mul(columns)
                .ok_or(TransformError::ImageSizeOverflow)?;
            if count <= max_image_tiles {
                grids.push(TiledCanvasGrid { rows, columns });
            }
        }
    }
    Ok(grids)
}

/// Returns the one-based aspect-ratio id for a tile grid.
///
/// # Errors
///
/// Returns an error when `max_image_tiles` is zero, the grid is empty, or the
/// grid uses more than `max_image_tiles` tiles.
pub fn tiled_canvas_aspect_ratio_id(
    grid: TiledCanvasGrid,
    max_image_tiles: usize,
) -> Result<usize, TransformError> {
    validate_tiled_canvas_grid(grid, max_image_tiles)?;
    let grids = supported_tiled_canvas_grids(max_image_tiles)?;
    grids
        .iter()
        .position(|candidate| *candidate == grid)
        .map(|index| index + 1)
        .ok_or(TransformError::InvalidTileGrid {
            rows: grid.rows,
            columns: grid.columns,
            max_image_tiles,
        })
}

/// Builds the aspect-ratio mask for one tiled-canvas image.
///
/// The mask has length `max_image_tiles` and marks valid tile positions for
/// `grid`; remaining entries are padding.
///
/// # Errors
///
/// Returns an error when the grid is invalid for `max_image_tiles`.
pub fn tiled_canvas_aspect_ratio_mask(
    grid: TiledCanvasGrid,
    max_image_tiles: usize,
) -> Result<Vec<bool>, TransformError> {
    validate_tiled_canvas_grid(grid, max_image_tiles)?;
    let tile_count = grid.tile_count()?;
    let mut mask = vec![false; max_image_tiles];
    mask.iter_mut()
        .take(tile_count)
        .for_each(|value| *value = true);
    Ok(mask)
}

/// Builds a tiled-canvas resize plan for one image.
///
/// # Errors
///
/// Returns an error when dimensions are invalid, `tile_size` or
/// `max_image_tiles` is zero, or arithmetic overflows.
pub fn tiled_canvas_plan(
    original_size: ImageSize,
    tile_size: usize,
    max_image_tiles: usize,
) -> Result<TiledCanvasPlan, TransformError> {
    validate_size(original_size)?;
    if tile_size == 0 {
        return Err(TransformError::InvalidScaleFactor(tile_size));
    }
    validate_max_image_tiles(max_image_tiles)?;

    let grid = select_tiled_canvas_grid(original_size, tile_size, max_image_tiles)?;
    let canvas_size = tiled_canvas_size(grid, tile_size)?;
    let resized_size = fit_size_to_tiled_canvas(original_size, canvas_size, tile_size)?;
    let padding = Padding::new(
        0,
        canvas_size.width - resized_size.width,
        canvas_size.height - resized_size.height,
        0,
    );
    let aspect_ratio_id = tiled_canvas_aspect_ratio_id(grid, max_image_tiles)?;

    Ok(TiledCanvasPlan {
        original_size,
        tile_size,
        max_image_tiles,
        grid,
        canvas_size,
        resized_size,
        padding,
        aspect_ratio_id,
    })
}

/// Builds tiled-canvas metadata for a nested batch of image grids.
///
/// Each outer entry is one batch sample and each inner entry is one image in
/// that sample. Aspect-ratio ids and masks are padded to the maximum image
/// count in the batch, matching tiled-canvas batched metadata contracts.
///
/// # Errors
///
/// Returns an error when no actual images are provided, `max_image_tiles` is
/// zero, a grid is invalid, or metadata shape arithmetic overflows.
pub fn tiled_canvas_batch_metadata(
    sample_grids: &[Vec<TiledCanvasGrid>],
    max_image_tiles: usize,
) -> Result<TiledCanvasBatchMetadata, TransformError> {
    validate_max_image_tiles(max_image_tiles)?;
    if sample_grids.is_empty() || !sample_grids.iter().any(|sample| !sample.is_empty()) {
        return Err(TransformError::EmptyImageBatch);
    }

    let max_images_per_sample = sample_grids
        .iter()
        .map(Vec::len)
        .max()
        .ok_or(TransformError::EmptyImageBatch)?;
    sample_grids
        .len()
        .checked_mul(max_images_per_sample)
        .and_then(|slots| slots.checked_mul(max_image_tiles))
        .ok_or(TransformError::ImageSizeOverflow)?;

    let mut num_tiles = Vec::with_capacity(sample_grids.len());
    let mut aspect_ratio_ids = vec![vec![0; max_images_per_sample]; sample_grids.len()];
    let mut aspect_ratio_mask =
        vec![vec![vec![false; max_image_tiles]; max_images_per_sample]; sample_grids.len()];
    for sample_mask in &mut aspect_ratio_mask {
        for image_mask in sample_mask {
            image_mask[0] = true;
        }
    }

    for (sample_index, grids) in sample_grids.iter().enumerate() {
        let mut sample_num_tiles = Vec::with_capacity(grids.len());
        for (image_index, grid) in grids.iter().copied().enumerate() {
            let tile_count = grid.tile_count()?;
            validate_tiled_canvas_grid(grid, max_image_tiles)?;
            sample_num_tiles.push(tile_count);
            aspect_ratio_ids[sample_index][image_index] =
                tiled_canvas_aspect_ratio_id(grid, max_image_tiles)?;
            aspect_ratio_mask[sample_index][image_index] =
                tiled_canvas_aspect_ratio_mask(grid, max_image_tiles)?;
        }
        num_tiles.push(sample_num_tiles);
    }

    Ok(TiledCanvasBatchMetadata {
        max_images_per_sample,
        num_tiles,
        aspect_ratio_ids,
        aspect_ratio_mask,
    })
}

#[derive(Clone, Copy)]
struct TiledCanvasCandidate {
    grid: TiledCanvasGrid,
    scale: f64,
    area: usize,
}

fn validate_max_image_tiles(max_image_tiles: usize) -> Result<(), TransformError> {
    if max_image_tiles == 0 {
        Err(TransformError::InvalidScaleFactor(max_image_tiles))
    } else {
        Ok(())
    }
}

fn validate_tiled_canvas_grid(
    grid: TiledCanvasGrid,
    max_image_tiles: usize,
) -> Result<(), TransformError> {
    validate_max_image_tiles(max_image_tiles)?;
    let tile_count = grid.tile_count()?;
    if grid.rows == 0 || grid.columns == 0 || tile_count > max_image_tiles {
        return Err(TransformError::InvalidTileGrid {
            rows: grid.rows,
            columns: grid.columns,
            max_image_tiles,
        });
    }
    Ok(())
}

fn select_tiled_canvas_grid(
    original_size: ImageSize,
    tile_size: usize,
    max_image_tiles: usize,
) -> Result<TiledCanvasGrid, TransformError> {
    let grids = supported_tiled_canvas_grids(max_image_tiles)?;
    let mut has_upscaling_candidate = false;
    for grid in grids.iter().copied() {
        let canvas = tiled_canvas_size(grid, tile_size)?;
        if tiled_canvas_scale(original_size, canvas) >= 1.0 {
            has_upscaling_candidate = true;
            break;
        }
    }

    let mut best = None;
    for grid in grids {
        let canvas = tiled_canvas_size(grid, tile_size)?;
        let scale = tiled_canvas_scale(original_size, canvas);
        if has_upscaling_candidate != (scale >= 1.0) {
            continue;
        }

        let candidate = TiledCanvasCandidate {
            grid,
            scale,
            area: checked_pixels(canvas)?,
        };
        if is_better_tiled_canvas_candidate(candidate, best, has_upscaling_candidate) {
            best = Some(candidate);
        }
    }

    best.map(|candidate| candidate.grid)
        .ok_or(TransformError::InvalidSmartResizeParams)
}

fn is_better_tiled_canvas_candidate(
    candidate: TiledCanvasCandidate,
    best: Option<TiledCanvasCandidate>,
    prefer_upscaling: bool,
) -> bool {
    let Some(best) = best else {
        return true;
    };

    if candidate.scale == best.scale {
        return candidate.area < best.area;
    }

    if prefer_upscaling {
        candidate.scale < best.scale
    } else {
        candidate.scale > best.scale
    }
}

fn tiled_canvas_size(grid: TiledCanvasGrid, tile_size: usize) -> Result<ImageSize, TransformError> {
    Ok(ImageSize {
        height: grid
            .rows
            .checked_mul(tile_size)
            .ok_or(TransformError::ImageSizeOverflow)?,
        width: grid
            .columns
            .checked_mul(tile_size)
            .ok_or(TransformError::ImageSizeOverflow)?,
    })
}

fn tiled_canvas_scale(original_size: ImageSize, canvas_size: ImageSize) -> f64 {
    ((canvas_size.height as f64) / (original_size.height as f64))
        .min((canvas_size.width as f64) / (original_size.width as f64))
}

fn fit_size_to_tiled_canvas(
    original_size: ImageSize,
    canvas_size: ImageSize,
    tile_size: usize,
) -> Result<ImageSize, TransformError> {
    let target_width = original_size.width.clamp(tile_size, canvas_size.width);
    let target_height = original_size.height.clamp(tile_size, canvas_size.height);
    let scale_h = (target_height as f64) / (original_size.height as f64);
    let scale_w = (target_width as f64) / (original_size.width as f64);

    if scale_w < scale_h {
        Ok(ImageSize {
            height: floor_scaled_dimension(original_size.height, scale_w)?
                .max(1)
                .min(target_height),
            width: target_width,
        })
    } else {
        Ok(ImageSize {
            height: target_height,
            width: floor_scaled_dimension(original_size.width, scale_h)?
                .max(1)
                .min(target_width),
        })
    }
}
