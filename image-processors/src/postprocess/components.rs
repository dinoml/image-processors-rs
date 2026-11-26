//! Four-connected binary mask components and integer image statistics.

/// Invalid binary mask geometry or component-label capacity.
#[derive(Debug, thiserror::Error)]
pub enum ConnectedComponentsError {
    /// The mask length differs from width times height, or the geometry overflows.
    #[error("binary component mask geometry is invalid")]
    Geometry,
    /// More than u32::MAX connected components were encountered.
    #[error("binary component label count exceeds u32 capacity")]
    Labels,
}

/// Statistics of one nonbackground binary connected component.
#[derive(Clone, Debug)]
pub struct ConnectedComponent {
    /// Nonzero label appearing in the label image.
    pub label: u32,
    /// Leftmost occupied column.
    pub left: usize,
    /// Topmost occupied row.
    pub top: usize,
    /// Number of columns spanned by the component.
    pub width: usize,
    /// Number of rows spanned by the component.
    pub height: usize,
    /// Number of occupied pixels.
    pub area: usize,
}

/// Label nonzero pixels with four-neighbor connectivity in row-major order.
///
/// Background is label zero. Components retain their first row-major pixel's
/// order, with complete integer statistics and no native image dependency.
///
/// # Errors
/// Reports inconsistent/overflowing geometry or label capacity exhaustion.
pub fn connected_components_4(
    mask: &[u8],
    width: usize,
    height: usize,
) -> Result<(Vec<u32>, Vec<ConnectedComponent>), ConnectedComponentsError> {
    if width == 0 || height == 0 || width.checked_mul(height) != Some(mask.len()) {
        return Err(ConnectedComponentsError::Geometry);
    }
    let mut labels = vec![0u32; mask.len()];
    let mut components = Vec::new();
    let mut pending = Vec::new();
    for first in 0..mask.len() {
        if mask[first] == 0 || labels[first] != 0 {
            continue;
        }
        let label = u32::try_from(components.len())
            .ok()
            .and_then(|n| n.checked_add(1))
            .ok_or(ConnectedComponentsError::Labels)?;
        let (mut left, mut top, mut right, mut bottom) =
            (first % width, first / width, first % width, first / width);
        let mut area = 0;
        labels[first] = label;
        pending.push(first);
        while let Some(index) = pending.pop() {
            let (x, y) = (index % width, index / width);
            left = left.min(x);
            top = top.min(y);
            right = right.max(x);
            bottom = bottom.max(y);
            area += 1;
            for neighbor in [
                x.checked_sub(1).map(|x| y * width + x),
                (x + 1 < width).then_some(index + 1),
                y.checked_sub(1).map(|y| y * width + x),
                (y + 1 < height).then_some(index + width),
            ]
            .into_iter()
            .flatten()
            {
                if mask[neighbor] != 0 && labels[neighbor] == 0 {
                    labels[neighbor] = label;
                    pending.push(neighbor);
                }
            }
        }
        components.push(ConnectedComponent {
            label,
            left,
            top,
            width: right - left + 1,
            height: bottom - top + 1,
            area,
        });
    }
    Ok((labels, components))
}
