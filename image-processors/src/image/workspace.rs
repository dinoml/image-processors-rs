use super::*;

/// Reusable owned buffers for repeated image preprocessing calls.
#[derive(Clone, Debug, Default)]
pub struct ImageProcessorWorkspace {
    f32_values: Vec<f32>,
    f16_values: Vec<f16>,
    resize: ResizeWorkspace,
    parallel_resize: Vec<ParallelResizeWorkspace>,
}

impl ImageProcessorWorkspace {
    /// Creates an empty workspace.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the reusable `f32` buffer capacity.
    pub fn f32_capacity(&self) -> usize {
        self.f32_values.capacity()
    }

    /// Returns the reusable `f16` buffer capacity.
    pub fn f16_capacity(&self) -> usize {
        self.f16_values.capacity()
    }

    /// Returns the number of cached resize coefficient tables.
    pub fn resize_coefficient_cache_len(&self) -> usize {
        self.resize.coefficient_cache_len()
    }

    /// Clears all reusable buffers while preserving their capacity.
    pub fn clear(&mut self) {
        self.f32_values.clear();
        self.f16_values.clear();
        self.resize.clear();
        for workspace in &mut self.parallel_resize {
            workspace.clear();
        }
    }

    /// Recycles a returned `F32` or `F16` tensor buffer for a future workspace call.
    ///
    /// # Errors
    ///
    /// Returns an error when the tensor does not contain `F32` or `F16` storage.
    pub fn recycle_tensor(&mut self, tensor: Tensor) -> Result<(), ImageProcessorError> {
        let (data, _, _, _) = tensor.into_parts();
        match data {
            TensorData::F32(values) => {
                self.f32_values = values;
                Ok(())
            }
            TensorData::F16(values) => {
                self.f16_values = values;
                Ok(())
            }
            _ => Err(ImageProcessorError::ExpectedDataType("F32 or F16")),
        }
    }

    pub(super) fn take_f32_values(&mut self, len: usize) -> Vec<f32> {
        let mut values = std::mem::take(&mut self.f32_values);
        values.clear();
        values.resize(len, 0.0);
        values
    }

    pub(crate) fn take_f16_values(&mut self, len: usize) -> Vec<f16> {
        let mut values = std::mem::take(&mut self.f16_values);
        values.clear();
        values.resize(len, f16::ZERO);
        values
    }

    pub(super) fn resize_workspace(&mut self) -> &mut ResizeWorkspace {
        &mut self.resize
    }

    #[cfg(feature = "parallel")]
    pub(super) fn parallel_resize_workspaces(
        &mut self,
        count: usize,
    ) -> &mut [ParallelResizeWorkspace] {
        self.parallel_resize
            .resize_with(count, ParallelResizeWorkspace::new);
        &mut self.parallel_resize[..count]
    }
}

#[derive(Clone, Debug, Default)]
pub(super) struct ParallelResizeWorkspace {
    resize: ResizeWorkspace,
}

impl ParallelResizeWorkspace {
    #[cfg(feature = "parallel")]
    fn new() -> Self {
        Self::default()
    }

    fn clear(&mut self) {
        self.resize.clear();
    }

    #[cfg(feature = "parallel")]
    pub(super) fn resize_workspace(&mut self) -> &mut ResizeWorkspace {
        &mut self.resize
    }
}
