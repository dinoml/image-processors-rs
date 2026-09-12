# SigLIP2 NaFlex

`processors::siglip2::Siglip2ImageProcessor` implements the original aspect-preserving
NaFlex path for RGB images. Construct it with the checkpoint's patch size and patch
budget, then call `preprocess` with a decoded `ImageFrame`.

The processor uses the original binary search for the largest admitted scale,
rounding each image axis upward to a patch boundary. It applies Pillow-compatible
bilinear resampling, rescales RGB by 1/255 and normalizes with mean/std 0.5.
Patches use grid row/column order and height/width/RGB order within each patch.
Unused patch slots are zero after normalization. The result includes the real
patch grid, integer attention mask, original size and resized size.

For patch size 16 and budget 256, a 640-by-480 image becomes 288-by-224 pixels
(18-by-14 patches), leaving four padded patch slots. A square image becomes
256-by-256 pixels with all 256 slots active. This public processor is separate
from the compact fixed-square catalog fixture entry. Learned positional resizing
and model attention remain the model runtime's responsibility.
