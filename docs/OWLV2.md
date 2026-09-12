# OWLv2 image preparation

The OWLv2 task processor rescales RGB values, pads the bottom/right to a square,
resizes with antialiasing and normalizes using the original CLIP statistics.
Enlargement follows SciPy's whole-sample mirror boundary with grid-mode pixel
centres. Edge samples reflect into the image instead of repeating its outermost
pixel. Singleton axes remain constant. Returned normalized boxes refer to the
padded square: multiply both axes by the original maximum side when restoring
coordinates to the source image.

The resize regression covers all pixels of a 2x2 to 4x4 enlargement and a
singleton image. Existing catalog fixtures also exercise the task processor.
