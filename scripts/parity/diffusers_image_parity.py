#!/usr/bin/env python
"""Generate Diffusers image-processor parity fixtures.

This development-only script records complete deterministic references from
Diffusers image processors. Rust tests consume the JSON fixture without
depending on Python, torch, Pillow, NumPy, or Diffusers at runtime.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import PIL
import torch

from _fixture_contract import (
    acceptance_contract,
    comparison_policy,
    verify_clean_git_checkout,
)
from PIL import Image


AUDIT_COMMIT = "208704a27a6f362b67cd1a04fa1db0b98036d26f"
AUDITED_SOURCE_PATHS = (
    "src/diffusers/image_processor.py",
    "src/diffusers/video_processor.py",
    "src/diffusers/pipelines/deprecated/blip_diffusion/blip_image_processing.py",
    "src/diffusers/pipelines/flux2/image_processor.py",
    "src/diffusers/pipelines/hunyuan_video1_5/image_processor.py",
    "src/diffusers/pipelines/joyimage/image_processor.py",
    "src/diffusers/pipelines/ltx2/image_processor.py",
    "src/diffusers/pipelines/marigold/marigold_image_processing.py",
    "src/diffusers/pipelines/visualcloze/visualcloze_utils.py",
    "src/diffusers/pipelines/wan/image_processor.py",
)
SOURCE_MANIFEST_SHA256 = "8d868a6ffa94e4cfa32234ac6c49a6c2d4edab72e4cdaa90ea091cecec6d49bb"

# Populated by `load_audited_diffusers` after the source checkout is verified.
IPAdapterMaskProcessor: Any = None
PixArtImageProcessor: Any = None
VaeImageProcessor: Any = None
VaeImageProcessorLDM3D: Any = None
BlipImageProcessor: Any = None
Flux2ImageProcessor: Any = None
HunyuanVideo15ImageProcessor: Any = None
JoyImageEditImageProcessor: Any = None
LTX2VideoHDRProcessor: Any = None
MarigoldImageProcessor: Any = None
VisualClozeProcessor: Any = None
WanAnimateImageProcessor: Any = None


def audited_source_paths() -> list[str]:
    """Return the relevant upstream source manifest in stable path order."""
    return sorted(AUDITED_SOURCE_PATHS)


def source_manifest_digest(source: Path) -> str:
    """Hash audited source paths and bytes in a deterministic order."""
    digest = hashlib.sha256()
    for relative_path in audited_source_paths():
        path = source / relative_path
        if not path.is_file():
            raise SystemExit(f"audited Diffusers source file not found: {path}")
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def load_audited_diffusers(source: Path) -> Any:
    """Import Diffusers only after verifying the requested source checkout."""
    source = verify_clean_git_checkout(
        source, expected_commit=AUDIT_COMMIT, project="Diffusers"
    )
    actual_manifest = source_manifest_digest(source)
    if actual_manifest != SOURCE_MANIFEST_SHA256:
        raise SystemExit(
            f"expected audited source manifest {SOURCE_MANIFEST_SHA256}, found {actual_manifest}"
        )

    source_root = source / "src"
    package_root = source_root / "diffusers"
    if not package_root.is_dir():
        raise SystemExit(f"Diffusers package not found at {package_root}")
    sys.path.insert(0, str(source_root))

    import diffusers as diffusers_module
    from diffusers.image_processor import (
        IPAdapterMaskProcessor as ip_adapter_mask_processor,
        PixArtImageProcessor as pixart_image_processor,
        VaeImageProcessor as vae_image_processor,
        VaeImageProcessorLDM3D as vae_image_processor_ldm3d,
    )
    from diffusers.pipelines.deprecated.blip_diffusion.blip_image_processing import (
        BlipImageProcessor as blip_image_processor,
    )
    from diffusers.pipelines.flux2.image_processor import (
        Flux2ImageProcessor as flux2_image_processor,
    )
    from diffusers.pipelines.hunyuan_video1_5.image_processor import (
        HunyuanVideo15ImageProcessor as hunyuan_video_15_image_processor,
    )
    from diffusers.pipelines.joyimage.image_processor import (
        JoyImageEditImageProcessor as joy_image_edit_image_processor,
    )
    from diffusers.pipelines.ltx2.image_processor import (
        LTX2VideoHDRProcessor as ltx2_video_hdr_processor,
    )
    from diffusers.pipelines.marigold.marigold_image_processing import (
        MarigoldImageProcessor as marigold_image_processor,
    )
    from diffusers.pipelines.visualcloze.visualcloze_utils import (
        VisualClozeProcessor as visual_cloze_processor,
    )
    from diffusers.pipelines.wan.image_processor import (
        WanAnimateImageProcessor as wan_animate_image_processor,
    )

    imported_root = Path(diffusers_module.__file__).resolve().parent
    if imported_root != package_root.resolve():
        raise SystemExit(
            f"imported Diffusers from {imported_root}, expected audited source {package_root}"
        )

    globals().update(
        {
            "IPAdapterMaskProcessor": ip_adapter_mask_processor,
            "PixArtImageProcessor": pixart_image_processor,
            "VaeImageProcessor": vae_image_processor,
            "VaeImageProcessorLDM3D": vae_image_processor_ldm3d,
            "BlipImageProcessor": blip_image_processor,
            "Flux2ImageProcessor": flux2_image_processor,
            "HunyuanVideo15ImageProcessor": hunyuan_video_15_image_processor,
            "JoyImageEditImageProcessor": joy_image_edit_image_processor,
            "LTX2VideoHDRProcessor": ltx2_video_hdr_processor,
            "MarigoldImageProcessor": marigold_image_processor,
            "VisualClozeProcessor": visual_cloze_processor,
            "WanAnimateImageProcessor": wan_animate_image_processor,
        }
    )
    return diffusers_module


def deterministic_rgb(width: int, height: int) -> Image.Image:
    values = np.arange(width * height * 3, dtype=np.uint32).reshape(height, width, 3)
    values = ((values * 37 + 17) % 256).astype(np.uint8)
    return Image.fromarray(values, mode="RGB")


def deterministic_luma(width: int, height: int) -> Image.Image:
    values = np.arange(width * height, dtype=np.uint32).reshape(height, width)
    values = ((values * 53 + 11) % 256).astype(np.uint8)
    return Image.fromarray(values, mode="L")


def deterministic_depth(width: int, height: int) -> Image.Image:
    values = np.arange(width * height, dtype=np.uint32).reshape(height, width)
    values = ((values * 4099 + 123) % 65536).astype(np.uint16)
    return Image.fromarray(values, mode="I;16")


def ltx2_reference_frames() -> list[Image.Image]:
    first = deterministic_rgb(50, 70)
    second = first.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    return [first, second]


def summarize(value: Any, sample: int) -> dict[str, Any]:
    if hasattr(value, "detach"):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    array = np.ascontiguousarray(array)
    if array.dtype.byteorder == ">" or (array.dtype.byteorder == "=" and not np.little_endian):
        array = array.byteswap().view(array.dtype.newbyteorder("<"))
    flat = array.reshape(-1)
    numeric = flat.astype(np.float64, copy=False) if flat.size else flat
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "min": float(numeric.min()) if flat.size else None,
        "mean": float(numeric.mean()) if flat.size else None,
        "max": float(numeric.max()) if flat.size else None,
        "sample": flat[:sample].tolist(),
        "data_base64": base64.b64encode(array.tobytes(order="C")).decode("ascii"),
        "data_encoding": "base64_little_endian",
    }


def vae_case(sample: int) -> dict[str, Any]:
    image = deterministic_rgb(2, 1)
    processor = VaeImageProcessor(do_resize=False, do_convert_rgb=True)
    postprocess_input = torch.tensor(
        [[[[-1.0, 0.0]], [[0.5, 1.0]], [[1.0, -1.0]]]],
        dtype=torch.float32,
    )
    return {
        "family": "vae",
        "class_name": "VaeImageProcessor",
        "config": {
            "do_resize": False,
            "do_normalize": True,
            "do_convert_rgb": True,
        },
        "image": {"mode": image.mode, "width": image.width, "height": image.height},
        "inputs": {"postprocess": summarize(postprocess_input, sample)},
        "outputs": {
            "preprocess": summarize(processor.preprocess(image), sample),
            "postprocess_np": summarize(processor.postprocess(postprocess_input, output_type="np"), sample),
        },
    }


def ip_adapter_mask_case(sample: int) -> dict[str, Any]:
    mask = torch.tensor([[[0.0, 0.25], [0.5, 1.0]]], dtype=torch.float32)
    downsampled = IPAdapterMaskProcessor.downsample(
        mask,
        batch_size=2,
        num_queries=5,
        value_embed_dim=3,
    )
    return {
        "family": "ip_adapter_mask",
        "class_name": "IPAdapterMaskProcessor",
        "config": {},
        "comparison": comparison_policy(1.0e-4),
        "inputs": {
            "mask": summarize(mask, sample),
            "batch_size": 2,
            "num_queries": 5,
            "value_embed_dim": 3,
        },
        "outputs": {"downsample": summarize(downsampled, sample)},
    }


def ldm3d_case(sample: int) -> dict[str, Any]:
    rgb = deterministic_rgb(16, 8)
    depth = deterministic_depth(16, 8)
    processor = VaeImageProcessorLDM3D(do_resize=True)
    rgb_tensor, depth_tensor = processor.preprocess(rgb, depth)
    postprocess_input = torch.tensor([[[[-1.0]], [[0.0]], [[1.0]], [[1.0]]]], dtype=torch.float32)
    post_rgb, post_depth = processor.postprocess(postprocess_input, output_type="pil")
    return {
        "family": "ldm3d",
        "class_name": "VaeImageProcessorLDM3D",
        "config": {"do_resize": True, "do_normalize": True},
        "image": {"mode": rgb.mode, "width": rgb.width, "height": rgb.height},
        "depth": {
            "mode": depth.mode,
            "width": depth.width,
            "height": depth.height,
            "values": np.asarray(depth).reshape(-1).astype(int).tolist(),
        },
        "inputs": {"postprocess": summarize(postprocess_input, sample)},
        "outputs": {
            "rgb_preprocess": summarize(rgb_tensor, sample),
            "depth_preprocess": summarize(depth_tensor, sample),
            "postprocess_rgb": summarize(np.asarray(post_rgb[0]), sample),
            "postprocess_depth": summarize(np.asarray(post_depth[0]), sample),
        },
    }


def blip_case(sample: int) -> dict[str, Any]:
    image = deterministic_rgb(3, 5)
    processor = BlipImageProcessor(
        do_resize=True,
        size={"height": 4, "width": 4},
        do_center_crop=True,
        do_convert_rgb=True,
    )
    pixel_values = processor.preprocess(image, return_tensors="np")["pixel_values"]
    postprocess_input = torch.tensor(
        [[[[-1.0, 0.0]], [[0.5, 1.0]], [[1.0, -1.0]]]],
        dtype=torch.float32,
    )
    return {
        "family": "blip",
        "class_name": "BlipImageProcessor",
        "config": {
            "do_resize": True,
            "size": {"height": 4, "width": 4},
            "do_center_crop": True,
            "do_convert_rgb": True,
            "do_rescale": True,
            "do_normalize": True,
        },
        "image": {"mode": image.mode, "width": image.width, "height": image.height},
        "inputs": {"postprocess": summarize(postprocess_input, sample)},
        "outputs": {
            "preprocess": summarize(pixel_values, sample),
            "postprocess_np": summarize(processor.postprocess(postprocess_input, output_type="np"), sample),
        },
    }


def flux2_case(sample: int) -> dict[str, Any]:
    image = deterministic_rgb(6, 4)
    second = deterministic_luma(2, 6)
    processor = Flux2ImageProcessor(
        do_resize=True,
        vae_scale_factor=16,
        vae_latent_channels=32,
        do_convert_rgb=True,
    )
    try:
        Flux2ImageProcessor.check_image_input(image)
        check_error = None
    except ValueError as error:
        check_error = str(error)
    concatenated = processor.concatenate_images([image, second])
    limited = processor._resize_if_exceeds_area(concatenated, target_area=12)
    preprocessed = processor.preprocess(image, height=32, width=48)
    return {
        "family": "flux2",
        "class_name": "Flux2ImageProcessor",
        "config": {
            "do_resize": True,
            "vae_scale_factor": 16,
            "vae_latent_channels": 32,
            "do_convert_rgb": True,
            "do_normalize": True,
        },
        "image": {"mode": image.mode, "width": image.width, "height": image.height},
        "inputs": {
            "second_image": {"mode": second.mode, "width": second.width, "height": second.height},
            "height": 32,
            "width": 48,
            "target_area": 12,
        },
        "outputs": {
            "check_image_input_error": check_error,
            "concatenate": summarize(np.asarray(concatenated), sample),
            "area_limited": summarize(np.asarray(limited), sample),
            "preprocess": summarize(preprocessed, sample),
        },
    }


def visual_cloze_case(sample: int) -> dict[str, Any]:
    context = deterministic_rgb(6, 4)
    context_second = deterministic_rgb(4, 4)
    query = deterministic_rgb(4, 6)
    input_images = [[context, context_second], [query, None]]
    processor = VisualClozeProcessor(
        resolution=64,
        vae_scale_factor=16,
        vae_latent_channels=16,
    )
    output = processor.preprocess(
        task_prompt="complete the missing panel",
        content_prompt="small deterministic fixture",
        input_images=input_images,
        vae_scale_factor=16,
    )
    upsampling_input = [[deterministic_rgb(2, 2)]]
    upsampling_output = processor.preprocess(
        task_prompt="upsample",
        content_prompt="fixture",
        input_images=upsampling_input,
        height=32,
        width=32,
        upsampling=True,
        vae_scale_factor=16,
    )
    return {
        "family": "visual_cloze",
        "class_name": "VisualClozeProcessor",
        "config": {
            "resolution": 64,
            "vae_scale_factor": 16,
            "vae_latent_channels": 16,
            "do_normalize": True,
        },
        "inputs": {
            "grid": [
                [
                    {"mode": context.mode, "width": context.width, "height": context.height},
                    {
                        "mode": context_second.mode,
                        "width": context_second.width,
                        "height": context_second.height,
                    },
                ],
                [
                    {"mode": query.mode, "width": query.width, "height": query.height},
                    None,
                ],
            ],
            "upsampling": {"width": 32, "height": 32},
        },
        "outputs": {
            "target_position": output["target_position"][0],
            "image_sizes": output["image_size"][0],
            "image_0_0": summarize(output["init_image"][0][0][0], sample),
            "image_0_1": summarize(output["init_image"][0][0][1], sample),
            "image_1_0": summarize(output["init_image"][0][1][0], sample),
            "image_1_1": summarize(output["init_image"][0][1][1], sample),
            "mask_0_0": summarize(output["mask"][0][0][0], sample),
            "mask_0_1": summarize(output["mask"][0][0][1], sample),
            "mask_1_0": summarize(output["mask"][0][1][0], sample),
            "mask_1_1": summarize(output["mask"][0][1][1], sample),
            "upsampling_image": summarize(upsampling_output["init_image"][0][0][0], sample),
            "upsampling_mask": summarize(upsampling_output["mask"][0][0][0], sample),
        },
    }


def hunyuan_video_15_case(sample: int) -> dict[str, Any]:
    processor = HunyuanVideo15ImageProcessor(
        do_resize=True,
        vae_scale_factor=16,
        vae_latent_channels=32,
        do_convert_rgb=True,
    )
    landscape = processor.calculate_default_height_width(720, 1280, 256)
    portrait = processor.calculate_default_height_width(1280, 720, 256)
    square = processor.calculate_default_height_width(512, 512, 256)
    first_frame = deterministic_rgb(16, 16)
    second_frame = deterministic_rgb(16, 16).transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    video = processor.preprocess_video(
        [first_frame, second_frame],
        height=16,
        width=16,
    )
    rust_video = video.permute(0, 2, 1, 3, 4).contiguous()[0]
    return {
        "family": "hunyuan_video_15",
        "class_name": "HunyuanVideo15ImageProcessor",
        "config": {
            "do_resize": True,
            "vae_scale_factor": 16,
            "vae_latent_channels": 32,
            "do_convert_rgb": True,
        },
        "inputs": {
            "target_size": 256,
            "landscape": {"height": 720, "width": 1280},
            "portrait": {"height": 1280, "width": 720},
            "square": {"height": 512, "width": 512},
            "video_frames": [
                {"mode": first_frame.mode, "width": first_frame.width, "height": first_frame.height},
                {
                    "mode": second_frame.mode,
                    "width": second_frame.width,
                    "height": second_frame.height,
                    "horizontal_flip": True,
                },
            ],
        },
        "outputs": {
            "landscape": {"height": landscape[0], "width": landscape[1]},
            "portrait": {"height": portrait[0], "width": portrait[1]},
            "square": {"height": square[0], "width": square[1]},
            "video_preprocess": summarize(rust_video, sample),
        },
    }


def marigold_case(sample: int) -> dict[str, Any]:
    image = deterministic_rgb(3, 2)
    processor = MarigoldImageProcessor(
        vae_scale_factor=4,
        do_normalize=True,
        do_range_check=True,
    )
    preprocessed, padding, original_resolution = processor.preprocess(image)
    max_edge = 4
    max_edge_preprocessed, max_edge_padding, max_edge_original_resolution = processor.preprocess(
        image,
        processing_resolution=max_edge,
    )
    padded_depth = torch.tensor(
        [[[[0.0, 0.2, 0.4, 9.0], [0.6, 0.8, 1.0, 9.0], [9.0, 9.0, 9.0, 9.0], [9.0, 9.0, 9.0, 9.0]]]],
        dtype=torch.float32,
    )
    depth = processor.unpad_image(padded_depth, padding)
    depth_u16 = processor.export_depth_to_16bit_png(depth)[0]
    padded_normals = torch.tensor(
        [
            [
                [[-1.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
                [[1.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            ]
        ],
        dtype=torch.float32,
    )
    normals_visual = processor.visualize_normals(processor.unpad_image(padded_normals, padding))[0]
    uncertainty_visual = processor.visualize_uncertainty(depth, saturation_percentile=100)[0]
    return {
        "family": "marigold",
        "class_name": "MarigoldImageProcessor",
        "comparison": comparison_policy(2.0e-5),
        "config": {
            "vae_scale_factor": 4,
            "do_normalize": True,
            "do_range_check": True,
            "resample": "bilinear",
        },
        "image": {"mode": image.mode, "width": image.width, "height": image.height},
        "inputs": {
            "processing_resolution": max_edge,
            "padded_depth": summarize(padded_depth, sample),
        },
        "outputs": {
            "preprocess": summarize(preprocessed, sample),
            "padding": list(padding),
            "original_resolution": {"height": original_resolution[0], "width": original_resolution[1]},
            "max_edge_preprocess": summarize(max_edge_preprocessed, sample),
            "max_edge_padding": list(max_edge_padding),
            "max_edge_original_resolution": {
                "height": max_edge_original_resolution[0],
                "width": max_edge_original_resolution[1],
            },
            "depth_u16": summarize(np.asarray(depth_u16), sample),
            "normals_visual": summarize(np.asarray(normals_visual), sample),
            "uncertainty_visual": summarize(np.asarray(uncertainty_visual), sample),
        },
    }


def joy_image_edit_case(sample: int) -> dict[str, Any]:
    image = deterministic_rgb(5, 3)
    processor = JoyImageEditImageProcessor(
        vae_scale_factor=8,
        basesize=1024,
        resample="bilinear",
    )
    height, width = processor.get_default_height_width(image, height=600, width=1000)
    processed = processor.resize_center_crop(image, (height, width))
    preprocessed = processor.preprocess(processed)
    return {
        "family": "joy_image_edit",
        "class_name": "JoyImageEditImageProcessor",
        "config": {
            "do_resize": True,
            "vae_scale_factor": 8,
            "basesize": 1024,
            "resample": "bilinear",
            "do_normalize": True,
        },
        "image": {"mode": image.mode, "width": image.width, "height": image.height},
        "inputs": {
            "height": 600,
            "width": 1000,
        },
        "outputs": {
            "target_size": {"height": height, "width": width},
            "preprocess": summarize(preprocessed, sample),
        },
    }


def wan_animate_case(sample: int) -> dict[str, Any]:
    image = deterministic_rgb(33, 20)
    processor = WanAnimateImageProcessor(
        vae_scale_factor=8,
        vae_latent_channels=16,
        spatial_patch_size=(2, 2),
        resample="lanczos",
    )
    height, width = processor.get_default_height_width(image, height=20, width=33)
    preprocessed = processor.preprocess(image, height=20, width=33, resize_mode="fill")
    return {
        "family": "wan_animate",
        "class_name": "WanAnimateImageProcessor",
        "comparison": comparison_policy(2.0e-5),
        "config": {
            "do_resize": True,
            "vae_scale_factor": 8,
            "vae_latent_channels": 16,
            "spatial_patch_size": [2, 2],
            "resample": "lanczos",
            "do_normalize": True,
            "fill_color": 0,
        },
        "image": {"mode": image.mode, "width": image.width, "height": image.height},
        "inputs": {
            "height": 20,
            "width": 33,
            "resize_mode": "fill",
        },
        "outputs": {
            "target_size": {"height": height, "width": width},
            "preprocess": summarize(preprocessed, sample),
        },
    }


def ltx2_video_hdr_case(sample: int) -> dict[str, Any]:
    processor = LTX2VideoHDRProcessor(vae_scale_factor=32, hdr_transform="logc3")
    frames = ltx2_reference_frames()
    vae_preprocessed = processor.preprocess_video(frames, height=None, width=None)
    rust_vae_preprocessed = vae_preprocessed[0].permute(1, 0, 2, 3).contiguous()
    reference = processor.preprocess_reference_video_hdr(frames, height=40, width=32)
    rust_reference = reference[0].permute(1, 0, 2, 3).contiguous()
    rust_postprocess_input = torch.tensor(
        [[[[[-0.814382]], [[0.0]], [[1.0]]], [[[-1.0]], [[-0.7006843]], [[-0.814382]]]]],
        dtype=torch.float32,
    )
    diffusers_postprocess_input = rust_postprocess_input.permute(0, 2, 1, 3, 4).contiguous()
    postprocessed = processor.postprocess_hdr_video(diffusers_postprocess_input, output_type="pt")
    return {
        "family": "ltx2_video_hdr",
        "class_name": "LTX2VideoHDRProcessor",
        "config": {"do_resize": True, "vae_scale_factor": 32, "hdr_transform": "logc3"},
        "inputs": {
            "video_frames": {
                "count": len(frames),
                "height": frames[0].height,
                "width": frames[0].width,
                "second_frame_horizontal_flip": True,
            },
            "target_size": {"height": 40, "width": 32},
            "postprocess": summarize(rust_postprocess_input, sample),
        },
        "outputs": {
            "vae_preprocess": summarize(rust_vae_preprocessed, sample),
            "reference_preprocess": summarize(rust_reference, sample),
            "hdr_postprocess": summarize(postprocessed, sample),
        },
    }


def pixart_case(sample_count: int) -> dict[str, Any]:
    ratios = {
        "1.0": (512, 512),
        str(512 / 768): (512, 768),
        str(768 / 512): (768, 512),
    }
    selected = PixArtImageProcessor.classify_height_width_bin(600, 1000, ratios)
    sample = torch.arange(1 * 3 * 4 * 8, dtype=torch.float32).reshape(1, 3, 4, 8) / 95.0
    target_height = 6
    target_width = 6
    resized = PixArtImageProcessor.resize_and_crop_tensor(
        sample,
        new_width=target_width,
        new_height=target_height,
    )
    ratio = max(target_height / sample.shape[2], target_width / sample.shape[3])
    resized_height = int(sample.shape[2] * ratio)
    resized_width = int(sample.shape[3] * ratio)
    crop_top = (resized_height - target_height) // 2
    crop_left = (resized_width - target_width) // 2
    return {
        "family": "pixart",
        "class_name": "PixArtImageProcessor",
        "config": {},
        "inputs": {
            "height": 600,
            "width": 1000,
            "ratios": {key: list(value) for key, value in ratios.items()},
            "resize_crop_input": summarize(sample, sample_count),
            "resize_crop_target": {"height": target_height, "width": target_width},
        },
        "outputs": {
            "selected_size": list(selected),
            "resize_crop": summarize(resized, sample_count),
            "resize_crop_plan": {
                "resized_size": {"height": resized_height, "width": resized_width},
                "crop_top": crop_top,
                "crop_left": crop_left,
            },
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=16, help="Number of flattened values to record.")
    parser.add_argument(
        "--diffusers-source",
        type=Path,
        required=True,
        help=f"Diffusers checkout pinned to {AUDIT_COMMIT}.",
    )
    parser.add_argument("--output", type=Path, required=True, help="JSON fixture output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    diffusers = load_audited_diffusers(args.diffusers_source)
    fixture = {
        "schema": "image-processors.diffusers-parity.v2",
        "generator": "scripts/parity/diffusers_image_parity.py",
        "upstream": {
            "library": "diffusers",
            "version": diffusers.__version__,
            "commit": AUDIT_COMMIT,
            "source": "diffusers.image_processor+diffusers.pipelines.deprecated.blip_diffusion.blip_image_processing+diffusers.pipelines.flux2.image_processor+diffusers.pipelines.hunyuan_video1_5.image_processor+diffusers.pipelines.joyimage.image_processor+diffusers.pipelines.ltx2.image_processor+diffusers.pipelines.marigold.marigold_image_processing+diffusers.pipelines.visualcloze.visualcloze_utils+diffusers.pipelines.wan.image_processor",
            "source_files": audited_source_paths(),
            "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        },
        "generator_versions": {
            "numpy": str(np.__version__),
            "pillow": str(PIL.__version__),
            "torch": str(torch.__version__),
        },
        "acceptance": acceptance_contract(
            input_locations=[
                "cases[].image",
                "cases[].depth",
                "cases[].inputs",
            ],
            input_description=(
                "RGB values use (index * 37 + 17) % 256; luma values use "
                "(index * 53 + 11) % 256; depth values use "
                "(index * 4099 + 123) % 65536; remaining tensors are embedded in full."
            ),
            config_locations=["cases[].config"],
            absolute_tolerance=1.0e-5,
        ),
        "cases": [
            vae_case(args.sample),
            ip_adapter_mask_case(args.sample),
            ldm3d_case(args.sample),
            blip_case(args.sample),
            flux2_case(args.sample),
            visual_cloze_case(args.sample),
            hunyuan_video_15_case(args.sample),
            marigold_case(args.sample),
            joy_image_edit_case(args.sample),
            ltx2_video_hdr_case(args.sample),
            wan_animate_case(args.sample),
            pixart_case(args.sample),
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
