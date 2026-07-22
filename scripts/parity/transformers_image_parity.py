#!/usr/bin/env python
"""Generate Transformers image-processor parity fixtures.

This script is intentionally outside the Rust crate. It records reference
outputs from Hugging Face Transformers processors so Rust tests can compare
shape, metadata, and numeric tolerances without binding Rust ownership to
Python.
"""

from __future__ import annotations

import argparse
import base64
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import transformers
from transformers import (
    AutoImageProcessor,
    AutoVideoProcessor,
    DonutImageProcessor,
    Gemma3ImageProcessor,
    Idefics3ImageProcessor,
    LlavaNextImageProcessor,
    MllamaImageProcessor,
    PixtralImageProcessor,
)
from transformers.image_processing_utils import get_patch_output_size
from transformers.image_utils import ChannelDimension
from transformers.models.llava_next.image_processing_llava_next import select_best_resolution
from transformers.models.mllama.image_processing_mllama import (
    get_image_size_fit_to_canvas as mllama_image_size_fit_to_canvas,
    get_optimal_tiled_canvas as mllama_optimal_tiled_canvas,
)
from transformers.models.pixtral.image_processing_pixtral import (
    get_resize_output_image_size as pixtral_resize_output_size,
)
from transformers.models.videomae.image_processing_videomae import VideoMAEImageProcessor
from transformers.models.vivit.image_processing_vivit import VivitImageProcessor

from _fixture_contract import verify_clean_git_checkout


DEFAULT_MODELS = {
    "clip": "openai/clip-vit-base-patch32",
    "vit": "google/vit-base-patch16-224-in21k",
    "videomae": "VideoMAEImageProcessor(size={'shortest_edge': 4}, crop_size={'height': 4, 'width': 4})",
    "vivit": "VivitImageProcessor(size={'shortest_edge': 4}, crop_size={'height': 4, 'width': 4})",
    "qwen_vl": "Qwen/Qwen2-VL-7B-Instruct",
    "llava_next": "LlavaNextImageProcessor()",
    "llava_next_nhwc": "LlavaNextImageProcessor(custom tiny grid, data_format=channels_last)",
    "pixtral": "PixtralImageProcessor()",
    "idefics3": "Idefics3ImageProcessor(size={'longest_edge': 10}, max_image_size={'longest_edge': 5})",
    "gemma3": "Gemma3ImageProcessor(size={'height': 5, 'width': 5}, do_pan_and_scan=True)",
    "mllama": "MllamaImageProcessor(size={'height': 5, 'width': 5}, max_image_tiles=4)",
    "donut": "DonutImageProcessor(size={'height': 12, 'width': 8})",
    "detr": "facebook/detr-resnet-50",
    "sam": "facebook/sam-vit-base",
}

# The dedicated fixtures share the exact source revision used by the catalog
# audit. Generation refuses a wheel or a different/dirty checkout.
TRANSFORMERS_FIXTURE_VERSION = "5.14.0.dev0"
TRANSFORMERS_FIXTURE_COMMIT = "6d960ca0a0eba0d2aebc920d8080a9353da468d3"

FLOAT_TOLERANCES = {
    "clip": (1.0e-6, 0.01),
    "vit": (1.0e-6, 0.01),
    "videomae": (0.06, 0.01),
    "vivit": (0.06, 0.01),
    "qwen_vl": (0.05, 0.01),
    "llava_next": (0.05, 0.01),
    "llava_next_nhwc": (0.05, 0.01),
    "pixtral": (0.05, 0.01),
    "idefics3": (1.0e-6, 1.0e-6),
    "gemma3": (1.0e-6, 1.0e-6),
    "mllama": (1.0e-6, 1.0e-6),
    "donut": (0.06, 0.01),
    "detr": (0.05, 0.01),
    "sam": (0.05, 0.01),
}


def transformers_source_root() -> Path:
    module_path = Path(transformers.__file__).resolve()
    for parent in module_path.parents:
        if (parent / ".git").exists():
            return parent
    raise SystemExit(
        "fixture generation requires Transformers from an exact Git checkout; "
        f"loaded {module_path}"
    )


def verify_fixture_upstream() -> Path:
    if transformers.__version__ != TRANSFORMERS_FIXTURE_VERSION:
        raise SystemExit(
            "fixture generation requires Transformers "
            f"{TRANSFORMERS_FIXTURE_VERSION} (Git commit "
            f"{TRANSFORMERS_FIXTURE_COMMIT}), found {transformers.__version__}"
        )
    return verify_clean_git_checkout(
        transformers_source_root(),
        expected_commit=TRANSFORMERS_FIXTURE_COMMIT,
        project="Transformers",
    )


def json_compatible(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return json_compatible(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_compatible(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    enum_value = getattr(value, "value", None)
    if enum_value is not None:
        return json_compatible(enum_value)
    return str(value)


def effective_processor_config(processor: Any) -> dict[str, Any]:
    config = processor.to_dict()
    return {
        key: json_compatible(value)
        for key, value in config.items()
        if not key.startswith("_") and key not in {"image_processor_type", "processor_class"}
    }


def comparison_policy(family: str) -> dict[str, Any]:
    full_values, statistics = FLOAT_TOLERANCES[family]
    return {
        "float_full_values": {
            "absolute_tolerance": full_values,
            "relative_tolerance": 0.0,
        },
        "float_statistics": {
            "absolute_tolerance": statistics,
            "relative_tolerance": 0.0,
        },
        "integer_values": "exact",
        "shape": "exact",
        "dtype": "exact",
        "metadata": "exact",
    }


def deterministic_image(width: int, height: int, offset: int = 0) -> Image.Image:
    values = np.arange(width * height * 3, dtype=np.uint32).reshape(height, width, 3)
    values = ((values * 37 + 17 + offset * 53) % 256).astype(np.uint8)
    return Image.fromarray(values, mode="RGB")


def image_summary(image: Image.Image) -> dict[str, Any]:
    return {
        "mode": image.mode,
        "width": image.width,
        "height": image.height,
    }


def processor_class_name(processor: Any) -> str:
    return processor.__class__.__name__


PROCESSOR_CLASS_NAMES = {
    "llava_next": "LlavaNextImageProcessor",
    "llava_next_nhwc": "LlavaNextImageProcessor",
    "pixtral": "PixtralImageProcessor",
    "idefics3": "Idefics3ImageProcessor",
    "gemma3": "Gemma3ImageProcessor",
    "mllama": "MllamaImageProcessor",
    "donut": "DonutImageProcessor",
    "videomae": "VideoMAEImageProcessor",
    "vivit": "VivitImageProcessor",
}


def processor_class(family: str, backend: str) -> type:
    class_name = PROCESSOR_CLASS_NAMES[family]
    if backend == "pil" and family != "vivit":
        class_name += "Pil"
    try:
        return getattr(transformers, class_name)
    except AttributeError as error:
        raise SystemExit(
            f"Transformers {TRANSFORMERS_FIXTURE_COMMIT} does not expose "
            f"{class_name} for {family}:{backend}"
        ) from error


def load_or_generate_images(
    path: Path | None,
    width: int,
    height: int,
    image_count: int,
    width_step: int,
    height_step: int,
) -> list[Image.Image]:
    if image_count < 1:
        raise SystemExit("--image-count must be at least 1")
    if path is None:
        images = []
        for index in range(image_count):
            image_width = width + index * width_step
            image_height = height + index * height_step
            if image_width < 1 or image_height < 1:
                raise SystemExit("generated image dimensions must stay positive")
            images.append(deterministic_image(image_width, image_height, index))
        return images
    if image_count != 1 or width_step != 0 or height_step != 0:
        raise SystemExit("--image cannot be combined with generated multi-image options")
    return [Image.open(path).convert("RGB")]


def input_contract(args: argparse.Namespace, images: list[Image.Image]) -> dict[str, Any]:
    """Describe the exact input bytes independently of the local source path."""
    common = {
        "image_count": len(images),
        "as_video": args.as_video,
        "video_batch_size": args.video_batch_size,
    }
    if args.image is not None:
        image = images[0]
        return {
            "kind": "rgb_file",
            "description": "The source image is converted to RGB before processing.",
            "parameters": {
                **common,
                "width": image.width,
                "height": image.height,
                "rgb_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
            },
        }
    return {
        "kind": "deterministic_rgb",
        "description": (
            "For flattened RGB byte index i and image offset n, "
            "value = (i * 37 + 17 + n * 53) % 256."
        ),
        "parameters": {
            **common,
            "width": args.width,
            "height": args.height,
            "width_step": args.width_step,
            "height_step": args.height_step,
        },
    }


def build_video_batch(images: list[Image.Image], batch_size: int) -> list[list[Image.Image]]:
    if batch_size < 1:
        raise SystemExit("--video-batch-size must be at least 1")
    if batch_size == 1:
        return [images]

    videos = []
    for batch_index in range(batch_size):
        frames = []
        for frame_index, image in enumerate(images):
            offset = batch_index * len(images) + frame_index
            frames.append(deterministic_image(image.width, image.height, offset))
        videos.append(frames)
    return videos


def summarize_tensor(value: Any, sample: int, include_data: bool = False) -> dict[str, Any]:
    if hasattr(value, "detach"):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    flat = array.reshape(-1)
    numeric = flat.astype(np.float64, copy=False) if flat.size else flat
    summary = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "min": float(numeric.min()) if flat.size else None,
        "mean": float(numeric.mean()) if flat.size else None,
        "max": float(numeric.max()) if flat.size else None,
        "sample": flat[:sample].tolist(),
    }
    if include_data:
        contiguous = np.ascontiguousarray(array)
        summary["data"] = {
            "encoding": "base64",
            "dtype": str(contiguous.dtype),
            "value": base64.b64encode(contiguous.tobytes()).decode("ascii"),
        }
    return summary


def summarize_value(value: Any, sample: int, include_data: bool = False) -> Any:
    if hasattr(value, "detach") or isinstance(value, np.ndarray):
        return summarize_tensor(value, sample, include_data)
    if isinstance(value, (list, tuple)):
        return [summarize_value(item, sample, include_data) for item in value]
    if isinstance(value, dict):
        return {key: summarize_value(item, sample, include_data) for key, item in value.items()}
    return value


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def llava_next_processor_config(processor: LlavaNextImageProcessor) -> dict[str, Any]:
    resample = getattr(processor.resample, "name", str(processor.resample))
    return {
        "size": processor.size,
        "crop_size": processor.crop_size,
        "image_grid_pinpoints": processor.image_grid_pinpoints,
        "resample": resample,
        "do_center_crop": processor.do_center_crop,
        "do_pad": processor.do_pad,
        "do_convert_rgb": processor.do_convert_rgb,
        "do_rescale": processor.do_rescale,
        "rescale_factor": processor.rescale_factor,
        "do_normalize": processor.do_normalize,
        "image_mean": processor.image_mean,
        "image_std": processor.image_std,
    }


def llava_next_metadata(images: list[Image.Image], processor: LlavaNextImageProcessor) -> dict[str, Any]:
    patch_size = int(processor.crop_size["height"])
    metadata: dict[str, Any] = {
        "image_sizes": [],
        "selected_sizes": [],
        "reshaped_input_sizes": [],
        "image_patch_counts": [],
        "patch_grids": [],
    }
    for image in images:
        original_size = (image.height, image.width)
        selected_size = select_best_resolution(original_size, processor.image_grid_pinpoints)
        image_array = np.asarray(image)
        reshaped_size = get_patch_output_size(
            image_array,
            selected_size,
            input_data_format=ChannelDimension.LAST,
        )
        patch_grid = [
            ceil_div(int(selected_size[0]), patch_size),
            ceil_div(int(selected_size[1]), patch_size),
        ]
        metadata["image_sizes"].append([int(original_size[0]), int(original_size[1])])
        metadata["selected_sizes"].append([int(selected_size[0]), int(selected_size[1])])
        metadata["reshaped_input_sizes"].append([int(reshaped_size[0]), int(reshaped_size[1])])
        metadata["image_patch_counts"].append(1 + patch_grid[0] * patch_grid[1])
        metadata["patch_grids"].append(patch_grid)
    return metadata


def generate_llava_next_case(
    model_id: str,
    images: list[Image.Image],
    sample: int,
    include_output_data: bool,
    include_stage_data: bool,
    backend: str,
) -> dict[str, Any]:
    processor = processor_class("llava_next", backend)()
    encoded = processor(images=images, return_tensors="pt")
    processed = processor(
        images=images,
        return_tensors="np",
        do_rescale=False,
        do_normalize=False,
    )["pixel_values"]
    outputs = {key: summarize_value(value, sample, include_output_data) for key, value in encoded.items()}
    stages = {
        "processed_image": summarize_tensor(
            processed,
            sample,
            include_data=include_stage_data,
        )
    }
    return {
        "family": "llava_next",
        "backend": backend,
        "class_name": processor_class_name(processor),
        "model_id": model_id,
        "image": image_summary(images[0]),
        "images": [image_summary(image) for image in images],
        "video_frames": [],
        "processor_config": llava_next_processor_config(processor),
        "metadata": llava_next_metadata(images, processor),
        "stages": stages,
        "outputs": outputs,
    }


def generate_llava_next_nhwc_case(
    model_id: str,
    images: list[Image.Image],
    sample: int,
    include_output_data: bool,
    include_stage_data: bool,
    backend: str,
) -> dict[str, Any]:
    processor = processor_class("llava_next_nhwc", backend)(
        size={"shortest_edge": 4},
        crop_size={"height": 4, "width": 4},
        image_grid_pinpoints=[[4, 4], [4, 8], [8, 4], [8, 8]],
        resample=0,
    )
    encoded = processor(
        images=images,
        return_tensors="np",
        data_format=ChannelDimension.LAST,
    )
    processed = processor(
        images=images,
        return_tensors="np",
        data_format=ChannelDimension.LAST,
        do_rescale=False,
        do_normalize=False,
    )["pixel_values"]
    outputs = {key: summarize_value(value, sample, include_output_data) for key, value in encoded.items()}
    stages = {
        "processed_image": summarize_tensor(
            processed,
            sample,
            include_data=include_stage_data,
        )
    }
    return {
        "family": "llava_next_nhwc",
        "backend": backend,
        "class_name": processor_class_name(processor),
        "model_id": model_id,
        "image": image_summary(images[0]),
        "images": [image_summary(image) for image in images],
        "video_frames": [],
        "processor_config": llava_next_processor_config(processor),
        "metadata": llava_next_metadata(images, processor),
        "stages": stages,
        "outputs": outputs,
    }


def size_pair(value: Any, default: tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, Mapping) or hasattr(value, "get"):
        if "height" in value and "width" in value:
            return int(value["height"]), int(value["width"])
        if "longest_edge" in value:
            edge = int(value["longest_edge"])
            return edge, edge
        if "shortest_edge" in value:
            edge = int(value["shortest_edge"])
            return edge, edge
        if "max_height" in value and "max_width" in value:
            return int(value["max_height"]), int(value["max_width"])
        return default
    if isinstance(value, (tuple, list)):
        return int(value[0]), int(value[1])
    if value is None:
        return default
    edge = int(value)
    return edge, edge


def pixtral_processor_config(processor: PixtralImageProcessor) -> dict[str, Any]:
    resample = getattr(processor.resample, "name", str(processor.resample))
    return {
        "size": processor.size,
        "patch_size": processor.patch_size,
        "resample": resample,
        "do_convert_rgb": processor.do_convert_rgb,
        "do_rescale": processor.do_rescale,
        "rescale_factor": processor.rescale_factor,
        "do_normalize": processor.do_normalize,
        "image_mean": processor.image_mean,
        "image_std": processor.image_std,
    }


def pixtral_metadata(images: list[Image.Image], processor: PixtralImageProcessor) -> dict[str, Any]:
    max_size = size_pair(processor.size, (1024, 1024))
    patch_size = size_pair(processor.patch_size, (16, 16))
    metadata: dict[str, Any] = {
        "original_sizes": [],
        "reshaped_input_sizes": [],
        "image_grid_thw": [],
        "image_patch_counts": [],
    }
    for image in images:
        original_size = (image.height, image.width)
        resized_size = pixtral_resize_output_size(
            np.asarray(image),
            max_size,
            patch_size,
            input_data_format=ChannelDimension.LAST,
        )
        patch_rows = ceil_div(int(resized_size[0]), patch_size[0])
        patch_columns = ceil_div(int(resized_size[1]), patch_size[1])
        metadata["original_sizes"].append([int(original_size[0]), int(original_size[1])])
        metadata["reshaped_input_sizes"].append([int(resized_size[0]), int(resized_size[1])])
        metadata["image_grid_thw"].append([1, patch_rows, patch_columns])
        metadata["image_patch_counts"].append(patch_rows * patch_columns)
    return metadata


def generate_pixtral_case(
    model_id: str,
    images: list[Image.Image],
    sample: int,
    include_output_data: bool,
    include_stage_data: bool,
    backend: str,
) -> dict[str, Any]:
    processor = processor_class("pixtral", backend)()
    encoded = processor(images=images, return_tensors="pt")
    processed = processor(
        images=images,
        return_tensors="np",
        do_rescale=False,
        do_normalize=False,
    )["pixel_values"]
    outputs = {key: summarize_value(value, sample, include_output_data) for key, value in encoded.items()}
    stages = {
        "processed_image": summarize_tensor(
            processed,
            sample,
            include_data=include_stage_data,
        )
    }
    return {
        "family": "pixtral",
        "backend": backend,
        "class_name": processor_class_name(processor),
        "model_id": model_id,
        "image": image_summary(images[0]),
        "images": [image_summary(image) for image in images],
        "video_frames": [],
        "processor_config": pixtral_processor_config(processor),
        "metadata": pixtral_metadata(images, processor),
        "stages": stages,
        "outputs": outputs,
    }


def idefics3_processor_config(processor: Idefics3ImageProcessor) -> dict[str, Any]:
    resample = getattr(processor.resample, "name", str(processor.resample))
    return {
        "size": processor.size,
        "max_image_size": processor.max_image_size,
        "resample": resample,
        "do_convert_rgb": processor.do_convert_rgb,
        "do_resize": processor.do_resize,
        "do_image_splitting": processor.do_image_splitting,
        "do_rescale": processor.do_rescale,
        "rescale_factor": processor.rescale_factor,
        "do_normalize": processor.do_normalize,
        "image_mean": processor.image_mean,
        "image_std": processor.image_std,
        "do_pad": processor.do_pad,
    }


def idefics3_resize_output_size(original_size: tuple[int, int], longest_edge: int) -> tuple[int, int]:
    height, width = original_size
    if width >= height:
        resized_width = longest_edge
        resized_height = int(height * longest_edge / width)
        if resized_height % 2 == 1:
            resized_height += 1
    else:
        resized_height = longest_edge
        resized_width = int(width * longest_edge / height)
        if resized_width % 2 == 1:
            resized_width += 1
    max_edge = max(resized_height, resized_width)
    if max_edge > 4096:
        scale = 4096 / max_edge
        resized_height = max(1, int(resized_height * scale))
        resized_width = max(1, int(resized_width * scale))
    return resized_height, resized_width


def idefics3_vision_encoder_size(resized_size: tuple[int, int], max_image_size: int) -> tuple[int, int]:
    height, width = resized_size
    aspect_ratio = width / height
    if width >= height:
        width = ceil_div(width, max_image_size) * max_image_size
        height = int(width / aspect_ratio)
        height = ceil_div(height, max_image_size) * max_image_size
    else:
        height = ceil_div(height, max_image_size) * max_image_size
        width = int(height * aspect_ratio)
        width = ceil_div(width, max_image_size) * max_image_size
    return height, width


def idefics3_metadata(images: list[Image.Image], processor: Idefics3ImageProcessor, encoded: Any) -> dict[str, Any]:
    longest_edge = int(processor.size["longest_edge"])
    max_image_edge = int(processor.max_image_size["longest_edge"])
    metadata: dict[str, Any] = {
        "original_sizes": [],
        "reshaped_input_sizes": [],
        "image_patch_counts": [],
        "rows": encoded["rows"],
        "cols": encoded["cols"],
    }
    for image, rows, cols in zip(images, encoded["rows"], encoded["cols"]):
        original_size = (image.height, image.width)
        if processor.do_resize:
            resized_size = idefics3_resize_output_size(original_size, longest_edge)
        else:
            resized_size = original_size
        if processor.do_image_splitting:
            reshaped_size = idefics3_vision_encoder_size(resized_size, max_image_edge)
        else:
            reshaped_size = (max_image_edge, max_image_edge)
        image_rows = int(rows[0])
        image_cols = int(cols[0])
        patch_count = image_rows * image_cols + 1 if image_rows and image_cols else 1
        metadata["original_sizes"].append([int(original_size[0]), int(original_size[1])])
        metadata["reshaped_input_sizes"].append([int(reshaped_size[0]), int(reshaped_size[1])])
        metadata["image_patch_counts"].append(patch_count)
    return metadata


def generate_idefics3_case(
    model_id: str,
    images: list[Image.Image],
    sample: int,
    include_output_data: bool,
    include_stage_data: bool,
    backend: str,
) -> dict[str, Any]:
    processor = processor_class("idefics3", backend)(
        size={"longest_edge": 10},
        max_image_size={"longest_edge": 5},
        resample=0,
    )
    nested_images = [[image] for image in images]
    encoded = processor(images=nested_images, return_tensors="np", return_row_col_info=True)
    processed = processor(
        images=nested_images,
        return_tensors="np",
        return_row_col_info=True,
        do_rescale=False,
        do_normalize=False,
    )["pixel_values"]
    outputs = {
        key: summarize_value(value, sample, include_output_data)
        for key, value in encoded.items()
        if key not in {"rows", "cols"}
    }
    stages = {
        "processed_image": summarize_tensor(
            processed,
            sample,
            include_data=include_stage_data,
        )
    }
    return {
        "family": "idefics3",
        "backend": backend,
        "class_name": processor_class_name(processor),
        "model_id": model_id,
        "image": image_summary(images[0]),
        "images": [image_summary(image) for image in images],
        "video_frames": [],
        "processor_config": idefics3_processor_config(processor),
        "metadata": idefics3_metadata(images, processor, encoded),
        "stages": stages,
        "outputs": outputs,
    }


def gemma3_processor_config(processor: Gemma3ImageProcessor) -> dict[str, Any]:
    resample = getattr(processor.resample, "name", str(processor.resample))
    return {
        "size": processor.size,
        "resample": resample,
        "do_convert_rgb": processor.do_convert_rgb,
        "do_resize": processor.do_resize,
        "do_pan_and_scan": processor.do_pan_and_scan,
        "pan_and_scan_min_crop_size": processor.pan_and_scan_min_crop_size,
        "pan_and_scan_max_num_crops": processor.pan_and_scan_max_num_crops,
        "pan_and_scan_min_ratio_to_activate": processor.pan_and_scan_min_ratio_to_activate,
        "do_rescale": processor.do_rescale,
        "rescale_factor": processor.rescale_factor,
        "do_normalize": processor.do_normalize,
        "image_mean": processor.image_mean,
        "image_std": processor.image_std,
    }


def gemma3_metadata(images: list[Image.Image], processor: Gemma3ImageProcessor, encoded: Any) -> dict[str, Any]:
    output_size = size_pair(processor.size, (224, 224))
    num_crops = [int(value) for value in encoded["num_crops"]]
    metadata: dict[str, Any] = {
        "original_sizes": [],
        "reshaped_input_sizes": [],
        "num_crops": num_crops,
    }
    for image, crop_count in zip(images, num_crops):
        metadata["original_sizes"].append([image.height, image.width])
        frame_count = 1 + crop_count
        if processor.do_resize:
            metadata["reshaped_input_sizes"].extend(
                [[int(output_size[0]), int(output_size[1])] for _ in range(frame_count)]
            )
        else:
            metadata["reshaped_input_sizes"].append([image.height, image.width])
            crops = processor.pan_and_scan(
                image=np.asarray(image),
                pan_and_scan_min_crop_size=processor.pan_and_scan_min_crop_size,
                pan_and_scan_max_num_crops=processor.pan_and_scan_max_num_crops,
                pan_and_scan_min_ratio_to_activate=processor.pan_and_scan_min_ratio_to_activate,
                input_data_format=ChannelDimension.LAST,
            )
            metadata["reshaped_input_sizes"].extend([[crop.shape[0], crop.shape[1]] for crop in crops])
    return metadata


def generate_gemma3_case(
    model_id: str,
    images: list[Image.Image],
    sample: int,
    include_output_data: bool,
    include_stage_data: bool,
    backend: str,
) -> dict[str, Any]:
    processor = processor_class("gemma3", backend)(
        size={"height": 5, "width": 5},
        resample=0,
        do_pan_and_scan=True,
        pan_and_scan_min_crop_size=5,
        pan_and_scan_max_num_crops=4,
        pan_and_scan_min_ratio_to_activate=1.2,
    )
    encoded = processor(images=images, return_tensors="np")
    processed = processor(
        images=images,
        return_tensors="np",
        do_rescale=False,
        do_normalize=False,
    )["pixel_values"]
    outputs = {
        key: summarize_value(value, sample, include_output_data)
        for key, value in encoded.items()
        if key != "num_crops"
    }
    stages = {
        "processed_image": summarize_tensor(
            processed,
            sample,
            include_data=include_stage_data,
        )
    }
    return {
        "family": "gemma3",
        "backend": backend,
        "class_name": processor_class_name(processor),
        "model_id": model_id,
        "image": image_summary(images[0]),
        "images": [image_summary(image) for image in images],
        "video_frames": [],
        "processor_config": gemma3_processor_config(processor),
        "metadata": gemma3_metadata(images, processor, encoded),
        "stages": stages,
        "outputs": outputs,
    }


def mllama_processor_config(processor: MllamaImageProcessor) -> dict[str, Any]:
    resample = getattr(processor.resample, "name", str(processor.resample))
    return {
        "size": processor.size,
        "resample": resample,
        "do_convert_rgb": processor.do_convert_rgb,
        "do_resize": processor.do_resize,
        "do_rescale": processor.do_rescale,
        "rescale_factor": processor.rescale_factor,
        "do_normalize": processor.do_normalize,
        "image_mean": processor.image_mean,
        "image_std": processor.image_std,
        "do_pad": processor.do_pad,
        "max_image_tiles": processor.max_image_tiles,
    }


def mllama_metadata(images: list[Image.Image], processor: MllamaImageProcessor, encoded: Any) -> dict[str, Any]:
    tile_size = int(processor.size["height"])
    metadata: dict[str, Any] = {
        "original_sizes": [],
        "reshaped_input_sizes": [],
        "canvas_sizes": [],
        "image_patch_counts": [],
        "num_tiles": encoded["num_tiles"],
        "aspect_ratio_ids": np.asarray(encoded["aspect_ratio_ids"]).tolist(),
        "aspect_ratio_mask": np.asarray(encoded["aspect_ratio_mask"]).astype(bool).tolist(),
    }
    for image in images:
        original_size = (image.height, image.width)
        canvas_size = mllama_optimal_tiled_canvas(
            image.height,
            image.width,
            processor.max_image_tiles,
            tile_size,
        )
        resized_size = mllama_image_size_fit_to_canvas(
            image.height,
            image.width,
            int(canvas_size[0]),
            int(canvas_size[1]),
            tile_size,
        )
        num_tiles = int(canvas_size[0] / tile_size) * int(canvas_size[1] / tile_size)
        metadata["original_sizes"].append([int(original_size[0]), int(original_size[1])])
        metadata["reshaped_input_sizes"].append([int(resized_size[0]), int(resized_size[1])])
        metadata["canvas_sizes"].append([int(canvas_size[0]), int(canvas_size[1])])
        metadata["image_patch_counts"].append(num_tiles)
    return metadata


def generate_mllama_case(
    model_id: str,
    images: list[Image.Image],
    sample: int,
    include_output_data: bool,
    include_stage_data: bool,
    backend: str,
) -> dict[str, Any]:
    processor = processor_class("mllama", backend)(
        size={"height": 5, "width": 5},
        max_image_tiles=4,
        resample=0,
    )
    nested_images = [[image] for image in images]
    encoded = processor(images=nested_images, return_tensors="np")
    processed = processor(
        images=nested_images,
        return_tensors="np",
        do_rescale=False,
        do_normalize=False,
    )["pixel_values"]
    outputs = {
        key: summarize_value(value, sample, include_output_data)
        for key, value in encoded.items()
        if key != "num_tiles"
    }
    stages = {
        "processed_image": summarize_tensor(
            processed,
            sample,
            include_data=include_stage_data,
        )
    }
    return {
        "family": "mllama",
        "backend": backend,
        "class_name": processor_class_name(processor),
        "model_id": model_id,
        "image": image_summary(images[0]),
        "images": [image_summary(image) for image in images],
        "video_frames": [],
        "processor_config": mllama_processor_config(processor),
        "metadata": mllama_metadata(images, processor, encoded),
        "stages": stages,
        "outputs": outputs,
    }


def donut_processor_config(processor: DonutImageProcessor) -> dict[str, Any]:
    resample = getattr(processor.resample, "name", str(processor.resample))
    return {
        "size": processor.size,
        "resample": resample,
        "do_resize": processor.do_resize,
        "do_thumbnail": processor.do_thumbnail,
        "do_align_long_axis": processor.do_align_long_axis,
        "do_pad": processor.do_pad,
        "do_rescale": processor.do_rescale,
        "rescale_factor": processor.rescale_factor,
        "do_normalize": processor.do_normalize,
        "image_mean": processor.image_mean,
        "image_std": processor.image_std,
    }


def generate_donut_case(
    model_id: str,
    images: list[Image.Image],
    sample: int,
    include_output_data: bool,
    include_stage_data: bool,
    backend: str,
) -> dict[str, Any]:
    processor = processor_class("donut", backend)(size={"height": 12, "width": 8})
    encoded = processor(images=images, return_tensors="np")
    processed = processor(
        images=images,
        return_tensors="np",
        do_rescale=False,
        do_normalize=False,
    )["pixel_values"]
    outputs = {key: summarize_value(value, sample, include_output_data) for key, value in encoded.items()}
    stages = {
        "processed_image": summarize_tensor(
            processed,
            sample,
            include_data=include_stage_data,
        )
    }
    return {
        "family": "donut",
        "backend": backend,
        "class_name": processor_class_name(processor),
        "model_id": model_id,
        "image": image_summary(images[0]),
        "images": [image_summary(image) for image in images],
        "video_frames": [],
        "processor_config": donut_processor_config(processor),
        "metadata": {
            "original_sizes": [[image.height, image.width] for image in images],
            "target_size": [int(processor.size["height"]), int(processor.size["width"])],
        },
        "stages": stages,
        "outputs": outputs,
    }


def videomae_processor_config(processor: VideoMAEImageProcessor) -> dict[str, Any]:
    resample = getattr(processor.resample, "name", str(processor.resample))
    return {
        "size": processor.size,
        "crop_size": processor.crop_size,
        "resample": resample,
        "do_resize": processor.do_resize,
        "do_center_crop": processor.do_center_crop,
        "do_rescale": processor.do_rescale,
        "rescale_factor": processor.rescale_factor,
        "do_normalize": processor.do_normalize,
        "image_mean": processor.image_mean,
        "image_std": processor.image_std,
    }


def generate_videomae_case(
    model_id: str,
    images: list[Image.Image],
    sample: int,
    include_output_data: bool,
    include_stage_data: bool,
    video_batch_size: int,
    backend: str,
) -> dict[str, Any]:
    processor = processor_class("videomae", backend)(
        size={"shortest_edge": 4},
        crop_size={"height": 4, "width": 4},
        resample=0,
    )
    videos = build_video_batch(images, video_batch_size)
    processor_input = images if video_batch_size == 1 else videos
    encoded = processor(images=processor_input, return_tensors="np")
    processed = processor(
        images=processor_input,
        return_tensors="np",
        do_rescale=False,
        do_normalize=False,
    )["pixel_values"]
    outputs = {key: summarize_value(value, sample, include_output_data) for key, value in encoded.items()}
    stages = {
        "processed_video": summarize_tensor(
            processed,
            sample,
            include_data=include_stage_data,
        )
    }
    return {
        "family": "videomae",
        "backend": backend,
        "class_name": processor_class_name(processor),
        "model_id": model_id,
        "image": image_summary(images[0]),
        "images": [image_summary(image) for image in images],
        "video_frames": [image_summary(image) for image in images],
        "video_batch_size": video_batch_size,
        "videos": [[image_summary(image) for image in video] for video in videos],
        "processor_config": videomae_processor_config(processor),
        "metadata": {
            "original_sizes": [[image.height, image.width] for image in images],
            "video_original_sizes": [
                [[image.height, image.width] for image in video] for video in videos
            ],
            "target_size": [int(processor.crop_size["height"]), int(processor.crop_size["width"])],
        },
        "stages": stages,
        "outputs": outputs,
    }


def vivit_processor_config(processor: VivitImageProcessor) -> dict[str, Any]:
    resample = getattr(processor.resample, "name", str(processor.resample))
    return {
        "size": processor.size,
        "crop_size": processor.crop_size,
        "resample": resample,
        "do_resize": processor.do_resize,
        "do_center_crop": processor.do_center_crop,
        "do_rescale": processor.do_rescale,
        "rescale_factor": processor.rescale_factor,
        "offset": processor.offset,
        "do_normalize": processor.do_normalize,
        "image_mean": processor.image_mean,
        "image_std": processor.image_std,
    }


def generate_vivit_case(
    model_id: str,
    images: list[Image.Image],
    sample: int,
    include_output_data: bool,
    include_stage_data: bool,
    video_batch_size: int,
    backend: str,
) -> dict[str, Any]:
    if backend != "pil":
        raise SystemExit("ViViT only exposes its PIL implementation at the pinned revision")
    processor = processor_class("vivit", backend)(
        size={"shortest_edge": 4},
        crop_size={"height": 4, "width": 4},
        resample=0,
    )
    videos = build_video_batch(images, video_batch_size)
    processor_input = images if video_batch_size == 1 else videos
    encoded = processor(images=processor_input, return_tensors="np")
    processed = processor(
        images=processor_input,
        return_tensors="np",
        do_rescale=False,
        offset=False,
        do_normalize=False,
    )["pixel_values"]
    outputs = {key: summarize_value(value, sample, include_output_data) for key, value in encoded.items()}
    stages = {
        "processed_video": summarize_tensor(
            processed,
            sample,
            include_data=include_stage_data,
        )
    }
    return {
        "family": "vivit",
        "backend": backend,
        "class_name": processor_class_name(processor),
        "model_id": model_id,
        "image": image_summary(images[0]),
        "images": [image_summary(image) for image in images],
        "video_frames": [image_summary(image) for image in images],
        "video_batch_size": video_batch_size,
        "videos": [[image_summary(image) for image in video] for video in videos],
        "processor_config": vivit_processor_config(processor),
        "metadata": {
            "original_sizes": [[image.height, image.width] for image in images],
            "video_original_sizes": [
                [[image.height, image.width] for image in video] for video in videos
            ],
            "target_size": [int(processor.crop_size["height"]), int(processor.crop_size["width"])],
        },
        "stages": stages,
        "outputs": outputs,
    }


def generate_case(
    family: str,
    model_id: str,
    images: list[Image.Image],
    sample: int,
    include_output_data: bool,
    include_stage_data: bool,
    backend: str,
    as_video: bool,
    video_batch_size: int,
) -> dict[str, Any]:
    if family == "qwen_vl" and as_video:
        processor = AutoVideoProcessor.from_pretrained(model_id)
        encoded = processor(videos=[images], return_tensors="pt")
        processed = processor(
            videos=[images],
            return_tensors="np",
            do_rescale=False,
            do_normalize=False,
        )["pixel_values_videos"]
        return {
            "family": family,
            "backend": "torchvision",
            "class_name": processor_class_name(processor),
            "model_id": model_id,
            "processor_config": effective_processor_config(processor),
            "image": image_summary(images[0]),
            "images": [image_summary(image) for image in images],
            "video_frames": [image_summary(image) for image in images],
            "stages": {
                "processed_video": summarize_tensor(
                    processed,
                    sample,
                    include_data=include_stage_data,
                )
            },
            "outputs": {
                key: summarize_value(value, sample, include_output_data)
                for key, value in encoded.items()
            },
        }
    if family == "llava_next":
        if as_video:
            raise SystemExit("--as-video is not supported for --family llava_next")
        return generate_llava_next_case(
            model_id,
            images,
            sample,
            include_output_data,
            include_stage_data,
            backend,
        )
    if family == "llava_next_nhwc":
        if as_video:
            raise SystemExit("--as-video is not supported for --family llava_next_nhwc")
        return generate_llava_next_nhwc_case(
            model_id,
            images,
            sample,
            include_output_data,
            include_stage_data,
            backend,
        )
    if family == "pixtral":
        if as_video:
            raise SystemExit("--as-video is not supported for --family pixtral")
        return generate_pixtral_case(
            model_id,
            images,
            sample,
            include_output_data,
            include_stage_data,
            backend,
        )
    if family == "idefics3":
        if as_video:
            raise SystemExit("--as-video is not supported for --family idefics3")
        return generate_idefics3_case(
            model_id,
            images,
            sample,
            include_output_data,
            include_stage_data,
            backend,
        )
    if family == "gemma3":
        if as_video:
            raise SystemExit("--as-video is not supported for --family gemma3")
        return generate_gemma3_case(
            model_id,
            images,
            sample,
            include_output_data,
            include_stage_data,
            backend,
        )
    if family == "mllama":
        if as_video:
            raise SystemExit("--as-video is not supported for --family mllama")
        return generate_mllama_case(
            model_id,
            images,
            sample,
            include_output_data,
            include_stage_data,
            backend,
        )
    if family == "donut":
        if as_video:
            raise SystemExit("--as-video is not supported for --family donut")
        return generate_donut_case(
            model_id,
            images,
            sample,
            include_output_data,
            include_stage_data,
            backend,
        )
    if family == "videomae":
        return generate_videomae_case(
            model_id,
            images,
            sample,
            include_output_data,
            include_stage_data,
            video_batch_size,
            backend,
        )
    if family == "vivit":
        return generate_vivit_case(
            model_id,
            images,
            sample,
            include_output_data,
            include_stage_data,
            video_batch_size,
            backend,
        )

    processor = AutoImageProcessor.from_pretrained(model_id, backend=backend)
    if as_video:
        encoded = processor(images=None, videos=[images], return_tensors="pt")
        stage_key = "processed_video"
        processed = processor(
            images=None,
            videos=[images],
            return_tensors="np",
            do_rescale=False,
            do_normalize=False,
        )["pixel_values_videos"]
    else:
        encoded = processor(images=images, return_tensors="pt")
        stage_key = "processed_image"
        processed = processor(
            images=images,
            return_tensors="np",
            do_rescale=False,
            do_normalize=False,
        )["pixel_values"]
    outputs = {key: summarize_value(value, sample, include_output_data) for key, value in encoded.items()}
    stages = {
        stage_key: summarize_tensor(
            processed,
            sample,
            include_data=include_stage_data,
        )
    }
    return {
        "family": family,
        "backend": backend,
        "class_name": processor_class_name(processor),
        "model_id": model_id,
        "processor_config": effective_processor_config(processor),
        "image": image_summary(images[0]),
        "images": [image_summary(image) for image in images],
        "video_frames": [image_summary(image) for image in images] if as_video else [],
        "stages": stages,
        "outputs": outputs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family",
        action="append",
        choices=sorted(DEFAULT_MODELS),
        help="Processor family to record. Repeat for multiple families.",
    )
    parser.add_argument("--model-id", help="Override model id when one family is selected.")
    parser.add_argument("--image", type=Path, help="Input image path. A deterministic image is generated by default.")
    parser.add_argument("--width", type=int, default=13, help="Generated image width.")
    parser.add_argument("--height", type=int, default=17, help="Generated image height.")
    parser.add_argument("--image-count", type=int, default=1, help="Number of deterministic generated images.")
    parser.add_argument("--width-step", type=int, default=0, help="Width increment for each generated image.")
    parser.add_argument("--height-step", type=int, default=0, help="Height increment for each generated image.")
    parser.add_argument("--as-video", action="store_true", help="Submit generated images as one video input.")
    parser.add_argument(
        "--video-batch-size",
        type=int,
        default=1,
        help="Number of deterministic same-length videos to record for VideoMAE/ViViT.",
    )
    parser.add_argument("--sample", type=int, default=16, help="Number of flattened values to record per tensor.")
    parser.add_argument(
        "--use-fast",
        choices=("true", "false"),
        help="Deprecated compatibility alias for --backend torchvision/pil.",
    )
    parser.add_argument(
        "--backend",
        action="append",
        choices=("pil", "torchvision"),
        help="Backend to record. Repeat to record both; both are generated by default.",
    )
    parser.add_argument(
        "--include-output-data",
        action="store_true",
        help="Store full output tensors as base64 for Rust full-value parity assertions.",
    )
    parser.add_argument(
        "--include-stage-data",
        action="store_true",
        help="Store full intermediate stage tensors as base64 for Rust diff diagnostics.",
    )
    parser.add_argument("--output", type=Path, required=True, help="JSON fixture output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verify_fixture_upstream()
    families = args.family or ["clip", "vit"]
    if args.model_id is not None and len(families) != 1:
        raise SystemExit("--model-id requires exactly one --family")
    video_families = {"qwen_vl", "videomae", "vivit"}
    if args.as_video and any(family not in video_families for family in families):
        raise SystemExit("--as-video is currently supported only for --family qwen_vl, videomae, or vivit")
    video_batch_families = {"videomae", "vivit"}
    if args.video_batch_size < 1:
        raise SystemExit("--video-batch-size must be at least 1")
    if args.video_batch_size > 1:
        if args.as_video:
            raise SystemExit("--video-batch-size cannot be combined with --as-video")
        if args.image is not None:
            raise SystemExit("--video-batch-size cannot be combined with --image")
        if any(family not in video_batch_families for family in families):
            raise SystemExit("--video-batch-size is currently supported only for --family videomae or vivit")

    images = load_or_generate_images(
        args.image,
        args.width,
        args.height,
        args.image_count,
        args.width_step,
        args.height_step,
    )
    if args.backend is not None and args.use_fast is not None:
        raise SystemExit("--backend cannot be combined with deprecated --use-fast")
    if args.backend is not None:
        backends = list(dict.fromkeys(args.backend))
    elif args.use_fast is not None:
        backends = ["torchvision" if args.use_fast == "true" else "pil"]
    else:
        backends = ["pil", "torchvision"]
    cases = []
    for family in families:
        model_id = args.model_id or DEFAULT_MODELS[family]
        if family == "vivit":
            family_backends = ["pil"]
        elif family == "qwen_vl" and args.as_video:
            family_backends = ["torchvision"]
        else:
            family_backends = backends
        for backend in family_backends:
            case = generate_case(
                family,
                model_id,
                images,
                args.sample,
                args.include_output_data,
                args.include_stage_data,
                backend,
                args.as_video,
                args.video_batch_size,
            )
            case["comparison"] = comparison_policy(family)
            cases.append(case)

    fixture = {
        "schema": "image-processors.transformers-parity.v1",
        "generator": "scripts/parity/transformers_image_parity.py",
        "upstream": {
            "library": "transformers",
            "version": transformers.__version__,
            "source": "transformers.image_processing",
            "commit": TRANSFORMERS_FIXTURE_COMMIT,
        },
        "input": input_contract(args, images),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(json_compatible(fixture), indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
