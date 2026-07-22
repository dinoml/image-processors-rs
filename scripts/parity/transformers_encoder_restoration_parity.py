#!/usr/bin/env python3
"""Generate compact full-payload encoder and Swin2SR parity fixtures.

The generator imports Transformers from an explicit official source checkout
at the audited commit. It never downloads model weights or processor configs.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from _fixture_contract import acceptance_contract, verify_clean_git_checkout


AUDIT_COMMIT = "6d960ca0a0eba0d2aebc920d8080a9353da468d3"
SOURCE_MANIFEST_SHA256 = "8a994581e9fa2d8e134dddafcb6f884afdcd8d198b48831db8ea5624bd8acccc"
SCHEMA = "image-processors.transformers-encoder-restoration-parity.v1"
IMAGE_WIDTH = 17
IMAGE_HEIGHT = 13
CLUSTERS = [[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]


@dataclass(frozen=True)
class ProcessorSpec:
    canonical_class: str
    aliases: tuple[str, ...]
    recipe_id: str
    kwargs: dict[str, Any]
    source_path: str


def square(edge: int) -> dict[str, int]:
    return {"height": edge, "width": edge}


SPECS = (
    ProcessorSpec(
        "BeitImageProcessor",
        ("BeitImageProcessorPil",),
        "transformers.beit_image_processor",
        {"size": {"height": 9, "width": 11}},
        "src/transformers/models/beit/image_processing_beit.py",
    ),
    ProcessorSpec(
        "BitImageProcessor",
        ("BitImageProcessorPil",),
        "transformers.bit_image_processor",
        {"size": {"shortest_edge": 9}, "crop_size": {"height": 7, "width": 9}},
        "src/transformers/models/bit/image_processing_bit.py",
    ),
    ProcessorSpec(
        "ChineseCLIPImageProcessor",
        ("ChineseCLIPImageProcessorPil",),
        "transformers.chinese_clip_image_processor",
        {"size": {"shortest_edge": 9}, "crop_size": {"height": 7, "width": 9}},
        "src/transformers/models/chinese_clip/image_processing_chinese_clip.py",
    ),
    ProcessorSpec(
        "ConvNextImageProcessor",
        ("ConvNextImageProcessorPil",),
        "transformers.convnext_image_processor",
        {"size": {"shortest_edge": 9}, "crop_pct": 1.0},
        "src/transformers/models/convnext/image_processing_convnext.py",
    ),
    ProcessorSpec(
        "DINOv3ViTImageProcessor",
        (),
        "transformers.dinov3_vit_image_processor",
        {"size": {"height": 9, "width": 11}},
        "src/transformers/models/dinov3_vit/image_processing_dinov3_vit.py",
    ),
    ProcessorSpec(
        "DeiTImageProcessor",
        ("DeiTImageProcessorPil",),
        "transformers.deit_image_processor",
        {"size": {"height": 9, "width": 11}, "crop_size": {"height": 7, "width": 9}},
        "src/transformers/models/deit/image_processing_deit.py",
    ),
    ProcessorSpec(
        "EfficientNetImageProcessor",
        ("EfficientNetImageProcessorPil",),
        "transformers.efficientnet_image_processor",
        {"size": {"height": 9, "width": 11}, "include_top": True},
        "src/transformers/models/efficientnet/image_processing_efficientnet.py",
    ),
    ProcessorSpec(
        "FlavaImageProcessor",
        ("FlavaImageProcessorPil",),
        "transformers.flava_image_processor",
        {
            "size": {"height": 9, "width": 11},
            "crop_size": {"height": 7, "width": 9},
            "return_codebook_pixels": False,
            "return_image_mask": False,
        },
        "src/transformers/models/flava/image_processing_flava.py",
    ),
    ProcessorSpec(
        "ImageGPTImageProcessor",
        ("ImageGPTImageProcessorPil",),
        "transformers.imagegpt_image_processor",
        {"size": {"height": 9, "width": 11}, "clusters": CLUSTERS, "do_color_quantize": True},
        "src/transformers/models/imagegpt/image_processing_imagegpt.py",
    ),
    ProcessorSpec(
        "LevitImageProcessor",
        ("LevitImageProcessorPil",),
        "transformers.levit_image_processor",
        {"size": {"shortest_edge": 8}, "crop_size": {"height": 7, "width": 9}},
        "src/transformers/models/levit/image_processing_levit.py",
    ),
    ProcessorSpec(
        "MobileNetV1ImageProcessor",
        ("MobileNetV1ImageProcessorPil",),
        "transformers.mobilenet_v1_image_processor",
        {"size": {"shortest_edge": 9}, "crop_size": {"height": 7, "width": 9}},
        "src/transformers/models/mobilenet_v1/image_processing_mobilenet_v1.py",
    ),
    ProcessorSpec(
        "MobileNetV2ImageProcessor",
        ("MobileNetV2ImageProcessorPil",),
        "transformers.mobilenet_v2_image_processor",
        {"size": {"shortest_edge": 9}, "crop_size": {"height": 7, "width": 9}},
        "src/transformers/models/mobilenet_v2/image_processing_mobilenet_v2.py",
    ),
    ProcessorSpec(
        "MobileViTImageProcessor",
        ("MobileViTImageProcessorPil",),
        "transformers.mobilevit_image_processor",
        {"size": {"shortest_edge": 9}, "crop_size": {"height": 13, "width": 9}},
        "src/transformers/models/mobilevit/image_processing_mobilevit.py",
    ),
    ProcessorSpec(
        "PPLCNetImageProcessor",
        (),
        "transformers.pp_lcnet_image_processor",
        {"resize_short": 9, "size_divisor": 1, "crop_size": {"height": 7, "width": 9}},
        "src/transformers/models/pp_lcnet/image_processing_pp_lcnet.py",
    ),
    ProcessorSpec(
        "PerceiverImageProcessor",
        ("PerceiverImageProcessorPil",),
        "transformers.perceiver_image_processor",
        {"size": {"height": 7, "width": 9}, "crop_size": {"height": 9, "width": 9}},
        "src/transformers/models/perceiver/image_processing_perceiver.py",
    ),
    ProcessorSpec(
        "PoolFormerImageProcessor",
        ("PoolFormerImageProcessorPil",),
        "transformers.poolformer_image_processor",
        {"size": {"shortest_edge": 9}, "crop_size": square(9), "crop_pct": 1.0},
        "src/transformers/models/poolformer/image_processing_poolformer.py",
    ),
    ProcessorSpec(
        "PvtImageProcessor",
        ("PvtImageProcessorPil",),
        "transformers.pvt_image_processor",
        {"size": {"height": 9, "width": 11}},
        "src/transformers/models/pvt/image_processing_pvt.py",
    ),
    ProcessorSpec(
        "SiglipImageProcessor",
        ("SiglipImageProcessorPil",),
        "transformers.siglip_image_processor",
        {"size": {"height": 9, "width": 11}},
        "src/transformers/models/siglip/image_processing_siglip.py",
    ),
    ProcessorSpec(
        "TimmWrapperImageProcessor",
        (),
        "transformers.timm_wrapper_image_processor",
        {
            "pretrained_cfg": {
                "input_size": [3, 9, 9],
                "crop_pct": 1.0,
                "interpolation": "bicubic",
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            }
        },
        "src/transformers/models/timm_wrapper/image_processing_timm_wrapper.py",
    ),
    ProcessorSpec(
        "Swin2SRImageProcessor",
        ("Swin2SRImageProcessorPil",),
        "transformers.swin2sr_image_processor",
        {"size_divisor": 4},
        "src/transformers/models/swin2sr/image_processing_swin2sr.py",
    ),
)

PIL_SOURCE_PATHS = (
    "src/transformers/models/beit/image_processing_pil_beit.py",
    "src/transformers/models/bit/image_processing_pil_bit.py",
    "src/transformers/models/chinese_clip/image_processing_pil_chinese_clip.py",
    "src/transformers/models/convnext/image_processing_pil_convnext.py",
    "src/transformers/models/deit/image_processing_pil_deit.py",
    "src/transformers/models/efficientnet/image_processing_pil_efficientnet.py",
    "src/transformers/models/flava/image_processing_pil_flava.py",
    "src/transformers/models/imagegpt/image_processing_pil_imagegpt.py",
    "src/transformers/models/levit/image_processing_pil_levit.py",
    "src/transformers/models/mobilenet_v1/image_processing_pil_mobilenet_v1.py",
    "src/transformers/models/mobilenet_v2/image_processing_pil_mobilenet_v2.py",
    "src/transformers/models/mobilevit/image_processing_pil_mobilevit.py",
    "src/transformers/models/perceiver/image_processing_pil_perceiver.py",
    "src/transformers/models/poolformer/image_processing_pil_poolformer.py",
    "src/transformers/models/pvt/image_processing_pil_pvt.py",
    "src/transformers/models/siglip/image_processing_pil_siglip.py",
    "src/transformers/models/swin2sr/image_processing_pil_swin2sr.py",
)

# Shared implementation modules are part of the audit boundary because model
# processors delegate resizing, normalization, tensor batching, and shape
# grouping to them. Hashing only the thin model modules would not bind the
# fixture to the code that performs most preprocessing operations.
COMMON_SOURCE_PATHS = (
    "src/transformers/feature_extraction_utils.py",
    "src/transformers/image_processing_backends.py",
    "src/transformers/image_processing_outputs.py",
    "src/transformers/image_processing_utils.py",
    "src/transformers/image_transforms.py",
    "src/transformers/image_utils.py",
    "src/transformers/processing_utils.py",
    "src/transformers/utils/generic.py",
    "src/transformers/utils/import_utils.py",
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transformers-source",
        type=Path,
        default=os.environ.get("TRANSFORMERS_SOURCE"),
        required=os.environ.get("TRANSFORMERS_SOURCE") is None,
        help="Official huggingface/transformers Git checkout at the audited commit",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root
        / "image-processors/tests/fixtures/transformers/catalog_encoder_restoration.json",
    )
    return parser.parse_args()


def import_audited_transformers(source: Path):
    source = verify_clean_git_checkout(
        source, expected_commit=AUDIT_COMMIT, project="Transformers"
    )
    package_root = source / "src"
    if not (package_root / "transformers/__init__.py").is_file():
        raise SystemExit(f"not a Transformers source checkout: {source}")
    source_paths = audited_source_paths()
    missing = [path for path in source_paths if not (source / path).is_file()]
    if missing:
        raise SystemExit(f"audited source files are missing: {missing}")
    digest = source_manifest_digest(source, source_paths)
    if digest != SOURCE_MANIFEST_SHA256:
        raise SystemExit(
            "Transformers source manifest does not match audited commit "
            f"{AUDIT_COMMIT}: expected {SOURCE_MANIFEST_SHA256}, got {digest}"
        )
    sys.path.insert(0, str(package_root))
    import transformers  # pylint: disable=import-outside-toplevel

    imported = Path(transformers.__file__).resolve()
    if source not in imported.parents:
        raise SystemExit(f"imported Transformers from {imported}, expected {source}")
    return transformers


def audited_source_paths() -> list[str]:
    return sorted(
        {spec.source_path for spec in SPECS}
        | set(PIL_SOURCE_PATHS)
        | set(COMMON_SOURCE_PATHS)
    )


def source_manifest_digest(source: Path, source_paths: list[str]) -> str:
    digest = hashlib.sha256()
    for path in source_paths:
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
        digest.update((source / path).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def deterministic_image() -> tuple[Image.Image, np.ndarray]:
    values = np.fromiter(
        ((index * 37 + 17) % 256 for index in range(IMAGE_WIDTH * IMAGE_HEIGHT * 3)),
        dtype=np.uint8,
    ).reshape(IMAGE_HEIGHT, IMAGE_WIDTH, 3)
    return Image.fromarray(values, mode="RGB"), values


def tensor_blob(tensor) -> dict[str, Any]:
    import torch  # pylint: disable=import-outside-toplevel

    tensor = tensor.detach().cpu().contiguous()
    dtype_names = {
        torch.float32: "float32",
        torch.int64: "int64",
        torch.uint8: "uint8",
    }
    if tensor.dtype not in dtype_names:
        raise TypeError(f"unsupported fixture dtype: {tensor.dtype}")
    array = tensor.numpy()
    return {
        "shape": list(array.shape),
        "dtype": dtype_names[tensor.dtype],
        "data": {
            "encoding": "base64",
            "byte_order": "little",
            "value": base64.b64encode(array.tobytes(order="C")).decode("ascii"),
        },
    }


def output_tensor(batch, preferred_name: str | None = None):
    if preferred_name is not None:
        return preferred_name, batch[preferred_name]
    names = list(batch.keys())
    if len(names) != 1:
        raise ValueError(f"expected one tensor output, got {names}")
    return names[0], batch[names[0]]


def stage_kwargs(spec: ProcessorSpec) -> dict[str, Any]:
    kwargs = json.loads(json.dumps(spec.kwargs))
    # DINOv3 rescales before interpolation, so its pre-normalization stage must
    # retain rescaling to preserve the upstream operation order.
    if spec.canonical_class != "DINOv3ViTImageProcessor":
        kwargs["do_rescale"] = False
    kwargs["do_normalize"] = False
    if spec.canonical_class == "EfficientNetImageProcessor":
        kwargs["include_top"] = False
    if spec.canonical_class == "ImageGPTImageProcessor":
        kwargs["do_color_quantize"] = False
    return kwargs


def timm_stage_kwargs(spec: ProcessorSpec) -> dict[str, Any]:
    kwargs = json.loads(json.dumps(spec.kwargs))
    kwargs["pretrained_cfg"]["mean"] = [0.0, 0.0, 0.0]
    kwargs["pretrained_cfg"]["std"] = [1.0, 1.0, 1.0]
    return kwargs


def generate_case(transformers, spec: ProcessorSpec, class_name: str, image: Image.Image) -> dict[str, Any]:
    processor_class = getattr(transformers, class_name)
    processor = processor_class(**spec.kwargs)
    output = processor.preprocess(image, return_tensors="pt")
    output_name, output_value = output_tensor(output)

    if spec.canonical_class == "TimmWrapperImageProcessor":
        stage_processor = processor_class(**timm_stage_kwargs(spec))
    else:
        stage_processor = processor_class(**stage_kwargs(spec))
    stage = stage_processor.preprocess(image, return_tensors="pt")
    _, stage_value = output_tensor(stage, "pixel_values")

    return {
        "class_name": class_name,
        "canonical_class": spec.canonical_class,
        "is_alias": class_name != spec.canonical_class,
        "recipe_id": spec.recipe_id,
        "processor_config": spec.kwargs,
        "stages": {"processed_image": tensor_blob(stage_value)},
        "outputs": {output_name: tensor_blob(output_value)},
    }


def main() -> None:
    args = parse_args()
    transformers = import_audited_transformers(args.transformers_source)
    image, image_values = deterministic_image()

    cases = []
    for spec in SPECS:
        for class_name in (spec.canonical_class, *spec.aliases):
            cases.append(generate_case(transformers, spec, class_name, image))

    import PIL  # pylint: disable=import-outside-toplevel
    import timm  # pylint: disable=import-outside-toplevel
    import torch  # pylint: disable=import-outside-toplevel
    import torchvision  # pylint: disable=import-outside-toplevel

    fixture = {
        "schema": SCHEMA,
        "generator": "scripts/parity/transformers_encoder_restoration_parity.py",
        "upstream": {
            "library": "transformers",
            "repository": "https://github.com/huggingface/transformers",
            "commit": AUDIT_COMMIT,
            "version": transformers.__version__,
            "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
            "source_files": audited_source_paths(),
        },
        "generator_versions": {
            "numpy": np.__version__,
            "pillow": PIL.__version__,
            "timm": timm.__version__,
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
        },
        "acceptance": acceptance_contract(
            input_locations=["image"],
            input_description=(
                "Embedded RGB8 bytes use value[index] = (index * 37 + 17) % 256."
            ),
            config_locations=["cases[].processor_config"],
            absolute_tolerance=1.0e-5,
        ),
        "image": {
            "mode": "RGB",
            "width": IMAGE_WIDTH,
            "height": IMAGE_HEIGHT,
            "data": {
                "encoding": "base64",
                "dtype": "uint8",
                "value": base64.b64encode(image_values.tobytes(order="C")).decode("ascii"),
            },
        },
        "cases": cases,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {len(cases)} full-payload cases to {args.output}")


if __name__ == "__main__":
    main()
