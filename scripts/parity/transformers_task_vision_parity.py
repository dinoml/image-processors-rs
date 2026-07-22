#!/usr/bin/env python3
"""Generate full-value task-vision fixtures from an audited Transformers checkout."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image

from _fixture_contract import acceptance_contract, verify_clean_git_checkout


AUDIT_COMMIT = "6d960ca0a0eba0d2aebc920d8080a9353da468d3"
SCHEMA = "image-processors.transformers-task-vision-parity.v3"
SOURCE_URL = f"https://github.com/huggingface/transformers/tree/{AUDIT_COMMIT}"

PIL_ALIAS_CLASSES = {
    "ConditionalDetrImageProcessor": "ConditionalDetrImageProcessorPil",
    "DeformableDetrImageProcessor": "DeformableDetrImageProcessorPil",
    "GroundingDinoImageProcessor": "GroundingDinoImageProcessorPil",
    "OwlViTImageProcessor": "OwlViTImageProcessorPil",
    "Owlv2ImageProcessor": "Owlv2ImageProcessorPil",
    "RTDetrImageProcessor": "RTDetrImageProcessorPil",
    "YolosImageProcessor": "YolosImageProcessorPil",
    "EomtImageProcessor": "EomtImageProcessorPil",
    "Mask2FormerImageProcessor": "Mask2FormerImageProcessorPil",
    "MaskFormerImageProcessor": "MaskFormerImageProcessorPil",
    "OneFormerImageProcessor": "OneFormerImageProcessorPil",
    "SegGptImageProcessor": "SegGptImageProcessorPil",
    "SegformerImageProcessor": "SegformerImageProcessorPil",
    "VitMatteImageProcessor": "VitMatteImageProcessorPil",
    "DPTImageProcessor": "DPTImageProcessorPil",
    "GLPNImageProcessor": "GLPNImageProcessorPil",
    "PromptDepthAnythingImageProcessor": "PromptDepthAnythingImageProcessorPil",
    "ZoeDepthImageProcessor": "ZoeDepthImageProcessorPil",
    "EfficientLoFTRImageProcessor": "EfficientLoFTRImageProcessorPil",
    "LightGlueImageProcessor": "LightGlueImageProcessorPil",
    "SuperGlueImageProcessor": "SuperGlueImageProcessorPil",
    "SuperPointImageProcessor": "SuperPointImageProcessorPil",
    "VitPoseImageProcessor": "VitPoseImageProcessorPil",
}


@dataclass(frozen=True)
class CaseSpec:
    class_name: str
    preset: str
    recipe_id: str
    family: str
    input_kind: str = "single"
    width: int = 6
    height: int = 4
    constructor: dict[str, Any] | None = None
    call: dict[str, Any] | None = None
    rust: dict[str, Any] | None = None


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
STANDARD_MEAN = [0.5, 0.5, 0.5]
STANDARD_STD = [0.5, 0.5, 0.5]
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def rust_config(
    preset: str,
    resize: dict[str, Any],
    *,
    mean: list[float] | None = IMAGENET_MEAN,
    std: list[float] | None = IMAGENET_STD,
    resample: str = "Bilinear",
    pre_padding: dict[str, Any] | None = None,
    post_padding: dict[str, Any] | None = None,
    color_mode: str = "rgb",
    reverse_channels: bool = False,
    emit_pixel_mask: bool = False,
    emit_original_sizes: bool = False,
    emit_reshaped_input_sizes: bool = False,
) -> dict[str, Any]:
    return {
        "preset": preset,
        "resize": resize,
        "pre_resize_padding": pre_padding or {"kind": "none"},
        "post_resize_padding": post_padding or {"kind": "none"},
        "color_mode": color_mode,
        "reverse_channels": reverse_channels,
        "resample": resample,
        "resize_parity": "Torchvision",
        "do_rescale": True,
        "rescale_factor": 1.0 / 255.0,
        "do_normalize": mean is not None,
        "image_mean": mean or [],
        "image_std": std or [],
        "output_layout": "ChannelsHeightWidth",
        "emit_pixel_mask": emit_pixel_mask,
        "emit_original_sizes": emit_original_sizes,
        "emit_reshaped_input_sizes": emit_reshaped_input_sizes,
    }


def fixed(height: int, width: int) -> dict[str, Any]:
    return {"kind": "fixed", "size": {"height": height, "width": width}}


def shortest(shortest_edge: int, longest_edge: int) -> dict[str, Any]:
    return {
        "kind": "shortest_edge",
        "shortest_edge": shortest_edge,
        "longest_edge": longest_edge,
    }


def cases() -> list[CaseSpec]:
    def detr_rust(preset: str) -> dict[str, Any]:
        return rust_config(
            preset,
            shortest(4, 6),
            emit_pixel_mask=True,
            emit_original_sizes=True,
            emit_reshaped_input_sizes=True,
        )

    def segmentation_short(preset: str, multiple: int | None = None) -> dict[str, Any]:
        return rust_config(
            preset,
            {**shortest(4, 6), **({"multiple": multiple} if multiple else {})},
            emit_pixel_mask=multiple is not None,
        )

    def fixed_imagenet(preset: str) -> dict[str, Any]:
        return rust_config(preset, fixed(4, 6))

    return [
        CaseSpec("ConditionalDetrImageProcessor", "conditional_detr", "transformers.conditional_detr_image_processor", "detection_grounding", width=5, constructor={"size": {"shortest_edge": 4, "longest_edge": 6}}, rust=detr_rust("conditional_detr")),
        CaseSpec("DeformableDetrImageProcessor", "deformable_detr", "transformers.conditional_detr_image_processor", "detection_grounding", width=5, constructor={"size": {"shortest_edge": 4, "longest_edge": 6}}, rust=detr_rust("deformable_detr")),
        CaseSpec("GroundingDinoImageProcessor", "grounding_dino", "transformers.grounding_dino_image_processor", "detection_grounding", width=5, constructor={"size": {"shortest_edge": 4, "longest_edge": 6}}, rust=detr_rust("grounding_dino")),
        CaseSpec("OwlViTImageProcessor", "owl_vit", "transformers.owlvit_image_processor", "detection_grounding", constructor={"size": {"height": 4, "width": 6}}, rust=rust_config("owl_vit", fixed(4, 6), mean=CLIP_MEAN, std=CLIP_STD, resample="Bicubic")),
        CaseSpec("Owlv2ImageProcessor", "owlv2", "transformers.owlv2_image_processor", "detection_grounding", constructor={"size": {"height": 4, "width": 6}, "do_pad": False}, rust=rust_config("owlv2", fixed(4, 6), mean=CLIP_MEAN, std=CLIP_STD)),
        CaseSpec("PPDocLayoutV2ImageProcessor", "pp_doc_layout_v2", "transformers.pp_doclayout_image_processor", "detection_grounding", constructor={"size": {"height": 4, "width": 6}}, rust=rust_config("pp_doc_layout_v2", fixed(4, 6), mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], resample="Bicubic")),
        CaseSpec("PPDocLayoutV3ImageProcessor", "pp_doc_layout_v3", "transformers.pp_doclayout_image_processor", "detection_grounding", constructor={"size": {"height": 4, "width": 6}}, rust=rust_config("pp_doc_layout_v3", fixed(4, 6), mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], resample="Bicubic")),
        CaseSpec("PPOCRV5ServerDetImageProcessor", "pp_ocr_v5_server_det", "transformers.pp_ocr_detection_image_processor", "detection_grounding", width=32, height=32, constructor={"limit_side_len": 64, "max_side_limit": 128}, rust=rust_config("pp_ocr_v5_server_det", {"kind": "paddle_detection", "limit_side_len": 64, "limit_type": "max", "max_side_limit": 128}, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], reverse_channels=True, emit_original_sizes=True)),
        CaseSpec("RTDetrImageProcessor", "rt_detr", "transformers.rt_detr_image_processor", "detection_grounding", constructor={"size": {"height": 4, "width": 6}}, rust=rust_config("rt_detr", fixed(4, 6), mean=None, std=None)),
        CaseSpec("RfDetrImageProcessor", "rf_detr", "transformers.rf_detr_image_processor", "detection_grounding", width=5, constructor={"size": {"shortest_edge": 4, "longest_edge": 6}}, rust=detr_rust("rf_detr")),
        CaseSpec("YolosImageProcessor", "yolos", "transformers.yolos_image_processor", "detection_grounding", width=5, constructor={"size": {"height": 4, "width": 5}}, rust=rust_config("yolos", fixed(4, 5), emit_pixel_mask=True, emit_original_sizes=True, emit_reshaped_input_sizes=True)),
        CaseSpec("EomtImageProcessor", "eomt", "transformers.eomt_image_processor", "segmentation", width=5, constructor={"size": {"shortest_edge": 4, "longest_edge": 6}}, rust=segmentation_short("eomt")),
        CaseSpec("Mask2FormerImageProcessor", "mask2_former", "transformers.maskformer_image_processor", "segmentation", width=5, constructor={"size": {"shortest_edge": 4, "longest_edge": 6}, "size_divisor": 2}, rust=segmentation_short("mask2_former", 2)),
        CaseSpec("MaskFormerImageProcessor", "mask_former", "transformers.maskformer_image_processor", "segmentation", width=5, constructor={"size": {"shortest_edge": 4, "longest_edge": 6}, "size_divisor": 2}, rust=segmentation_short("mask_former", 2)),
        CaseSpec("OneFormerImageProcessor", "one_former", "transformers.oneformer_image_processor", "segmentation", width=5, constructor={"size": {"shortest_edge": 4, "longest_edge": 6}}, rust=rust_config("one_former", shortest(4, 6), emit_pixel_mask=True)),
        CaseSpec("Sam2ImageProcessor", "sam2", "transformers.sam2_image_processor", "segmentation", constructor={"size": {"height": 4, "width": 6}}, rust=rust_config("sam2", fixed(4, 6), emit_original_sizes=True)),
        CaseSpec("Sam3ImageProcessor", "sam3", "transformers.sam3_image_processor", "segmentation", constructor={"size": {"height": 4, "width": 6}}, rust=rust_config("sam3", fixed(4, 6), mean=STANDARD_MEAN, std=STANDARD_STD, emit_original_sizes=True)),
        CaseSpec("Sapiens2ImageProcessor", "sapiens2", "transformers.sapiens2_image_processor", "segmentation", constructor={"size": {"height": 4, "width": 6}}, rust=fixed_imagenet("sapiens2")),
        CaseSpec("SegGptImageProcessor", "seg_gpt", "transformers.seggpt_image_processor", "segmentation", constructor={"size": {"height": 4, "width": 6}}, rust=rust_config("seg_gpt", fixed(4, 6), resample="Bicubic")),
        CaseSpec("SegformerImageProcessor", "segformer", "transformers.segformer_image_processor", "segmentation", constructor={"size": {"height": 4, "width": 6}}, rust=fixed_imagenet("segformer")),
        CaseSpec("VitMatteImageProcessor", "vit_matte", "transformers.vitmatte_image_processor", "segmentation", input_kind="matte", width=7, height=5, constructor={"size_divisor": 2}, rust=rust_config("vit_matte", {"kind": "none"}, mean=STANDARD_MEAN, std=STANDARD_STD, post_padding={"kind": "bottom_right_to_multiple", "multiple": 2})),
        CaseSpec("CHMv2ImageProcessor", "chm_v2", "transformers.chmv2_image_processor", "depth_geometry", constructor={"do_resize": False, "size_divisor": 2}, rust=rust_config("chm_v2", {"kind": "none"}, mean=[0.42, 0.411, 0.296], std=[0.213, 0.156, 0.143], resample="Bicubic", post_padding={"kind": "center_to_multiple", "multiple": 2})),
        CaseSpec("DPTImageProcessor", "dpt", "transformers.dpt_image_processor", "depth_geometry", constructor={"size": {"height": 4, "width": 6}}, rust=rust_config("dpt", fixed(4, 6), mean=STANDARD_MEAN, std=STANDARD_STD, resample="Bicubic")),
        CaseSpec("DepthProImageProcessor", "depth_pro", "transformers.depth_pro_image_processor", "depth_geometry", constructor={"size": {"height": 4, "width": 6}}, rust=rust_config("depth_pro", fixed(4, 6), mean=STANDARD_MEAN, std=STANDARD_STD)),
        CaseSpec("GLPNImageProcessor", "glpn", "transformers.glpn_image_processor", "depth_geometry", constructor={"size_divisor": 2}, rust=rust_config("glpn", {"kind": "round_down_to_multiple", "multiple": 2}, mean=None, std=None)),
        CaseSpec("PromptDepthAnythingImageProcessor", "prompt_depth_anything", "transformers.dpt_image_processor", "depth_geometry", constructor={"size": {"height": 4, "width": 6}}, rust=rust_config("prompt_depth_anything", fixed(4, 6), mean=STANDARD_MEAN, std=STANDARD_STD, resample="Bicubic")),
        CaseSpec("Tipsv2DptImageProcessor", "tips_v2_dpt", "transformers.tipsv2_image_processor", "depth_geometry", constructor={"size": {"height": 4, "width": 6}}, rust=rust_config("tips_v2_dpt", fixed(4, 6), mean=None, std=None)),
        CaseSpec("Tipsv2ImageProcessor", "tips_v2", "transformers.tipsv2_image_processor", "depth_geometry", constructor={"size": {"height": 4, "width": 6}}, rust=rust_config("tips_v2", fixed(4, 6), mean=None, std=None)),
        CaseSpec("ZoeDepthImageProcessor", "zoe_depth", "transformers.zoedepth_image_processor", "depth_geometry", constructor={"do_resize": False, "do_pad": False}, rust=rust_config("zoe_depth", {"kind": "none"}, mean=STANDARD_MEAN, std=STANDARD_STD)),
        CaseSpec("EfficientLoFTRImageProcessor", "efficient_lo_ftr", "transformers.keypoint_matching_image_processor", "keypoint_matching_pose", input_kind="pair", constructor={"size": {"height": 4, "width": 6}, "do_grayscale": False}, rust=rust_config("efficient_lo_ftr", fixed(4, 6), mean=None, std=None)),
        CaseSpec("LightGlueImageProcessor", "light_glue", "transformers.keypoint_matching_image_processor", "keypoint_matching_pose", input_kind="pair", constructor={"size": {"height": 4, "width": 6}, "do_grayscale": False}, rust=rust_config("light_glue", fixed(4, 6), mean=None, std=None)),
        CaseSpec("SuperGlueImageProcessor", "super_glue", "transformers.keypoint_matching_image_processor", "keypoint_matching_pose", input_kind="pair", constructor={"size": {"height": 4, "width": 6}, "do_grayscale": False}, rust=rust_config("super_glue", fixed(4, 6), mean=None, std=None)),
        CaseSpec("SuperPointImageProcessor", "super_point", "transformers.superpoint_image_processor", "keypoint_matching_pose", constructor={"size": {"height": 4, "width": 6}}, rust=rust_config("super_point", fixed(4, 6), mean=None, std=None)),
        CaseSpec("VitPoseImageProcessor", "vit_pose", "transformers.vitpose_image_processor", "keypoint_matching_pose", constructor={"do_affine_transform": False}, call={"boxes": [[[0.0, 0.0, 6.0, 4.0]]]}, rust=rust_config("vit_pose", {"kind": "none"})),
    ]


def deterministic_rgb(width: int, height: int, factors: tuple[int, int, int, int]) -> Image.Image:
    row, column, channel, offset = factors
    values = np.fromfunction(
        lambda y, x, c: (y * row + x * column + c * channel + offset) % 256,
        (height, width, 3),
        dtype=int,
    ).astype(np.uint8)
    return Image.fromarray(values, mode="RGB")


def deterministic_trimap(width: int, height: int) -> Image.Image:
    values = np.fromfunction(
        lambda y, x: (y * 85 + x * 51) % 256,
        (height, width),
        dtype=int,
    ).astype(np.uint8)
    return Image.fromarray(values, mode="L")


def tensor_fixture(name: str, tensor: Any) -> dict[str, Any]:
    tensor = tensor.detach().cpu()
    if name == "pixel_values" and tensor.ndim == 5:
        layout = "npchw"
    else:
        layout = "nchw"
    dtype = "float32" if tensor.dtype.is_floating_point else "int64"
    return {
        "shape": list(tensor.shape),
        "layout": layout,
        "dtype": dtype,
        "data": tensor.flatten().tolist(),
    }


def json_compatible(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_compatible(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def generate_processor_payload(processor: Any, spec: CaseSpec) -> dict[str, Any]:
    first = deterministic_rgb(spec.width, spec.height, (53, 29, 71, 11))
    second = deterministic_rgb(spec.width, spec.height, (31, 47, 17, 23))
    kwargs: dict[str, Any] = dict(spec.call or {})
    if spec.input_kind == "pair":
        kwargs["images"] = [[first, second]]
    else:
        kwargs["images"] = first
    if spec.input_kind == "matte":
        kwargs["trimaps"] = deterministic_trimap(spec.width, spec.height)
    kwargs["return_tensors"] = "pt"
    output = processor(**kwargs)

    tensors: dict[str, Any] = {}
    metadata: dict[str, list[list[int]]] = {}
    auxiliary: dict[str, Any] = {}
    for name, value in output.items():
        if name in {"original_sizes", "reshaped_input_sizes", "target_sizes"}:
            values = value.detach().cpu().to(dtype=getattr(__import__("torch"), "int64")).tolist()
            metadata[name] = values
        elif hasattr(value, "detach"):
            tensors[name] = tensor_fixture(name, value)
        else:
            auxiliary[name] = json_compatible(value)
    return {"outputs": tensors, "metadata": metadata, "auxiliary": auxiliary}


def generate_case(transformers: Any, spec: CaseSpec) -> dict[str, Any]:
    processor = getattr(transformers, spec.class_name)(**(spec.constructor or {}))
    payload = generate_processor_payload(processor, spec)
    return {
        "class_name": spec.class_name,
        "family": spec.family,
        "recipe_id": spec.recipe_id,
        "config": spec.rust,
        "input": {"kind": spec.input_kind, "width": spec.width, "height": spec.height},
        "outputs": payload["outputs"],
        "metadata": payload["metadata"],
    }


def alias_case_spec(spec: CaseSpec) -> tuple[CaseSpec, str]:
    constructor = copy.deepcopy(spec.constructor or {})
    call = copy.deepcopy(spec.call or {})
    rust = copy.deepcopy(spec.rust)
    geometry = "resize"

    if spec.class_name == "ZoeDepthImageProcessor":
        constructor = {"size": {"height": 4, "width": 6}, "do_pad": False}
        rust["resize"] = fixed(4, 6)
        rust["pre_resize_padding"] = {"kind": "none"}
    elif spec.class_name == "VitPoseImageProcessor":
        constructor = {
            "size": {"height": 4, "width": 6},
            "do_affine_transform": True,
        }
        call = {
            "boxes": [
                [
                    [0.0, 0.0, 7.0, 5.0],
                    [1.25, 0.75, 3.5, 2.25],
                ]
            ]
        }
        rust["resize"] = fixed(4, 6)
        geometry = "bounding_box_affine_resize"
    elif spec.class_name == "VitMatteImageProcessor":
        geometry = "padding_only_processor_has_no_resize_operation"

    return (
        replace(
            spec,
            width=7,
            height=5,
            constructor=constructor,
            call=call,
            rust=rust,
        ),
        geometry,
    )


def max_float_output_difference(
    canonical: dict[str, Any], pil: dict[str, Any]
) -> float:
    maximum = 0.0
    for name, canonical_tensor in canonical["outputs"].items():
        pil_tensor = pil["outputs"][name]
        if canonical_tensor["dtype"] != "float32":
            continue
        if canonical_tensor["shape"] != pil_tensor["shape"]:
            return float("inf")
        differences = (
            abs(left - right)
            for left, right in zip(canonical_tensor["data"], pil_tensor["data"])
        )
        maximum = max(maximum, max(differences, default=0.0))
    return maximum


def generate_alias_case(transformers: Any, spec: CaseSpec) -> dict[str, Any]:
    alias_spec, geometry = alias_case_spec(spec)
    alias_name = PIL_ALIAS_CLASSES[spec.class_name]
    canonical = generate_processor_payload(
        getattr(transformers, spec.class_name)(**(alias_spec.constructor or {})),
        alias_spec,
    )
    pil = generate_processor_payload(
        getattr(transformers, alias_name)(**(alias_spec.constructor or {})),
        alias_spec,
    )
    return {
        "class_name": spec.class_name,
        "alias_class_name": alias_name,
        "family": spec.family,
        "config": alias_spec.rust,
        "input": {
            "kind": alias_spec.input_kind,
            "width": alias_spec.width,
            "height": alias_spec.height,
        },
        "pose_boxes": alias_spec.call.get("boxes", [[]])[0]
        if spec.class_name == "VitPoseImageProcessor"
        else [],
        "geometry": geometry,
        "canonical_resize_parity": "Torchvision",
        "pil_resize_parity": "Compatibility",
        "max_float_output_difference": max_float_output_difference(canonical, pil),
        "backend_byte_identical": canonical == pil,
        "canonical": canonical,
        "pil": pil,
    }


def generate_postprocess(transformers: Any) -> dict[str, Any]:
    import torch

    detection_outputs = SimpleNamespace(
        logits=torch.tensor([[[3.0, 1.0, -2.0], [0.2, 2.5, -1.0]]]),
        pred_boxes=torch.tensor([[[0.5, 0.5, 0.4, 0.2], [0.25, 0.75, 0.2, 0.4]]]),
    )
    target_sizes = torch.tensor([[10, 20]])

    def detection_payload(processor: Any, **kwargs: Any) -> dict[str, Any]:
        result = processor.post_process_object_detection(
            detection_outputs, target_sizes=target_sizes, **kwargs
        )[0]
        return {
            "scores": result["scores"].tolist(),
            "labels": result["labels"].tolist(),
            "output_boxes": result["boxes"].tolist(),
        }

    sigmoid_top_k_100 = detection_payload(
        transformers.ConditionalDetrImageProcessor(), threshold=0.5, top_k=100
    )
    sigmoid_top_queries = detection_payload(
        transformers.RTDetrImageProcessor(), threshold=0.5
    )
    sigmoid_best = detection_payload(
        transformers.GroundingDinoImageProcessor(), threshold=0.1
    )
    softmax_background = detection_payload(
        transformers.YolosImageProcessor(), threshold=0.5
    )

    depth_processor = transformers.DPTImageProcessor()
    depth_outputs = SimpleNamespace(
        predicted_depth=torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]])
    )
    depth = depth_processor.post_process_depth_estimation(
        depth_outputs, target_sizes=[(2, 3)]
    )[0]["predicted_depth"]

    segmentation_outputs = SimpleNamespace(
        class_queries_logits=torch.tensor(
            [[[2.0, 0.0, -1.0, -2.0], [0.0, 2.0, 1.0, -2.0]]]
        ),
        masks_queries_logits=torch.tensor(
            [[[[2.0, 1.0, -1.0], [0.0, -1.0, -2.0]], [[-2.0, 0.0, 2.0], [-1.0, 1.0, 2.0]]]]
        ),
    )
    segmentation_output = transformers.Mask2FormerImageProcessor().post_process_semantic_segmentation(
        segmentation_outputs,
        target_sizes=[(2, 3)],
    )[0]

    sam_masks = torch.tensor([[[[[-1.0, 1.0, -1.0], [1.0, 1.0, -1.0]]]]])
    sam_output = transformers.Sam2ImageProcessor().post_process_masks(
        sam_masks, [[3, 4]], mask_threshold=0.0, binarize=True
    )[0]

    lightglue = transformers.LightGlueImageProcessor()
    matching_outputs = SimpleNamespace(
        mask=torch.tensor([[[1, 1, 0], [1, 1, 0]]]),
        keypoints=torch.tensor(
            [[[[0.25, 0.50], [0.75, 0.25], [0.0, 0.0]], [[0.50, 0.50], [0.25, 0.75], [0.0, 0.0]]]]
        ),
        matches=torch.tensor([[[1, 0, -1]]]),
        matching_scores=torch.tensor([[[0.8, 0.6, 0.0]]]),
    )
    matching = lightglue.post_process_keypoint_matching(
        matching_outputs,
        target_sizes=[[(10, 20), (8, 12)]],
        threshold=0.1,
    )[0]

    return {
        "detection": {
            "target_size": [10, 20],
            "logits": detection_outputs.logits.flatten().tolist(),
            "logits_shape": list(detection_outputs.logits.shape),
            "boxes": detection_outputs.pred_boxes.flatten().tolist(),
            "boxes_shape": list(detection_outputs.pred_boxes.shape),
            "softmax_background": softmax_background,
            "sigmoid_best_per_query": sigmoid_best,
            "sigmoid_top_k_100": sigmoid_top_k_100,
            "sigmoid_top_queries": sigmoid_top_queries,
        },
        "depth": {
            "target_size": [2, 3],
            "input_shape": list(depth_outputs.predicted_depth.shape),
            "input": depth_outputs.predicted_depth.flatten().tolist(),
            "output": depth.flatten().tolist(),
        },
        "segmentation": {
            "target_size": [2, 3],
            "class_logits_shape": list(segmentation_outputs.class_queries_logits.shape),
            "class_logits": segmentation_outputs.class_queries_logits.flatten().tolist(),
            "mask_logits_shape": list(segmentation_outputs.masks_queries_logits.shape),
            "mask_logits": segmentation_outputs.masks_queries_logits.flatten().tolist(),
            "class_ids": segmentation_output.flatten().tolist(),
        },
        "sam2_masks": {
            "input_shape": list(sam_masks.shape),
            "input": sam_masks.flatten().tolist(),
            "target_size": [3, 4],
            "output_shape": list(sam_output.shape),
            "output": sam_output.flatten().to(dtype=torch.int64).tolist(),
        },
        "matching": {
            "target_sizes": [[10, 20], [8, 12]],
            "normalized_keypoints": matching_outputs.keypoints[0, :, :2, :].tolist(),
            "match_indices": [1, 0],
            "input_scores": matching_outputs.matching_scores[0, 0, :2].tolist(),
            "keypoints0": matching["keypoints0"].tolist(),
            "keypoints1": matching["keypoints1"].tolist(),
            "scores": matching["matching_scores"].tolist(),
        },
    }


def verify_source(source: Path) -> None:
    source = verify_clean_git_checkout(
        source, expected_commit=AUDIT_COMMIT, project="Transformers"
    )
    source_module = source / "src"
    sys.path.insert(0, str(source_module))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify_source(args.source.resolve())

    import transformers

    loaded_from = Path(transformers.__file__).resolve()
    if args.source.resolve() not in loaded_from.parents:
        raise RuntimeError(f"Transformers loaded from unexpected path: {loaded_from}")

    fixture = {
        "schema": SCHEMA,
        "upstream": {
            "library": "transformers",
            "version": transformers.__version__,
            "source": SOURCE_URL,
            "commit": AUDIT_COMMIT,
        },
        "generator": "scripts/parity/transformers_task_vision_parity.py",
        "acceptance": acceptance_contract(
            input_locations=[
                "cases[].input",
                "alias_cases[].input",
                "postprocess",
            ],
            input_description=(
                "RGB values use (row * 53 + column * 29 + channel * 71 + 11) % 256; "
                "paired RGB values use factors (31, 47, 17, 23); trimaps use "
                "(row * 85 + column * 51) % 256; postprocess inputs are embedded in full."
            ),
            config_locations=["cases[].config", "alias_cases[].config"],
            absolute_tolerance=1.0e-5,
            statistics_absolute_tolerance=1.0e-12,
        ),
        "cases": [generate_case(transformers, spec) for spec in cases()],
        "alias_cases": [
            generate_alias_case(transformers, spec)
            for spec in cases()
            if spec.class_name in PIL_ALIAS_CLASSES
        ],
        "postprocess": generate_postprocess(transformers),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
