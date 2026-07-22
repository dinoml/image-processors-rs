#!/usr/bin/env python
"""Generate deterministic multimodal catalog fixtures from Transformers.

The script must be run against the exact audited Transformers checkout.  It
uses deliberately small processor settings so every numeric output can be
stored in full while still exercising each class's characteristic output
contract (patch grids, masks, tiles, document fields, or temporal axes).
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image
import torch
import transformers

from _fixture_contract import acceptance_contract, verify_clean_git_checkout


AUDIT_COMMIT = "6d960ca0a0eba0d2aebc920d8080a9353da468d3"
SCHEMA = "image-processors.transformers-multimodal-parity.v2"
IMAGE_SIZE = 8
PATCH_SIZE = 2


@dataclass(frozen=True)
class CaseSpec:
    category: str
    class_name: str
    model_type: str
    profile: str
    kwargs: dict[str, Any]
    input_kind: str = "image"


def fixed(category: str, class_name: str, model_type: str, **kwargs: Any) -> CaseSpec:
    return CaseSpec(category, class_name, model_type, "fixed", kwargs)


def mask(category: str, class_name: str, model_type: str, **kwargs: Any) -> CaseSpec:
    return CaseSpec(category, class_name, model_type, "fixed_mask", kwargs)


def tiled(category: str, class_name: str, model_type: str, **kwargs: Any) -> CaseSpec:
    return CaseSpec(category, class_name, model_type, "tiled", kwargs)


def patch_grid(category: str, class_name: str, model_type: str, **kwargs: Any) -> CaseSpec:
    return CaseSpec(category, class_name, model_type, "patch_grid", kwargs)


def adaptive(category: str, class_name: str, model_type: str, **kwargs: Any) -> CaseSpec:
    return CaseSpec(category, class_name, model_type, "adaptive_patches", kwargs)


CASES: tuple[CaseSpec, ...] = (
    tiled("vision_language", "AriaImageProcessor", "aria", min_image_size=1, max_image_size=490, split_image=False),
    fixed("vision_language", "BlipImageProcessor", "blip", size={"height": 8, "width": 8}),
    mask(
        "vision_language",
        "BridgeTowerImageProcessor",
        "bridgetower",
        size={"shortest_edge": 8},
        crop_size={"shortest_edge": 8},
        size_divisor=1,
    ),
    fixed(
        "vision_language",
        "ChameleonImageProcessor",
        "chameleon",
        size={"shortest_edge": 8},
        crop_size={"height": 8, "width": 8},
    ),
    tiled(
        "vision_language",
        "Cohere2VisionImageProcessor",
        "cohere2_vision",
        size={"height": 8, "width": 8},
        crop_to_patches=False,
        min_patches=1,
        max_patches=1,
    ),
    tiled(
        "document_understanding",
        "DeepseekOcr2ImageProcessor",
        "deepseek_ocr2",
        size={"height": 8, "width": 8},
        tile_size=8,
        crop_to_patches=False,
        min_patches=1,
        max_patches=1,
    ),
    fixed(
        "vision_language",
        "DeepseekVLHybridImageProcessor",
        "deepseek_vl_hybrid",
        size={"height": 8, "width": 8},
        high_res_size={"height": 8, "width": 8},
        min_size=1,
    ),
    fixed(
        "vision_language",
        "DeepseekVLImageProcessor",
        "deepseek_vl",
        size={"height": 8, "width": 8},
        min_size=1,
    ),
    mask(
        "vision_language",
        "Emu3ImageProcessor",
        "emu3",
        min_pixels=64,
        max_pixels=64,
        spatial_factor=1,
    ),
    patch_grid(
        "vision_language",
        "Ernie4_5_VLMoeImageProcessor",
        "ernie4_5_vl_moe",
        patch_size=2,
        merge_size=1,
        size={"shortest_edge": 64, "longest_edge": 64},
    ),
    CaseSpec(
        "vision_language",
        "FuyuImageProcessor",
        "fuyu",
        "fuyu",
        {"size": {"height": 8, "width": 8}, "patch_size": {"height": 2, "width": 2}},
    ),
    CaseSpec(
        "vision_language",
        "Gemma4ImageProcessor",
        "gemma4",
        "gemma4",
        {"patch_size": 2, "pooling_kernel_size": 1, "max_soft_tokens": 70},
    ),
    CaseSpec(
        "vision_language",
        "Gemma4UnifiedImageProcessor",
        "gemma4_unified",
        "gemma4_unified",
        {"patch_size": 2, "pooling_kernel_size": 1, "max_soft_tokens": 70},
    ),
    patch_grid(
        "vision_language",
        "Glm46VImageProcessor",
        "glm46v",
        patch_size=2,
        temporal_patch_size=1,
        merge_size=1,
        size={"shortest_edge": 64, "longest_edge": 64},
    ),
    patch_grid(
        "vision_language",
        "Glm4vImageProcessor",
        "glm4v",
        patch_size=2,
        temporal_patch_size=1,
        merge_size=1,
        size={"shortest_edge": 64, "longest_edge": 64},
    ),
    patch_grid(
        "vision_language",
        "GlmImageImageProcessor",
        "glm_image",
        patch_size=2,
        temporal_patch_size=1,
        merge_size=1,
        size={"shortest_edge": 64, "longest_edge": 64},
    ),
    patch_grid(
        "vision_language",
        "GlmgaImageProcessor",
        "glmga",
        patch_size=2,
        temporal_patch_size=1,
        merge_size=1,
        patch_expand_factor=1,
        size={"shortest_edge": 64, "longest_edge": 64},
    ),
    tiled(
        "document_understanding",
        "GotOcr2ImageProcessor",
        "got_ocr2",
        size={"height": 8, "width": 8},
        crop_to_patches=False,
        min_patches=1,
        max_patches=1,
    ),
    patch_grid(
        "vision_language",
        "HunYuanVLImageProcessor",
        "hunyuan_vl",
        patch_size=2,
        temporal_patch_size=1,
        merge_size=1,
        size={"shortest_edge": 64, "longest_edge": 64},
    ),
    mask(
        "vision_language",
        "Idefics2ImageProcessor",
        "idefics2",
        size={"shortest_edge": 8, "longest_edge": 8},
        do_image_splitting=False,
    ),
    fixed(
        "vision_language",
        "IdeficsImageProcessor",
        "idefics",
        image_size=8,
        size={"height": 8, "width": 8},
    ),
    fixed(
        "vision_language",
        "JanusImageProcessor",
        "janus",
        size={"height": 8, "width": 8},
        min_size=1,
    ),
    CaseSpec(
        "vision_language",
        "Kimi_K25ImageProcessor",
        "kimi_k25",
        "patch_frames",
        {"patch_size": 2, "merge_size": 1, "max_patches": 16, "size": {"max_height": 8, "max_width": 8}},
    ),
    adaptive(
        "document_understanding",
        "Kosmos2_5ImageProcessor",
        "kosmos2_5",
        patch_size={"height": 2, "width": 2},
        max_patches=16,
    ),
    fixed(
        "document_understanding",
        "LayoutLMv2ImageProcessor",
        "layoutlmv2",
        apply_ocr=False,
        size={"height": 8, "width": 8},
    ),
    fixed(
        "document_understanding",
        "LayoutLMv3ImageProcessor",
        "layoutlmv3",
        apply_ocr=False,
        size={"height": 8, "width": 8},
    ),
    adaptive(
        "vision_language",
        "Lfm2VlImageProcessor",
        "lfm2_vl",
        size={"height": 8, "width": 8},
        tile_size=8,
        encoder_patch_size=2,
        downsample_factor=1,
        min_image_tokens=16,
        max_image_tokens=16,
        max_num_patches=16,
        min_tiles=1,
        max_tiles=1,
        do_image_splitting=False,
        return_row_col_info=True,
    ),
    tiled(
        "vision_language",
        "Llama4ImageProcessor",
        "llama4",
        size={"height": 8, "width": 8},
        max_patches=1,
        resize_to_max_canvas=True,
    ),
    fixed(
        "vision_language",
        "LlavaImageProcessor",
        "llava",
        size={"shortest_edge": 8},
        crop_size={"height": 8, "width": 8},
    ),
    tiled(
        "vision_language",
        "LlavaOnevisionImageProcessor",
        "llava_onevision",
        size={"height": 8, "width": 8},
        image_grid_pinpoints=[[8, 8]],
    ),
    CaseSpec(
        "vision_language",
        "MiniCPMV4_6ImageProcessor",
        "minicpmv4_6",
        "minicpm",
        {"patch_size": 2, "scale_resolution": 8, "slice_mode": False, "max_slice_nums": 1},
    ),
    patch_grid(
        "vision_language",
        "MiniMaxM3VLImageProcessor",
        "minimax_m3_vl",
        patch_size=2,
        temporal_patch_size=1,
        merge_size=1,
        max_pixels=3136,
        size={"height": 8, "width": 8},
    ),
    fixed(
        "document_understanding",
        "NougatImageProcessor",
        "nougat",
        size={"height": 8, "width": 8},
        do_crop_margin=False,
        do_thumbnail=False,
        do_align_long_axis=False,
    ),
    tiled(
        "vision_language",
        "Ovis2ImageProcessor",
        "ovis2",
        size={"height": 8, "width": 8},
        crop_to_patches=False,
        min_patches=1,
        max_patches=1,
    ),
    mask(
        "vision_language",
        "PI0ImageProcessor",
        "pi0",
        size={"max_height": 8, "max_width": 8},
        pad_size={"height": 8, "width": 8},
    ),
    fixed(
        "document_understanding",
        "PPChart2TableImageProcessor",
        "pp_chart2table",
        size={"height": 8, "width": 8},
    ),
    fixed(
        "document_understanding",
        "PPFormulaNetImageProcessor",
        "pp_formulanet",
        size={"height": 8, "width": 8},
        do_crop_margin=False,
        do_thumbnail=False,
        do_align_long_axis=False,
    ),
    fixed(
        "document_understanding",
        "PPOCRV5ServerRecImageProcessor",
        "pp_ocrv5_server_rec",
        size={"height": 8, "width": 8},
        pad_size={"height": 8, "width": 8},
        max_image_width=8,
        character_list=["", "a", "b", "c"],
    ),
    fixed(
        "document_understanding",
        "PPOCRV6SmallRecImageProcessor",
        "pp_ocrv6_small_rec",
        size={"height": 8, "width": 8},
        pad_size={"height": 8, "width": 8},
        max_image_width=8,
        character_list=["", "a", "b", "c"],
    ),
    CaseSpec(
        "document_understanding",
        "PaddleOCRVLImageProcessor",
        "paddleocr_vl",
        "patch_frames",
        {"patch_size": 2, "temporal_patch_size": 1, "merge_size": 1, "size": {"shortest_edge": 64, "longest_edge": 64}},
    ),
    tiled(
        "vision_language",
        "PerceptionLMImageProcessor",
        "perception_lm",
        size={"height": 8, "width": 8},
        tile_size=8,
        max_num_tiles=1,
        vision_input_type="tile",
    ),
    CaseSpec(
        "vision_language",
        "Phi4MultimodalImageProcessor",
        "phi4_multimodal",
        "phi4",
        {"size": {"height": 12, "width": 12}, "patch_size": 2, "dynamic_hd": 1},
    ),
    adaptive(
        "document_understanding",
        "Pix2StructImageProcessor",
        "pix2struct",
        patch_size={"height": 2, "width": 2},
        max_patches=16,
    ),
    mask(
        "document_understanding",
        "SLANeXtImageProcessor",
        "slanext",
        size={"height": 8, "width": 8},
        pad_size={"height": 8, "width": 8},
    ),
    adaptive(
        "vision_language",
        "Siglip2ImageProcessor",
        "siglip2",
        patch_size=2,
        max_num_patches=16,
    ),
    tiled(
        "vision_language",
        "SmolVLMImageProcessor",
        "smolvlm",
        size={"longest_edge": 8},
        max_image_size={"longest_edge": 8},
        do_image_splitting=False,
    ),
    fixed(
        "document_understanding",
        "TextNetImageProcessor",
        "textnet",
        size={"shortest_edge": 8},
        crop_size={"height": 8, "width": 8},
        size_divisor=1,
    ),
    CaseSpec(
        "video",
        "TvpImageProcessor",
        "tvp",
        "video",
        {"size": {"longest_edge": 8}, "crop_size": {"height": 8, "width": 8}, "pad_size": {"height": 8, "width": 8}},
        "video",
    ),
    CaseSpec(
        "document_understanding",
        "UVDocImageProcessor",
        "uvdoc",
        "uvdoc",
        {"size": {"height": 8, "width": 8}},
    ),
    CaseSpec(
        "video",
        "VideoLlama3ImageProcessor",
        "video_llama_3",
        "patch_grid",
        {"patch_size": 2, "temporal_patch_size": 1, "merge_size": 1, "size": {"shortest_edge": 64, "longest_edge": 64}},
        "image_sequence",
    ),
    CaseSpec(
        "video",
        "VideoLlavaImageProcessor",
        "video_llava",
        "video",
        {"size": {"shortest_edge": 8}, "crop_size": {"height": 8, "width": 8}},
        "image_sequence",
    ),
    mask(
        "vision_language",
        "ViltImageProcessor",
        "vilt",
        size={"shortest_edge": 8},
        size_divisor=1,
    ),
)

ALIASES: tuple[tuple[str, str], ...] = (
    ("AriaImageProcessorPil", "AriaImageProcessor"),
    ("BlipImageProcessorPil", "BlipImageProcessor"),
    ("BridgeTowerImageProcessorPil", "BridgeTowerImageProcessor"),
    ("ChameleonImageProcessorPil", "ChameleonImageProcessor"),
    ("DeepseekOcr2ImageProcessorPil", "DeepseekOcr2ImageProcessor"),
    ("DeepseekVLHybridImageProcessorPil", "DeepseekVLHybridImageProcessor"),
    ("DeepseekVLImageProcessorPil", "DeepseekVLImageProcessor"),
    ("Ernie4_5_VLMoeImageProcessorPil", "Ernie4_5_VLMoeImageProcessor"),
    ("Ernie4_5_VL_MoeImageProcessor", "Ernie4_5_VLMoeImageProcessor"),
    ("Ernie4_5_VL_MoeImageProcessorPil", "Ernie4_5_VLMoeImageProcessor"),
    ("FuyuImageProcessorPil", "FuyuImageProcessor"),
    ("Gemma4ImageProcessorPil", "Gemma4ImageProcessor"),
    ("Glm46VImageProcessorPil", "Glm46VImageProcessor"),
    ("Glm4vImageProcessorPil", "Glm4vImageProcessor"),
    ("GlmImageImageProcessorPil", "GlmImageImageProcessor"),
    ("GlmgaImageProcessorPil", "GlmgaImageProcessor"),
    ("GotOcr2ImageProcessorPil", "GotOcr2ImageProcessor"),
    ("HunYuanVLImageProcessorPil", "HunYuanVLImageProcessor"),
    ("Idefics2ImageProcessorPil", "Idefics2ImageProcessor"),
    ("IdeficsImageProcessorPil", "IdeficsImageProcessor"),
    ("JanusImageProcessorPil", "JanusImageProcessor"),
    ("Kosmos2_5ImageProcessorPil", "Kosmos2_5ImageProcessor"),
    ("LayoutLMv2ImageProcessorPil", "LayoutLMv2ImageProcessor"),
    ("LayoutLMv3ImageProcessorPil", "LayoutLMv3ImageProcessor"),
    ("LlavaImageProcessorPil", "LlavaImageProcessor"),
    ("LlavaOnevisionImageProcessorPil", "LlavaOnevisionImageProcessor"),
    ("MiniCPMV4_6ImageProcessorPil", "MiniCPMV4_6ImageProcessor"),
    ("NougatImageProcessorPil", "NougatImageProcessor"),
    ("Ovis2ImageProcessorPil", "Ovis2ImageProcessor"),
    ("PPChart2TableImageProcessorPil", "PPChart2TableImageProcessor"),
    ("PaddleOCRVLImageProcessorPil", "PaddleOCRVLImageProcessor"),
    ("Pix2StructImageProcessorPil", "Pix2StructImageProcessor"),
    ("Siglip2ImageProcessorPil", "Siglip2ImageProcessor"),
    ("SmolVLMImageProcessorPil", "SmolVLMImageProcessor"),
    ("TextNetImageProcessorPil", "TextNetImageProcessor"),
    ("TvpImageProcessorPil", "TvpImageProcessor"),
    ("VideoLlama3ImageProcessorPil", "VideoLlama3ImageProcessor"),
    ("ViltImageProcessorPil", "ViltImageProcessor"),
)


def deterministic_image(offset: int = 0, edge: int = IMAGE_SIZE) -> Image.Image:
    values = np.arange(edge * edge * 3, dtype=np.uint32).reshape(edge, edge, 3)
    values = ((values * 37 + 17 + offset * 53) % 256).astype(np.uint8)
    return Image.fromarray(values, mode="RGB")


def numpy_array(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float()
        return tensor.numpy()
    return np.asarray(value)


def array_payload(value: Any) -> dict[str, Any]:
    array = np.ascontiguousarray(numpy_array(value))
    return {
        "kind": "array",
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "data": base64.b64encode(array.tobytes()).decode("ascii"),
    }


def value_payload(value: Any) -> Any:
    if isinstance(value, torch.Tensor) or isinstance(value, np.ndarray):
        return array_payload(value)
    if isinstance(value, (list, tuple)):
        return {"kind": "sequence", "items": [value_payload(item) for item in value]}
    if isinstance(value, dict):
        return {"kind": "mapping", "items": {str(key): value_payload(item) for key, item in value.items()}}
    if isinstance(value, np.generic):
        return value.item()
    return value


def encoded_mapping(encoded: Any) -> dict[str, Any]:
    if isinstance(encoded, torch.Tensor):
        return {"pixel_values": encoded}
    if isinstance(encoded, dict):
        return dict(encoded)
    if hasattr(encoded, "items"):
        return dict(encoded.items())
    raise TypeError(f"unsupported processor output type: {type(encoded).__name__}")


def audited_input_edge(class_name: str) -> int:
    return {
        "AriaImageProcessor": 490,
        "Gemma4ImageProcessor": 16,
        "Gemma4UnifiedImageProcessor": 16,
        "GlmImageImageProcessor": 4,
        "MiniMaxM3VLImageProcessor": 56,
        "Phi4MultimodalImageProcessor": 12,
    }.get(class_name, IMAGE_SIZE)


def resize_probe_edge(class_name: str) -> int:
    return audited_input_edge(class_name) + 3


def encode_inputs(processor: Any, spec: CaseSpec, first: Image.Image, second: Image.Image) -> Any:
    if spec.class_name == "IdeficsImageProcessor":
        return processor([first], return_tensors="pt")
    if spec.class_name == "TvpImageProcessor":
        return processor([[first, second]], return_tensors="pt")
    if spec.class_name in {"VideoLlavaImageProcessor", "VideoLlama3ImageProcessor"}:
        return processor([first, second], return_tensors="pt")
    return processor(first, return_tensors="pt")


def input_payload(spec: CaseSpec, edge: int) -> dict[str, Any]:
    return {
        "kind": spec.input_kind,
        "mode": "RGB",
        "width": edge,
        "height": edge,
        "frames": 2 if spec.input_kind != "image" else 1,
    }


def output_payload(encoded: Any) -> dict[str, Any]:
    return {key: value_payload(value) for key, value in encoded_mapping(encoded).items()}


def source_payload(processor_class: type) -> dict[str, str]:
    module_path = Path(*processor_class.__module__.split(".")).with_suffix(".py")
    source_path = Path(transformers.__file__).resolve().parent.parent / module_path
    return {
        "path": module_path.as_posix(),
        "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
    }


def process_case(spec: CaseSpec) -> dict[str, Any]:
    processor_class = getattr(transformers, spec.class_name)
    processor = processor_class(**spec.kwargs)
    input_edge = audited_input_edge(spec.class_name)
    first = deterministic_image(0, input_edge)
    second = deterministic_image(1, input_edge)
    encoded = encode_inputs(processor, spec, first, second)

    probe_edge = resize_probe_edge(spec.class_name)
    probe_first = deterministic_image(2, probe_edge)
    probe_second = deterministic_image(3, probe_edge)
    probe_encoded = encode_inputs(processor, spec, probe_first, probe_second)

    processor_config = processor.to_dict()
    processor_config.pop("image_processor_type", None)
    return {
        "category": spec.category,
        "class_name": spec.class_name,
        "model_type": spec.model_type,
        "profile": spec.profile,
        "recipe_id": f"transformers.{spec.model_type}_image_processor",
        "source": source_payload(processor_class),
        "input": input_payload(spec, input_edge),
        "processor_config": processor_config,
        "outputs": output_payload(encoded),
        "resize_probe": {
            "input": input_payload(spec, probe_edge),
            "outputs": output_payload(probe_encoded),
        },
    }


def process_alias(alias: str, canonical: str) -> dict[str, Any]:
    spec = next(spec for spec in CASES if spec.class_name == canonical)
    processor_class = getattr(transformers, alias)
    processor = processor_class(**spec.kwargs)
    probe_edge = resize_probe_edge(canonical)
    first = deterministic_image(2, probe_edge)
    second = deterministic_image(3, probe_edge)
    encoded = encode_inputs(processor, spec, first, second)
    processor_config = processor.to_dict()
    processor_config.pop("image_processor_type", None)
    return {
        "alias": alias,
        "canonical": canonical,
        "source": source_payload(processor_class),
        "input": input_payload(spec, probe_edge),
        "processor_config": processor_config,
        "outputs": output_payload(encoded),
    }


def postprocess_cases() -> dict[str, Any]:
    logits = torch.tensor(
        [[[0.0, 3.0, 1.0, 0.0], [0.0, 4.0, 1.0, 0.0], [5.0, 0.0, 0.0, 0.0], [0.0, 0.0, 6.0, 0.0]]]
    )
    token_ids = logits.argmax(dim=2)
    token_scores = logits.max(dim=2).values
    ctc_sequences = []
    for batch_ids, batch_scores in zip(token_ids.tolist(), token_scores.tolist(), strict=True):
        retained_ids = []
        retained_scores = []
        previous = None
        for token_id, score in zip(batch_ids, batch_scores, strict=True):
            if token_id != 0 and token_id != previous:
                retained_ids.append(token_id)
                retained_scores.append(score)
            previous = token_id
        ctc_sequences.append({"token_ids": retained_ids, "scores": retained_scores})
    recognition = {}
    for class_name in ("PPOCRV5ServerRecImageProcessor", "PPOCRV6SmallRecImageProcessor"):
        processor = getattr(transformers, class_name)(character_list=["", "a", "b", "c"])
        recognition[class_name] = {
            "logits": array_payload(logits),
            "decoder": {"strategy": "ctc_greedy", "blank_token_id": 0},
            "token_sequences": ctc_sequences,
            "result": processor.post_process_text_recognition(SimpleNamespace(last_hidden_state=logits)),
        }

    table_processor = transformers.SLANeXtImageProcessor()
    table_logits = torch.full((1, 4, len(table_processor.character)), -4.0)
    table_logits[0, 0, table_processor.dict["<thead>"]] = 2.0
    table_logits[0, 1, table_processor.dict["<tr>"]] = 3.0
    table_logits[0, 2, table_processor.dict["<td></td>"]] = 4.0
    table_logits[0, 3, table_processor.eos_id] = 5.0
    table_ids = table_logits.argmax(dim=2)[0].tolist()
    table_scores = table_logits.max(dim=2).values[0].tolist()
    retained_table_ids = []
    retained_table_scores = []
    for position, (token_id, score) in enumerate(zip(table_ids, table_scores, strict=True)):
        if position > 0 and token_id == table_processor.eos_id:
            break
        if token_id in (table_processor.bos_id, table_processor.eos_id):
            continue
        retained_table_ids.append(token_id)
        retained_table_scores.append(score)
    table = {
        "logits": array_payload(table_logits),
        "decoder": {
            "strategy": "greedy",
            "begin_token_id": table_processor.bos_id,
            "end_token_id": table_processor.eos_id,
        },
        "token_sequences": [{"token_ids": retained_table_ids, "scores": retained_table_scores}],
        "result": table_processor.post_process_table_recognition(SimpleNamespace(last_hidden_state=table_logits)),
    }

    uvdoc = transformers.UVDocImageProcessor(size={"height": 2, "width": 2})
    original = torch.tensor([[[0.0, 0.25], [0.5, 0.75]], [[0.1, 0.35], [0.6, 0.85]], [[0.2, 0.45], [0.7, 0.95]]])
    identity_grid = torch.tensor([[[[-1.0, -1.0], [1.0, -1.0]], [[-1.0, 1.0], [1.0, 1.0]]]])
    prediction = identity_grid.permute(0, 3, 1, 2)
    rectified = uvdoc.post_process_document_rectification(prediction, [original])
    rectification = {
        "prediction": array_payload(prediction),
        "original_images": value_payload([original]),
        "result": value_payload(rectified),
    }

    return {"recognition": recognition, "table": table, "rectification": rectification}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transformers-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("image-processors/tests/fixtures/transformers/catalog_multimodal.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = verify_clean_git_checkout(
        args.transformers_root, expected_commit=AUDIT_COMMIT, project="Transformers"
    )

    imported_root = Path(transformers.__file__).resolve().parents[2]
    if imported_root != root:
        raise SystemExit(f"PYTHONPATH loaded Transformers from {imported_root}, expected {root}")

    cases = [process_case(spec) for spec in CASES]
    if len(cases) != 52:
        raise AssertionError(f"expected 52 multimodal cases, got {len(cases)}")
    aliases = [process_alias(alias, canonical) for alias, canonical in ALIASES]

    payload = {
        "schema": SCHEMA,
        "generator": "scripts/parity/transformers_multimodal_parity.py",
        "upstream": {
            "library": "transformers",
            "source": "huggingface/transformers",
            "version": transformers.__version__,
            "commit": AUDIT_COMMIT,
        },
        "acceptance": acceptance_contract(
            input_locations=[
                "fixture",
                "cases[].input",
                "cases[].resize_probe.input",
                "aliases[].input",
                "postprocess",
            ],
            input_description=(
                "RGB values use (index * 37 + 17 + frame_offset * 53) % 256; "
                "postprocess tensors are embedded in full."
            ),
            config_locations=["cases[].processor_config", "aliases[].processor_config"],
            absolute_tolerance=1.0e-5,
        ),
        "fixture": {"width": IMAGE_SIZE, "height": IMAGE_SIZE, "patch_size": PATCH_SIZE},
        "cases": cases,
        "aliases": aliases,
        "postprocess": postprocess_cases(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {len(cases)} cases and {len(aliases)} aliases to {args.output}")


if __name__ == "__main__":
    main()
