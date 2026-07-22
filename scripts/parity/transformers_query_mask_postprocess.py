#!/usr/bin/env python3
"""Generate query-mask postprocess fixtures from the audited Transformers source."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from _fixture_contract import acceptance_contract, verify_clean_git_checkout


AUDIT_COMMIT = "6d960ca0a0eba0d2aebc920d8080a9353da468d3"
GENERATOR = "scripts/parity/transformers_query_mask_postprocess.py"
SCHEMA = "image-processors.transformers-query-mask-postprocess.v1"
SOURCE = f"https://github.com/huggingface/transformers/tree/{AUDIT_COMMIT}"
EOMT_SIZE = {"shortest_edge": 4, "longest_edge": 6}
TARGET_SIZE = [3, 7]


SCENARIOS: dict[str, dict[str, Any]] = {
    "primary": {
        "class_logits_shape": [1, 4, 6],
        "class_logits": [
            7.0,
            -4.0,
            -4.0,
            -4.0,
            -4.0,
            -5.0,
            -4.0,
            -4.0,
            -4.0,
            -4.0,
            6.5,
            -5.0,
            -1.0,
            -1.0,
            -1.0,
            0.1,
            -1.0,
            0.0,
            -5.0,
            -5.0,
            -5.0,
            -5.0,
            -5.0,
            7.0,
        ],
        "mask_logits_shape": [1, 4, 2, 3],
        "mask_logits": [
            6.0,
            4.0,
            -3.0,
            5.0,
            -2.0,
            -6.0,
            -6.0,
            -2.0,
            4.0,
            -4.0,
            5.0,
            6.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            6.0,
            6.0,
            6.0,
            6.0,
            6.0,
            6.0,
        ],
        "target_size": TARGET_SIZE,
    },
    "fusion": {
        "class_logits_shape": [1, 3, 6],
        "class_logits": [
            -4.0,
            -4.0,
            7.0,
            -4.0,
            -4.0,
            -5.0,
            -4.0,
            -4.0,
            6.5,
            -4.0,
            -4.0,
            -5.0,
            -5.0,
            -5.0,
            -5.0,
            -5.0,
            -5.0,
            7.0,
        ],
        "mask_logits_shape": [1, 3, 2, 3],
        "mask_logits": [
            6.0,
            5.0,
            -5.0,
            6.0,
            -3.0,
            -6.0,
            -6.0,
            -3.0,
            6.0,
            -5.0,
            5.0,
            6.0,
            6.0,
            6.0,
            6.0,
            6.0,
            6.0,
            6.0,
        ],
        "target_size": TARGET_SIZE,
        "label_ids_to_fuse": [2],
    },
    "no_mask": {
        "class_logits_shape": [1, 2, 6],
        "class_logits": [-6.0, -6.0, -6.0, -6.0, -6.0, 8.0] * 2,
        "mask_logits_shape": [1, 2, 2, 3],
        "mask_logits": [6.0, -6.0, 6.0, -6.0, 6.0, -6.0] * 2,
        "target_size": TARGET_SIZE,
    },
    "assigned_overlap": {
        "class_logits_shape": [1, 3, 6],
        "class_logits": [
            7.0,
            -4.0,
            -4.0,
            -4.0,
            -4.0,
            -5.0,
            -4.0,
            6.5,
            -4.0,
            -4.0,
            -4.0,
            -5.0,
            -5.0,
            -5.0,
            -5.0,
            -5.0,
            -5.0,
            7.0,
        ],
        "mask_logits_shape": [1, 3, 4, 6],
        "mask_logits": (
            [2.0] * 5
            + [-0.1] * 19
            + [-4.0] * 4
            + [4.0]
            + [-4.0] * 19
            + [6.0] * 24
        ),
        "target_size": [4, 6],
    },
    "oneformer_downsample": {
        "class_logits_shape": [1, 3, 6],
        "class_logits": [
            7.0,
            -4.0,
            -4.0,
            -4.0,
            -4.0,
            -5.0,
            -4.0,
            -4.0,
            -4.0,
            -4.0,
            6.5,
            -5.0,
            -5.0,
            -5.0,
            -5.0,
            -5.0,
            -5.0,
            7.0,
        ],
        "mask_logits_shape": [1, 3, 4, 6],
        "mask_logits": (
            [6.0, 6.0, 6.0, -6.0, -6.0, -6.0] * 4
            + [-6.0, -6.0, -6.0, 6.0, 6.0, 6.0] * 4
            + [6.0] * 24
        ),
        "target_size": [2, 3],
    },
}


def tensor(values: list[float], shape: list[int], torch: Any) -> Any:
    return torch.tensor(values, dtype=torch.float32).reshape(shape)


def outputs_for(scenario: dict[str, Any], torch: Any) -> SimpleNamespace:
    return SimpleNamespace(
        class_queries_logits=tensor(
            scenario["class_logits"], scenario["class_logits_shape"], torch
        ),
        masks_queries_logits=tensor(
            scenario["mask_logits"], scenario["mask_logits_shape"], torch
        ),
        patch_offsets=[],
    )


def semantic_expected(output: Any) -> dict[str, Any]:
    return {
        "size": list(output.segmentation.shape),
        "num_labels": int(output.segmentation_scores.shape[0]),
        "class_ids": [int(value) for value in output.segmentation.flatten().tolist()],
        "scores": [float(value) for value in output.segmentation_scores.flatten().tolist()],
    }


def segmentation_expected(
    output: dict[str, Any], label_ids_to_fuse: list[int]
) -> dict[str, Any]:
    fused = set(label_ids_to_fuse)
    segments = []
    for segment in output["segments_info"]:
        label_id = int(segment["label_id"])
        segments.append(
            {
                "id": int(segment["id"]),
                "label_id": label_id,
                "was_fused": bool(segment.get("was_fused", label_id in fused)),
                "score": float(segment["score"]),
            }
        )
    segmentation = output["segmentation"]
    return {
        "size": list(segmentation.shape),
        "segmentation": [int(value) for value in segmentation.flatten().tolist()],
        "segments": segments,
    }


def generate_cases(processors: dict[str, Any], torch: Any) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for family, processor in processors.items():
        scenario = SCENARIOS["primary"]
        output = outputs_for(scenario, torch)
        semantic = processor.post_process_semantic_segmentation(
            output,
            target_sizes=[tuple(scenario["target_size"])],
            return_segmentation_scores=True,
            **({"size": EOMT_SIZE} if family == "eomt" else {}),
        )[0]
        cases.append(
            {
                "family": family,
                "task": "semantic",
                "scenario": "primary",
                "expected": semantic_expected(semantic),
            }
        )

    oneformer_downsample = SCENARIOS["oneformer_downsample"]
    semantic = processors["one_former"].post_process_semantic_segmentation(
        outputs_for(oneformer_downsample, torch),
        target_sizes=[tuple(oneformer_downsample["target_size"])],
        return_segmentation_scores=True,
    )[0]
    cases.append(
        {
            "family": "one_former",
            "task": "semantic",
            "scenario": "oneformer_downsample",
            "expected": semantic_expected(semantic),
        }
    )

    for family in ["mask_former", "mask2_former", "eomt"]:
        processor = processors[family]
        for scenario_name in ["primary", "no_mask"]:
            scenario = SCENARIOS[scenario_name]
            kwargs: dict[str, Any] = {
                "target_sizes": [tuple(scenario["target_size"])]
            }
            if family == "eomt":
                kwargs.update({"threshold": 0.8, "size": EOMT_SIZE})
            result = processor.post_process_instance_segmentation(
                outputs_for(scenario, torch), **kwargs
            )[0]
            cases.append(
                {
                    "family": family,
                    "task": "instance",
                    "scenario": scenario_name,
                    "expected": segmentation_expected(result, []),
                }
            )

    for family, processor in processors.items():
        scenario_names = ["fusion", "no_mask"]
        if family == "eomt":
            scenario_names.append("assigned_overlap")
        elif family == "one_former":
            scenario_names.append("oneformer_downsample")
        for scenario_name in scenario_names:
            scenario = SCENARIOS[scenario_name]
            label_ids_to_fuse = scenario.get("label_ids_to_fuse", [])
            kwargs = {"target_sizes": [tuple(scenario["target_size"])]}
            if family == "eomt":
                kwargs.update(
                    {
                        "threshold": 0.8,
                        "stuff_classes": label_ids_to_fuse,
                        "size": EOMT_SIZE,
                    }
                )
            else:
                kwargs["label_ids_to_fuse"] = set(label_ids_to_fuse)
            result = processor.post_process_panoptic_segmentation(
                outputs_for(scenario, torch), **kwargs
            )[0]
            cases.append(
                {
                    "family": family,
                    "task": "panoptic",
                    "scenario": scenario_name,
                    "expected": segmentation_expected(result, label_ids_to_fuse),
                }
            )
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transformers-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = verify_clean_git_checkout(
        args.transformers_root, expected_commit=AUDIT_COMMIT, project="Transformers"
    )
    sys.path.insert(0, str(root / "src"))

    import torch
    import transformers
    from transformers.models.eomt.image_processing_eomt import EomtImageProcessor
    from transformers.models.mask2former.image_processing_mask2former import (
        Mask2FormerImageProcessor,
    )
    from transformers.models.maskformer.image_processing_maskformer import (
        MaskFormerImageProcessor,
    )
    from transformers.models.oneformer.image_processing_oneformer import (
        OneFormerImageProcessor,
    )

    imported_from = Path(transformers.__file__).resolve()
    if root not in imported_from.parents:
        raise RuntimeError(f"imported Transformers from {imported_from}, not {root}")

    processors = {
        "mask_former": MaskFormerImageProcessor(size=EOMT_SIZE),
        "mask2_former": Mask2FormerImageProcessor(size=EOMT_SIZE),
        "one_former": OneFormerImageProcessor(size=EOMT_SIZE),
        "eomt": EomtImageProcessor(size=EOMT_SIZE),
    }

    oneformer_error = None
    try:
        processors["one_former"].post_process_instance_segmentation(
            outputs_for(SCENARIOS["primary"], torch),
            target_sizes=[tuple(TARGET_SIZE)],
        )
    except TypeError as error:
        oneformer_error = f"{type(error).__name__}: {error}"
    if oneformer_error is None:
        raise RuntimeError("OneFormer unexpectedly accepted instance output without class metadata")

    fixture = {
        "schema": SCHEMA,
        "upstream": {
            "library": "transformers",
            "version": transformers.__version__,
            "source": SOURCE,
            "commit": AUDIT_COMMIT,
        },
        "generator": GENERATOR,
        "acceptance": acceptance_contract(
            input_locations=["scenarios.*.class_logits", "scenarios.*.mask_logits"],
            input_description=(
                "Deterministic synthetic query/class and query/mask logits covering "
                "five foreground classes, background and low scores, output resize, "
                "stuff fusion, assigned-area overlap filtering, and no-mask results."
            ),
            config_locations=["families", "scenarios.*.target_size"],
            absolute_tolerance=1.0e-5,
        ),
        "families": {
            "mask_former": {"preset": "mask_former"},
            "mask2_former": {
                "preset": "mask2_former",
                "intermediate_size": [384, 384],
            },
            "one_former": {
                "preset": "one_former",
                "instance_metadata": "external",
            },
            "eomt": {"preset": "eomt", "size": EOMT_SIZE},
        },
        "scenarios": SCENARIOS,
        "cases": generate_cases(processors, torch),
        "unsupported": [
            {
                "family": "one_former",
                "task": "instance",
                "reason": "class_info_file and thing class metadata are external model artifacts",
                "upstream_without_metadata": oneformer_error,
            }
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
