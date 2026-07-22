"""Benchmark Transformers CLIP image preprocessing over an image directory."""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

from PIL import Image
from transformers import CLIPImageProcessor


SUPPORTED_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".ico",
    ".jpg",
    ".jpeg",
    ".png",
    ".pnm",
    ".tif",
    ".tiff",
    ".webp",
}


def main() -> None:
    args = parse_args()
    paths = collect_image_paths(args.image_dir)
    if args.limit is not None:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit(f"no supported images found under {args.image_dir}")

    processor = CLIPImageProcessor(
        do_resize=True,
        size={"shortest_edge": 224},
        resample=Image.Resampling.BICUBIC,
        do_center_crop=True,
        crop_size={"height": 224, "width": 224},
        do_rescale=True,
        rescale_factor=1.0 / 255.0,
        do_normalize=True,
        image_mean=[0.48145466, 0.4578275, 0.40821073],
        image_std=[0.26862954, 0.26130258, 0.27577711],
    )

    with ThreadPoolExecutor(max_workers=args.loader_workers) as loader:
        warmup_paths = cycle_take(paths, args.batch_size * args.warmup_batches)
        for batch in chunks(warmup_paths, args.batch_size):
            process_chunk(processor, batch, args.skip_errors, loader)

        started = time.perf_counter()
        processed = 0
        failed = 0
        batches = 0
        last_shape: tuple[int, ...] | None = None
        for batch in chunks(paths, args.batch_size):
            outcome = process_chunk(processor, batch, args.skip_errors, loader)
            processed += outcome.processed
            failed += outcome.failed
            batches += outcome.batches
            if outcome.last_shape is not None:
                last_shape = outcome.last_shape
        elapsed = time.perf_counter() - started

    print("processor,transformers_clip")
    print(f"image_dir,{args.image_dir}")
    print(f"images,{processed}")
    print(f"failed_images,{failed}")
    print(f"batches,{batches}")
    print(f"batch_size,{args.batch_size}")
    print(f"warmup_batches,{args.warmup_batches}")
    print(f"loader_workers,{args.loader_workers}")
    print("return_tensors,pt")
    print(f"skip_errors,{args.skip_errors}")
    print(f"total_ms,{elapsed * 1000.0:.3f}")
    print(f"images_per_second,{processed / elapsed:.3f}")
    print(f"ms_per_image,{elapsed * 1000.0 / processed:.6f}")
    print(f"last_shape,{list(last_shape) if last_shape is not None else []}")
    print("last_layout,NCHW")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--batch-size", default=32, type=positive_int)
    parser.add_argument("--limit", type=positive_int)
    parser.add_argument("--warmup-batches", default=1, type=non_negative_int)
    parser.add_argument(
        "--loader-workers",
        default=default_loader_workers(),
        type=positive_int,
        help="number of ThreadPoolExecutor workers used for image open/convert",
    )
    parser.add_argument("--skip-errors", action="store_true")
    return parser.parse_args()


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def default_loader_workers() -> int:
    return min(32, (os.cpu_count() or 1) + 4)


def collect_image_paths(image_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def chunks(paths: list[Path], batch_size: int) -> Iterable[list[Path]]:
    for start in range(0, len(paths), batch_size):
        yield paths[start : start + batch_size]


def cycle_take(paths: list[Path], count: int) -> list[Path]:
    if count == 0:
        return []
    return [paths[index % len(paths)] for index in range(count)]


def process_batch(
    processor: CLIPImageProcessor,
    paths: list[Path],
    loader: ThreadPoolExecutor,
):
    images = []
    try:
        for image in loader.map(load_rgb_image, paths):
            images.append(image)
        return processor(images=images, return_tensors="pt")
    finally:
        for image in images:
            image.close()


def load_rgb_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


class ChunkOutcome:
    def __init__(
        self,
        processed: int = 0,
        failed: int = 0,
        batches: int = 0,
        last_shape: tuple[int, ...] | None = None,
    ) -> None:
        self.processed = processed
        self.failed = failed
        self.batches = batches
        self.last_shape = last_shape


def process_chunk(
    processor: CLIPImageProcessor,
    paths: list[Path],
    skip_errors: bool,
    loader: ThreadPoolExecutor,
) -> ChunkOutcome:
    try:
        output = process_batch(processor, paths, loader)
    except Exception as exc:
        if not skip_errors:
            raise
        if len(paths) == 1:
            print(f"skipping {paths[0]}, error={exc}", flush=True)
            return ChunkOutcome(failed=1)
        mid = len(paths) // 2
        left = process_chunk(processor, paths[:mid], True, loader)
        right = process_chunk(processor, paths[mid:], True, loader)
        return ChunkOutcome(
            processed=left.processed + right.processed,
            failed=left.failed + right.failed,
            batches=left.batches + right.batches,
            last_shape=right.last_shape or left.last_shape,
        )

    tensor = output["pixel_values"]
    black_box(tensor)
    return ChunkOutcome(
        processed=len(paths),
        failed=0,
        batches=1,
        last_shape=tuple(tensor.shape),
    )


def black_box(value) -> None:
    if getattr(value, "shape", None) is None:
        raise RuntimeError("processor did not return a tensor-like value")


if __name__ == "__main__":
    main()
