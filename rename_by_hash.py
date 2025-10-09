#!/usr/bin/env python3
"""Rename dataset image/label pairs by image hash and drop duplicate images.

The script walks through an images directory, computes a content hash for each
image file, renames the image and its corresponding label to the hash-based
filename, and removes any duplicate images (same hash). Duplicate labels are
deleted alongside their images.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Set, Tuple


SUPPORTED_IMAGE_EXTENSIONS: Set[str] = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


def compute_hash(file_path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 hash of a file."""
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_label(label_dir: Path, stem: str) -> Optional[Path]:
    """Locate the label file that matches the given stem."""
    candidates = [
        candidate
        for candidate in label_dir.glob(f"{stem}.*")
        if candidate.is_file()
    ]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    for candidate in candidates:
        if candidate.suffix.lower() == ".txt":
            return candidate
    return candidates[0]


def iter_image_files(image_dir: Path) -> Iterable[Path]:
    """Yield supported image files in the directory."""
    for path in sorted(image_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            yield path


def rename_dataset(images_dir: Path, labels_dir: Path) -> Tuple[int, int, int]:
    """Rename images/labels to hash-based names and remove duplicates.

    Returns:
        A tuple of (renamed_images, removed_duplicates, missing_labels).
    """
    seen_hashes: Dict[str, Path] = {}
    rename_queue = []
    duplicates = []
    missing_labels = 0

    for image_path in iter_image_files(images_dir):
        content_hash = compute_hash(image_path)

        if content_hash in seen_hashes:
            duplicates.append((image_path, find_label(labels_dir, image_path.stem)))
            continue

        label_path = find_label(labels_dir, image_path.stem)
        if label_path is None:
            missing_labels += 1

        new_image_path = images_dir / f"{content_hash}{image_path.suffix.lower()}"
        label_suffix = label_path.suffix if label_path else ".txt"
        new_label_path = labels_dir / f"{content_hash}{label_suffix}"

        rename_queue.append((image_path, new_image_path, label_path, new_label_path))
        seen_hashes[content_hash] = new_image_path

    removed_duplicates = 0
    for duplicate_image, duplicate_label in duplicates:
        if duplicate_image.exists():
            duplicate_image.unlink()
            removed_duplicates += 1
        if duplicate_label and duplicate_label.exists():
            duplicate_label.unlink()

    renamed_images = 0
    for image_path, target_image_path, label_path, target_label_path in rename_queue:
        if target_image_path != image_path:
            if target_image_path.exists():
                print(
                    f"Error: target image name already exists: {target_image_path}",
                    file=sys.stderr,
                )
                continue
            image_path.rename(target_image_path)
            renamed_images += 1

        if label_path and target_label_path != label_path:
            if target_label_path.exists():
                print(
                    f"Error: target label name already exists: {target_label_path}",
                    file=sys.stderr,
                )
                continue
            label_path.rename(target_label_path)

    return renamed_images, removed_duplicates, missing_labels


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rename images and labels by image hash, drop duplicates."
    )
    parser.add_argument(
        "--images-dir",
        default="images",
        help="Path to the images directory (default: %(default)s).",
    )
    parser.add_argument(
        "--labels-dir",
        default="labels",
        help="Path to the labels directory (default: %(default)s).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    images_dir = Path(args.images_dir).expanduser().resolve()
    labels_dir = Path(args.labels_dir).expanduser().resolve()

    if not images_dir.exists():
        print(f"Images directory not found: {images_dir}", file=sys.stderr)
        raise SystemExit(1)
    if not labels_dir.exists():
        print(f"Labels directory not found: {labels_dir}", file=sys.stderr)
        raise SystemExit(1)

    renamed, duplicates_removed, missing_labels = rename_dataset(
        images_dir, labels_dir
    )
    print(
        f"Renamed {renamed} image(s); removed {duplicates_removed} duplicate(s); "
        f"missing label(s): {missing_labels}."
    )


if __name__ == "__main__":
    main()
