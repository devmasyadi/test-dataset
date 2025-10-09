#!/usr/bin/env python3
"""
Rename dataset image/label pairs by image hash and drop duplicate images.

- Setiap gambar di-hash (SHA-256), lalu image & label di-rename ke <hash>.<ext>
- Duplikat (hash sama) dihapus (beserta labelnya)
- Optimisasi:
  * Hashing paralel (ThreadPoolExecutor)
  * Pencarian label prioritas .txt
  * Opsi --dry-run dan --recursive
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, Optional, Set, Tuple, List

SUPPORTED_IMAGE_EXTENSIONS: Set[str] = {
    ".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp",
}


def compute_hash(file_path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 hash of a file (streaming)."""
    digest = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_label(label_dir: Path, stem: str) -> Optional[Path]:
    """
    Cari file label yang stem-nya sama. Prioritas .txt, jika tidak ada ambil yang lain.
    Contoh: for image 'abc.jpg' -> cari 'labels/abc.txt' dulu.
    """
    txt = label_dir / f"{stem}.txt"
    if txt.is_file():
        return txt

    # fallback: label apa pun yang stem-nya sama
    candidates = list(label_dir.glob(f"{stem}.*"))
    candidates = [c for c in candidates if c.is_file()]
    if not candidates:
        return None
    # pilih .txt kalau ada, kalau tidak ambil kandidat pertama
    for c in candidates:
        if c.suffix.lower() == ".txt":
            return c
    return candidates[0]


def iter_image_files(image_dir: Path, recursive: bool = False) -> Iterable[Path]:
    """Yield supported image files in the directory (non-recursive atau recursive)."""
    if recursive:
        it = image_dir.rglob("*")
    else:
        it = image_dir.iterdir()
    for path in it:
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            yield path


def plan_work(
    images_dir: Path,
    labels_dir: Path,
    recursive: bool = False,
    workers: int = max(1, (os.cpu_count() or 4) // 2),
) -> Tuple[
    List[Tuple[Path, str]],
    Dict[str, List[Path]]
]:
    """
    Menghasilkan:
      - pairs: list (image_path, content_hash)
      - by_hash: dict hash -> list of image paths (untuk deteksi duplikat)
    """
    images = list(iter_image_files(images_dir, recursive=recursive))
    pairs: List[Tuple[Path, str]] = []
    by_hash: Dict[str, List[Path]] = {}

    # Hash paralel
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut2img = {ex.submit(compute_hash, img): img for img in images}
        for fut in as_completed(fut2img):
            img = fut2img[fut]
            try:
                h = fut.result()
            except Exception as e:
                print(f"[WARN] Gagal hash {img}: {e}", file=sys.stderr)
                continue
            pairs.append((img, h))
            by_hash.setdefault(h, []).append(img)
    # Stabilkan urutan untuk deterministik
    pairs.sort(key=lambda x: str(x[0]))
    return pairs, by_hash


def rename_dataset(
    images_dir: Path,
    labels_dir: Path,
    recursive: bool = False,
    dry_run: bool = False,
) -> Tuple[int, int, int]:
    """
    Rename images/labels ke nama hash dan remove duplikat.
    Returns: (renamed_images, removed_duplicates, missing_labels)
    """
    pairs, by_hash = plan_work(images_dir, labels_dir, recursive=recursive)

    # 1) Hapus duplikat: semua kecuali image pertama per hash
    removed_duplicates = 0
    for h, img_list in by_hash.items():
        if len(img_list) <= 1:
            continue
        # keep first (berdasar urutan pairs.sort di plan_work)
        img_list_sorted = sorted(img_list, key=str)
        keep = img_list_sorted[0]
        dups = img_list_sorted[1:]
        for dup in dups:
            # hapus image duplicate
            if not dry_run and dup.exists():
                dup.unlink(missing_ok=True)
            removed_duplicates += 1
            # hapus label duplicate (stem sama sebelum rename)
            dup_label = find_label(labels_dir, dup.stem)
            if dup_label and dup_label.exists() and not dry_run:
                dup_label.unlink(missing_ok=True)

    # 2) Rename yang tersisa ke <hash>.<ext>
    renamed_images = 0
    missing_labels = 0

    # Set setelah duplikat dihapus
    alive_images = {img for img, _ in pairs if img.exists()}

    for img, h in pairs:
        if img not in alive_images:
            continue  # sudah terhapus sebagai duplikat

        label = find_label(labels_dir, img.stem)
        if label is None:
            missing_labels += 1

        # target path
        new_img = img.with_name(f"{h}{img.suffix.lower()}")
        if label:
            label_suffix = label.suffix if label.suffix else ".txt"
        else:
            label_suffix = ".txt"
        new_label = labels_dir / f"{h}{label_suffix}"

        # Deteksi konflik nama (kalau target sudah ada tapi bukan file yang sama)
        image_conflict = new_img.exists() and new_img.resolve() != img.resolve()
        label_conflict = (
            label is not None
            and new_label.exists()
            and new_label.resolve() != label.resolve()
        )
        if image_conflict or label_conflict:
            print(
                f"[SKIP] Konflik nama target "
                f"(image: {new_img}, label: {new_label})",
                file=sys.stderr,
            )
            continue

        # Rename label dulu (supaya tetap match saat image di-rename)
        if label and new_label != label:
            print(f"[RENAME] LABEL: {label.name} -> {new_label.name}")
            if not dry_run:
                label.rename(new_label)

        # Rename image
        if new_img != img:
            print(f"[RENAME] IMAGE: {img.name} -> {new_img.name}")
            if not dry_run:
                img.rename(new_img)
            renamed_images += 1

    return renamed_images, removed_duplicates, missing_labels


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rename images and labels by image hash, drop duplicates."
    )
    p.add_argument("--images-dir", default="images", help="Folder images (default: %(default)s)")
    p.add_argument("--labels-dir", default="labels", help="Folder labels (default: %(default)s)")
    p.add_argument("--recursive", action="store_true", help="Scan subfolder (rglob)")
    p.add_argument("--dry-run", action="store_true", help="Tampilkan rencana tanpa eksekusi")
    return p.parse_args(argv)


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

    renamed, dup_removed, missing = rename_dataset(
        images_dir=images_dir,
        labels_dir=labels_dir,
        recursive=args.recursive,
        dry_run=args.dry_run,
    )
    note = " (DRY-RUN)" if args.dry_run else ""
    print(
        f"Renamed {renamed} image(s); removed {dup_removed} duplicate(s); "
        f"missing label(s): {missing}.{note}"
    )


if __name__ == "__main__":
    main()
