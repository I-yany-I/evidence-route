"""Prepare commit-pinned NLTK metric assets without using the mutable NLTK index."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path, PurePosixPath

DEFAULT_TIMEOUT_S = 60.0
DEFAULT_MAX_MEMBER_BYTES = 256 * 1024 * 1024


def safe_member_path(name: str) -> Path:
    """Return a relative extraction path, rejecting traversal and platform escapes."""

    if not isinstance(name, str) or not name or "\x00" in name:
        raise ValueError(f"unsafe ZIP member: {name!r}")
    normalized = name.replace("\\", "/")
    candidate = PurePosixPath(normalized)
    if candidate.is_absolute() or candidate.drive or re.match(r"^[A-Za-z]:", normalized):
        raise ValueError(f"unsafe ZIP member: {name!r}")
    parts = tuple(part for part in candidate.parts if part not in ("", "."))
    if not parts or any(part == ".." for part in parts):
        raise ValueError(f"unsafe ZIP member: {name!r}")
    return Path(*parts)


def _is_symlink(info: zipfile.ZipInfo) -> bool:
    mode = (info.external_attr >> 16) & 0xFFFF
    return stat.S_ISLNK(mode)


def _extract_zip_into(
    archive_path: Path | str,
    output_root: Path | str,
    *,
    max_member_bytes: int = DEFAULT_MAX_MEMBER_BYTES,
) -> None:
    """Extract into an owned staging tree after validating every ZIP member."""

    archive_path = Path(archive_path)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    seen: set[Path] = set()
    with zipfile.ZipFile(archive_path) as archive:
        entries: list[tuple[zipfile.ZipInfo, Path]] = []
        for info in archive.infolist():
            relative = safe_member_path(info.filename)
            if relative in seen:
                raise ValueError(f"duplicate ZIP member: {info.filename}")
            seen.add(relative)
            if _is_symlink(info):
                raise ValueError(f"symlink ZIP member is not allowed: {info.filename}")
            destination = (output_root / relative).resolve()
            try:
                destination.relative_to(output_root.resolve())
            except ValueError as exc:
                raise ValueError(f"unsafe ZIP member: {info.filename!r}") from exc
            if info.file_size > max_member_bytes:
                raise ValueError(f"ZIP member exceeds size limit: {info.filename}")
            entries.append((info, relative))

        for info, relative in entries:
            destination = (output_root / relative).resolve()
            if info.is_dir() or info.filename.endswith(("/", "\\")):
                if destination.exists() and not destination.is_dir():
                    raise ValueError(f"duplicate ZIP member: {info.filename}")
                destination.mkdir(parents=True, exist_ok=True)
                continue
            if destination.exists():
                raise ValueError(f"duplicate ZIP member: {info.filename}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            written = 0
            with archive.open(info, "r") as source, destination.open("wb") as target:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > max_member_bytes:
                        raise ValueError(f"ZIP member exceeds size limit: {info.filename}")
                    target.write(chunk)
            if written != info.file_size:
                raise ValueError(f"ZIP member size mismatch: {info.filename}")


def extract_checked_zip(
    archive_path: Path | str,
    output_root: Path | str,
    *,
    max_member_bytes: int = DEFAULT_MAX_MEMBER_BYTES,
) -> None:
    """Extract a ZIP atomically, never deleting a pre-existing caller directory."""

    archive_path = Path(archive_path)
    output_root = Path(output_root)
    if output_root.exists():
        raise FileExistsError(f"extraction output already exists: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_root.name}-", dir=output_root.parent))
    try:
        _extract_zip_into(archive_path, stage, max_member_bytes=max_member_bytes)
        os.replace(stage, output_root)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def tree_sha256(root: Path | str) -> str:
    """Hash extracted files in sorted relative-path order (excluding the receipt)."""

    root = Path(root)
    digest = hashlib.sha256()
    files = (
        item
        for item in root.rglob("*")
        if item.is_file()
        and item.relative_to(root).as_posix() != "PREPARATION_RECEIPT.json"
    )
    for path in sorted(files, key=lambda p: p.as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _read_spec(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid NLTK source spec: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("NLTK source spec must be a JSON object")
    repository = payload.get("repository")
    commit = payload.get("commit")
    files = payload.get("files")
    if not isinstance(repository, str) or not re.fullmatch(
        r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository
    ):
        raise ValueError("NLTK source spec requires a repository owner/name")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("NLTK commit must be a 40-character lowercase SHA-1")
    if not isinstance(files, dict) or not files:
        raise ValueError("NLTK source spec requires files")
    for relative, metadata in files.items():
        if not isinstance(relative, str) or not isinstance(metadata, dict):
            raise ValueError("NLTK file entries must map paths to metadata")
        if not isinstance(metadata.get("size"), int) or metadata["size"] < 0:
            raise ValueError(f"invalid NLTK file size: {relative}")
        digest = metadata.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64 or any(
            char not in "0123456789abcdef" for char in digest
        ):
            raise ValueError(f"invalid NLTK file SHA-256: {relative}")
        safe_member_path(relative)
        if not relative.startswith("packages/") or not relative.endswith(".zip"):
            raise ValueError(f"NLTK source must be a packages ZIP: {relative}")
    return payload


def _download(url: str, *, timeout_s: float = DEFAULT_TIMEOUT_S) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "evidence-route/0.1"})
    with urllib.request.urlopen(request, timeout=timeout_s) as response:  # noqa: S310
        return response.read()


def _package_destination(relative: str) -> tuple[Path, str]:
    package = PurePosixPath(relative)
    # packages/tokenizers/punkt_tab.zip -> tokenizers / punkt_tab
    category = package.parts[1]
    name = package.name.removesuffix(".zip")
    if category not in {"tokenizers", "corpora", "taggers", "chunkers"} or not name:
        raise ValueError(f"unsupported NLTK package path: {relative}")
    return Path(category), name


def _extract_package(
    archive_path: Path,
    category_root: Path,
    package_name: str,
    *,
    max_member_bytes: int,
) -> None:
    """Extract either a package-rooted archive or a flat archive into NLTK's layout."""

    with zipfile.ZipFile(archive_path) as archive:
        members = [
            safe_member_path(info.filename)
            for info in archive.infolist()
            if not info.is_dir()
        ]
    rooted = bool(members) and all(path.parts[0] == package_name for path in members)
    target = category_root if rooted else category_root / package_name
    _extract_zip_into(archive_path, target, max_member_bytes=max_member_bytes)


def _receipt_matches(
    output_root: Path,
    spec: dict[str, object],
) -> dict[str, object] | None:
    """Validate an existing installation before allowing an idempotent reuse."""

    receipt_path = output_root / "PREPARATION_RECEIPT.json"
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(receipt, dict):
        return None
    if (
        receipt.get("schema_version") != "1"
        or receipt.get("repository") != spec.get("repository")
        or receipt.get("commit") != spec.get("commit")
    ):
        return None
    expected_files = spec.get("files")
    actual_files = receipt.get("files")
    if not isinstance(expected_files, dict) or not isinstance(actual_files, dict):
        return None
    if set(expected_files) != set(actual_files):
        return None
    repository = str(spec["repository"])
    commit = str(spec["commit"])
    for relative, metadata in expected_files.items():
        if not isinstance(metadata, dict) or not isinstance(actual_files.get(relative), dict):
            return None
        actual = actual_files[relative]
        expected_url = f"https://raw.githubusercontent.com/{repository}/{commit}/{relative}"
        if (
            actual.get("size") != metadata.get("size")
            or actual.get("sha256") != metadata.get("sha256")
            or actual.get("url") != expected_url
        ):
            return None
    recorded_tree = receipt.get("extracted_tree_sha256")
    if not isinstance(recorded_tree, str) or recorded_tree != tree_sha256(output_root):
        return None
    return receipt


def _install_stage(stage: Path, output_root: Path, *, force: bool) -> None:
    """Swap a fully prepared tree into place, retaining a rollback directory until done."""

    if not output_root.exists():
        os.replace(stage, output_root)
        return
    if not force:
        raise FileExistsError(f"output root already exists: {output_root}")
    backup = output_root.with_name(f".{output_root.name}.backup-{os.getpid()}")
    if backup.exists():
        raise FileExistsError(f"stale NLTK backup exists: {backup}")
    os.replace(output_root, backup)
    try:
        os.replace(stage, output_root)
    except Exception:
        os.replace(backup, output_root)
        raise
    shutil.rmtree(backup, ignore_errors=True)


def prepare_from_spec(
    source_spec: Path | str,
    output_root: Path | str,
    *,
    fetch: Callable[[str], bytes] | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    max_member_bytes: int = DEFAULT_MAX_MEMBER_BYTES,
    force: bool = False,
) -> dict[str, object]:
    """Download, verify and atomically extract all packages in a source spec."""

    source_spec = Path(source_spec)
    output_root = Path(output_root)
    spec = _read_spec(source_spec)
    repository = str(spec["repository"])
    commit = str(spec["commit"])
    files = spec["files"]
    assert isinstance(files, dict)
    if output_root.exists():
        existing = _receipt_matches(output_root, spec)
        if existing is not None and not force:
            return existing
        if existing is None and not force:
            raise ValueError(f"existing NLTK output failed integrity: {output_root}")

    fetcher = fetch or (lambda url: _download(url, timeout_s=timeout_s))
    stage_parent = output_root.parent
    stage_parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_root.name}-", dir=stage_parent))
    archive_dir = Path(tempfile.mkdtemp(prefix="evidence-route-nltk-"))
    downloaded: dict[str, dict[str, object]] = {}
    try:
        for relative in sorted(files):
            metadata = files[relative]
            assert isinstance(metadata, dict)
            url = f"https://raw.githubusercontent.com/{repository}/{commit}/{relative}"
            payload = fetcher(url)
            if not isinstance(payload, (bytes, bytearray)):
                raise ValueError(f"fetcher returned non-bytes for {relative}")
            payload = bytes(payload)
            expected_size = int(metadata["size"])
            expected_digest = str(metadata["sha256"])
            if len(payload) != expected_size:
                raise ValueError(f"NLTK download size mismatch: {relative}")
            actual_digest = hashlib.sha256(payload).hexdigest()
            if actual_digest != expected_digest:
                raise ValueError(f"NLTK download SHA-256 mismatch: {relative}")
            archive_path = archive_dir / Path(relative).name
            archive_path.write_bytes(payload)
            category, package_name = _package_destination(relative)
            _extract_package(
                archive_path,
                stage / category,
                package_name,
                max_member_bytes=max_member_bytes,
            )
            downloaded[relative] = {
                "size": len(payload),
                "sha256": actual_digest,
                "url": url,
            }

        extracted_digest = tree_sha256(stage)
        receipt: dict[str, object] = {
            "schema_version": "1",
            "repository": repository,
            "commit": commit,
            "files": downloaded,
            "extracted_tree_sha256": extracted_digest,
        }
        receipt_path = stage / "PREPARATION_RECEIPT.json"
        temporary = receipt_path.with_name(f".{receipt_path.name}.tmp")
        temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, receipt_path)
        _install_stage(stage, output_root, force=force)
        return receipt
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(archive_dir, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare pinned NLTK data assets.")
    parser.add_argument("--source-spec", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--timeout-s", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--max-member-bytes", type=int, default=DEFAULT_MAX_MEMBER_BYTES)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    receipt = prepare_from_spec(
        args.source_spec,
        args.output_root,
        timeout_s=args.timeout_s,
        max_member_bytes=args.max_member_bytes,
        force=args.force,
    )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
