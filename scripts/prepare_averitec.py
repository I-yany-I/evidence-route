"""Prepare a small, pinned and gold-isolated AVeriTeC corpus.

The normal command reads the two metadata files and selected members from the
upstream Hugging Face repository.  Tests can inject a byte fetcher and a
RemoteZip factory, which keeps all security and manifest logic deterministic
without requiring a network connection.
"""

from __future__ import annotations

import argparse
import binascii
import hashlib
import ipaddress
import json
import os
import re
import tempfile
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path, PurePosixPath
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from zipfile import BadZipFile

try:  # Import lazily in practice so pure helpers remain usable in small envs.
    from remotezip import RemoteZip
except ImportError:  # pragma: no cover - dependency is pinned by pyproject.toml
    RemoteZip = None  # type: ignore[assignment,misc]


LABELS: tuple[str, ...] = (
    "Supported",
    "Refuted",
    "Not Enough Evidence",
    "Conflicting Evidence/Cherrypicking",
)
FORBIDDEN_GOLD_KEYS = frozenset(
    {"label", "questions", "justification", "gold", "claim_types"}
)
SCHEMA_VERSION = "1"
DEFAULT_MAX_MEMBER_UNCOMPRESSED_BYTES = 268_435_456
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _validate_split(split: str) -> None:
    if split not in {"train", "dev"}:
        raise ValueError(f"unsupported AVeriTeC split: {split!r}")


def _validate_label(label: object, *, split: str, original_id: int | None = None) -> str:
    if not isinstance(label, str) or label not in LABELS:
        suffix = "" if original_id is None else f" at {split}:{original_id}"
        raise ValueError(f"unexpected AVeriTeC label{suffix}: {label!r}")
    return label


def select_balanced_ids(
    rows: Sequence[Mapping[str, object]], *, split: str, per_label: int, seed: int
) -> list[int]:
    """Select a deterministic, balanced set using original array indices."""

    _validate_split(split)
    if not isinstance(per_label, int) or per_label < 1:
        raise ValueError("per_label must be a positive integer")
    if not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    by_label: dict[str, list[int]] = {label: [] for label in LABELS}
    for original_id, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"{split}:{original_id} is not an object")
        label = _validate_label(row.get("label"), split=split, original_id=original_id)
        by_label[label].append(original_id)
    selected: list[int] = []
    for label in LABELS:
        ranked = sorted(
            by_label[label],
            key=lambda item: hashlib.sha256(
                f"{seed}:{split}:{label}:{item}".encode()
            ).hexdigest(),
        )
        if len(ranked) < per_label:
            raise ValueError(f"{split}:{label} has only {len(ranked)} rows")
        selected.extend(ranked[:per_label])
    return sorted(selected)


def select_stability_ids(
    rows: Sequence[Mapping[str, object]],
    selected_ids: Iterable[int],
    *,
    per_label: int,
    seed: int,
) -> list[int]:
    """Select a deterministic stability subset from already selected dev rows."""

    if not isinstance(per_label, int) or per_label < 1:
        raise ValueError("per_label must be a positive integer")
    selected = list(selected_ids)
    if len(selected) != len(set(selected)):
        raise ValueError("selected_ids contains duplicates")
    by_label: dict[str, list[int]] = {label: [] for label in LABELS}
    for original_id in selected:
        if not isinstance(original_id, int) or not 0 <= original_id < len(rows):
            raise ValueError(f"selected ID is outside {len(rows)} rows: {original_id!r}")
        row = rows[original_id]
        if not isinstance(row, Mapping):
            raise ValueError(f"dev:{original_id} is not an object")
        label = _validate_label(row.get("label"), split="dev", original_id=original_id)
        by_label[label].append(original_id)
    result: list[int] = []
    for label in LABELS:
        ranked = sorted(
            by_label[label],
            key=lambda item: hashlib.sha256(
                f"{seed}:stability:{label}:{item}".encode()
            ).digest(),
        )
        if len(ranked) < per_label:
            raise ValueError(f"dev stability:{label} has only {len(ranked)} rows")
        result.extend(ranked[:per_label])
    return sorted(result)


def _copy_json_value(value: object) -> object:
    # JSON input is expected, and this avoids sharing mutable question lists.
    return json.loads(json.dumps(value, ensure_ascii=False))


def split_runtime_and_gold(
    split: str, original_id: int, row: Mapping[str, object]
) -> tuple[dict[str, object], dict[str, object]]:
    """Split one source row before corpus metadata is added.

    Runtime fields are deliberately a closed four-key set.  The scorer copy
    follows the pinned AVeriTeC paper-era schema and is never imported by the
    graph process.
    """

    _validate_split(split)
    if not isinstance(original_id, int) or original_id < 0:
        raise ValueError("original_id must be a non-negative integer")
    if not isinstance(row, Mapping):
        raise ValueError("AVeriTeC row must be an object")
    claim = row.get("claim")
    if not isinstance(claim, str) or not claim.strip():
        raise ValueError(f"{split}:{original_id} has an empty claim")
    label = _validate_label(row.get("label"), split=split, original_id=original_id)
    questions = row.get("questions", [])
    justification = row.get("justification", "")
    claim_types = row.get("claim_types", [])
    if not isinstance(questions, list):
        raise ValueError(f"{split}:{original_id} questions must be a list")
    if not isinstance(justification, str):
        raise ValueError(f"{split}:{original_id} justification must be a string")
    if not isinstance(claim_types, list):
        raise ValueError(f"{split}:{original_id} claim_types must be a list")
    claim_id = f"{split}-{original_id}"
    runtime = {
        "claim_id": claim_id,
        "original_id": original_id,
        "claim": claim,
        "split": split,
    }
    gold = {
        "claim_id": claim_id,
        "original_id": original_id,
        "claim": claim,
        "label": label,
        "questions": _copy_json_value(questions),
        "justification": justification,
        "claim_types": _copy_json_value(claim_types),
    }
    return runtime, gold


def archive_member(split: str, original_id: int) -> tuple[str, str]:
    """Return the pinned archive path and member path for an original ID."""

    _validate_split(split)
    if not isinstance(original_id, int) or original_id < 0:
        raise ValueError("original_id must be a non-negative integer")
    if split == "dev":
        return (
            "data_store/knowledge_store/dev_knowledge_store.zip",
            f"output_dev/{original_id}.json",
        )
    if original_id < 1000:
        return (
            "data_store/knowledge_store/train/train_0_999.zip",
            f"{original_id}.json",
        )
    if original_id < 2000:
        return (
            "data_store/knowledge_store/train/train_1000_1999.zip",
            f"{original_id}.json",
        )
    return (
        "data_store/knowledge_store/train/train_2000_3067.zip",
        f"data_store/train/{original_id}.json",
    )


def _public_http_url(value: object) -> tuple[str, str]:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("evidence source URL must be a non-empty string")
    url = value.strip()
    parsed = urlparse(url)
    if parsed.scheme.lower() not in {"http", "https"}:
        raise ValueError(f"unsupported evidence URL scheme: {url!r}")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("evidence URL userinfo is not allowed")
    try:
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"invalid evidence URL: {url!r}") from exc
    if hostname is None:
        raise ValueError(f"evidence URL has no hostname: {url!r}")
    hostname = hostname.rstrip(".").lower()
    if not hostname or hostname in {"localhost", "localhost.localdomain"}:
        raise ValueError(f"local/private evidence URL is not allowed: {url!r}")
    if hostname.endswith((".local", ".localhost", ".lan", ".internal", ".home")):
        raise ValueError(f"local/private evidence URL is not allowed: {url!r}")
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        address = None
    if address is not None and not address.is_global:
        raise ValueError(f"local/private evidence URL is not allowed: {url!r}")
    if port is not None and not 1 <= port <= 65535:
        raise ValueError(f"invalid evidence URL port: {url!r}")
    return url, hostname


def normalize_member(
    split: str, original_id: int, source: Sequence[Mapping[str, object]]
) -> Iterator[dict[str, str]]:
    """Yield strict evidence records from one AVeriTeC JSONL member."""

    _validate_split(split)
    if not isinstance(original_id, int) or original_id < 0:
        raise ValueError("original_id must be a non-negative integer")
    if not isinstance(source, Sequence) or isinstance(source, (str, bytes, bytearray)):
        raise ValueError("AVeriTeC member must be a JSON array of source records")
    for source_index, item in enumerate(source):
        if not isinstance(item, Mapping):
            raise ValueError(f"member source record {source_index} is not an object")
        source_url, title = _public_http_url(item.get("url"))
        texts = item.get("url2text")
        if isinstance(texts, str):
            texts = [texts]
        if not isinstance(texts, Sequence) or isinstance(texts, (bytes, bytearray)):
            raise ValueError(f"member source record {source_index} url2text must be a list")
        for sentence_index, sentence in enumerate(texts):
            if not isinstance(sentence, str):
                raise ValueError(
                    f"member source record {source_index} sentence {sentence_index} is not text"
                )
            text = sentence.strip()
            if not text:
                continue
            yield {
                "evidence_id": f"av:{split}:{original_id}:{source_index}:{sentence_index}",
                "title": title,
                "source_url": source_url,
                "text": text,
                "snapshot_sha256": _sha256(text.encode("utf-8")),
            }


def _normalise_member_path(name: object) -> str:
    if not isinstance(name, str) or not name:
        raise ValueError("archive member path must be a non-empty string")
    if "\\" in name or name.startswith("/") or re.match(r"^[A-Za-z]:", name):
        raise ValueError(f"unsafe archive member path: {name!r}")
    path = PurePosixPath(name)
    parts = path.parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"unsafe archive member path: {name!r}")
    normalised = "/".join(parts)
    if normalised != name:
        raise ValueError(f"unsafe archive member path: {name!r}")
    return normalised


def _archive_info_name(info: object) -> str:
    name = getattr(info, "filename", None)
    if name is None:
        name = getattr(info, "orig_filename", None)
    if name is None:
        raise ValueError("archive central-directory entry has no filename")
    return str(name)


def _validated_member_info(archive: object, member_name: str, allowed: set[str], max_bytes: int):
    requested = _normalise_member_path(member_name)
    normalised_allowed = {_normalise_member_path(item) for item in allowed}
    if requested not in normalised_allowed:
        raise ValueError(f"member is not in selected allowlist: {requested}")
    infos = list(archive.infolist())
    by_name: dict[str, object] = {}
    for info in infos:
        raw_name = _archive_info_name(info)
        normalised = _normalise_member_path(raw_name)
        if normalised in by_name:
            raise ValueError(f"duplicate archive member: {normalised}")
        by_name[normalised] = info
    info = by_name.get(requested)
    if info is None:
        raise ValueError(f"selected archive member is absent: {requested}")
    file_size = getattr(info, "file_size", None)
    if not isinstance(file_size, int) or file_size < 0:
        raise ValueError(f"archive member has invalid size: {requested}")
    if file_size > max_bytes:
        raise ValueError(f"member exceeds {max_bytes} bytes: {requested}")
    return info, requested


def _read_stream(stream: object, *, max_bytes: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = stream.read(1024 * 1024)  # type: ignore[attr-defined]
        if chunk == b"":
            break
        if not isinstance(chunk, bytes):
            raise ValueError("archive member stream returned non-bytes")
        total += len(chunk)
        if total > max_bytes:
            raise ValueError(f"member exceeds {max_bytes} bytes while streaming")
        chunks.append(chunk)
    return b"".join(chunks)


def _stream_selected_member_with_info(
    archive: object,
    member_name: str,
    *,
    allowed: set[str],
    max_uncompressed_bytes: int,
) -> tuple[object, str, bytes]:
    if max_uncompressed_bytes < 1:
        raise ValueError("max_uncompressed_bytes must be positive")
    info, normalised = _validated_member_info(
        archive, member_name, allowed, max_uncompressed_bytes
    )
    try:
        stream = archive.open(info)
    except (KeyError, TypeError):
        # A small test double may implement only the string form; RemoteZip
        # and zipfile both support the validated ZipInfo form above.
        stream = archive.open(normalised)
    try:
        payload = _read_stream(stream, max_bytes=max_uncompressed_bytes)
    finally:
        close = getattr(stream, "close", None)
        if close is not None:
            close()
    expected_crc = getattr(info, "CRC", None)
    if isinstance(expected_crc, int) and (binascii.crc32(payload) & 0xFFFFFFFF) != expected_crc:
        raise BadZipFile(f"CRC mismatch for archive member: {normalised}")
    return info, normalised, payload


def stream_selected_member(
    archive: object,
    member_name: str,
    *,
    allowed: set[str],
    max_uncompressed_bytes: int,
) -> bytes:
    """Read exactly one allowlisted ZIP member without extraction helpers."""

    _info, _normalised, payload = _stream_selected_member_with_info(
        archive,
        member_name,
        allowed=allowed,
        max_uncompressed_bytes=max_uncompressed_bytes,
    )
    return payload


def _parse_jsonl_member(payload: bytes, *, member_name: str) -> list[dict[str, object]]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"archive member is not UTF-8: {member_name}") from exc
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {member_name}:{line_number}") from exc
        if isinstance(value, dict):
            rows.append(value)
        elif line_number == 1 and isinstance(value, list):
            # Some archived revisions contain one JSON array despite the JSONL-oriented
            # member name. Accept it only when every element is an object.
            if not all(isinstance(item, dict) for item in value):
                raise ValueError(f"JSON array at {member_name}:1 contains a non-object")
            rows.extend(value)
        else:
            raise ValueError(f"JSONL row at {member_name}:{line_number} is not an object")
    if not rows:
        raise ValueError(f"archive member is empty: {member_name}")
    return rows


def _serialised_corpus(records: Iterable[Mapping[str, str]]) -> tuple[bytes, int]:
    lines: list[bytes] = []
    count = 0
    for record in records:
        if set(record) != {"evidence_id", "title", "source_url", "text", "snapshot_sha256"}:
            raise ValueError("normalized corpus contains an unexpected field")
        lines.append(_canonical_json_bytes(dict(record)) + b"\n")
        count += 1
    if not lines:
        raise ValueError("selected claim has no non-empty public evidence")
    return b"".join(lines), count


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: object) -> str:
    payload = _canonical_json_bytes(value)
    _atomic_write(path, payload)
    sidecar = (_sha256(payload) + "\n").encode("ascii")
    _atomic_write(path.with_suffix(path.suffix + ".sha256"), sidecar)
    return _sha256(payload)


def _load_json_bytes(path: Path) -> tuple[object, bytes]:
    payload = path.read_bytes()
    try:
        return json.loads(payload.decode("utf-8")), payload
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON source: {path}") from exc


def _fetch_url_bytes(url: str, *, timeout_s: float) -> bytes:
    request = Request(url, headers={"User-Agent": "EvidenceRoute/0.1"})
    try:
        with urlopen(request, timeout=timeout_s) as response:  # nosec B310 - pinned URL
            return response.read()
    except (OSError, URLError) as exc:
        raise RuntimeError(f"failed to download pinned source: {url}") from exc


def verify_lfs_pointer_bytes(payload: bytes, *, expected_sha256: str, expected_size: int) -> None:
    """Verify the text pointer returned by a Git-LFS raw endpoint."""

    if not _SHA256_RE.fullmatch(expected_sha256) or expected_size < 0:
        raise ValueError("invalid expected LFS identity")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("LFS pointer is not UTF-8") from exc
    lines = text.splitlines()
    if not lines or lines[0] != "version https://git-lfs.github.com/spec/v1":
        raise ValueError("LFS pointer version is missing or unsupported")
    fields: dict[str, str] = {}
    for line in lines[1:]:
        key, separator, value = line.partition(" ")
        if separator and key in {"oid", "size"}:
            if key in fields:
                raise ValueError(f"LFS pointer has duplicate {key} field")
            fields[key] = value.strip()
    actual_oid = fields.get("oid", "")
    if actual_oid.startswith("sha256:"):
        actual_oid = actual_oid.removeprefix("sha256:")
    if actual_oid != expected_sha256:
        raise ValueError("LFS pointer SHA-256 mismatch")
    if fields.get("size") != str(expected_size):
        raise ValueError("LFS pointer size mismatch")


def _source_url(spec: Mapping[str, object], revision: str, path: str, *, raw: bool) -> str:
    repo = spec.get("huggingface_repo")
    if not isinstance(repo, str) or not repo:
        raise ValueError("source spec has no huggingface_repo")
    base = spec.get("huggingface_base_url", "https://huggingface.co")
    if not isinstance(base, str) or not base.startswith(("http://", "https://")):
        raise ValueError("source spec has an invalid huggingface_base_url")
    action = "raw" if raw else "resolve"
    return f"{base.rstrip('/')}/{repo}/{action}/{revision}/{path}"


def _validate_source_spec(spec: Mapping[str, object]) -> None:
    if spec.get("dataset") != "AVeriTeC":
        raise ValueError("source spec dataset must be AVeriTeC")
    revision = spec.get("huggingface_revision")
    if not isinstance(revision, str) or not _REVISION_RE.fullmatch(revision):
        raise ValueError("source spec revision must be a full 40-character SHA")
    metadata = spec.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("source spec metadata is missing")
    for split in ("train", "dev"):
        entry = metadata.get(f"data/{split}.json")
        if not isinstance(entry, Mapping):
            raise ValueError(f"source spec metadata is missing data/{split}.json")
        digest = entry.get("sha256")
        size = entry.get("size")
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            raise ValueError(f"invalid metadata hash for {split}")
        if not isinstance(size, int) or size < 1:
            raise ValueError(f"invalid metadata size for {split}")
    store = spec.get("knowledge_store")
    if not isinstance(store, Mapping) or not store:
        raise ValueError("source spec knowledge_store is missing")
    for path, entry in store.items():
        if not isinstance(path, str) or not isinstance(entry, Mapping):
            raise ValueError("invalid knowledge_store entry")
        digest = entry.get("lfs_oid_sha256")
        size = entry.get("size")
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            raise ValueError(f"invalid LFS hash for {path}")
        if not isinstance(size, int) or size < 1:
            raise ValueError(f"invalid LFS size for {path}")


def _load_split_rows(
    split: str,
    spec: Mapping[str, object],
    *,
    source_spec_dir: Path,
    timeout_s: float,
    fetcher: Callable[[str, float], bytes],
) -> list[dict[str, object]]:
    metadata_path = f"data/{split}.json"
    entry = spec["metadata"][metadata_path]  # type: ignore[index]
    local_files = spec.get("local_metadata_files")
    payload: bytes
    if isinstance(local_files, Mapping) and isinstance(local_files.get(split), str):
        local_path = Path(str(local_files[split]))
        if not local_path.is_absolute():
            local_path = source_spec_dir / local_path
        payload = local_path.read_bytes()
    else:
        url = _source_url(spec, str(spec["huggingface_revision"]), metadata_path, raw=False)
        payload = fetcher(url, timeout_s)
    expected_size = entry["size"]  # type: ignore[index]
    expected_sha = entry["sha256"]  # type: ignore[index]
    if len(payload) != expected_size:
        raise ValueError(f"{metadata_path} size mismatch")
    if _sha256(payload) != expected_sha:
        raise ValueError(f"{metadata_path} SHA-256 mismatch")
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"{metadata_path} must contain an array")
    rows: list[dict[str, object]] = []
    for index, row in enumerate(value):
        if not isinstance(row, dict):
            raise ValueError(f"{metadata_path}:{index} is not an object")
        rows.append(row)
    return rows


def _archive_context(
    spec: Mapping[str, object],
    archive_path: str,
    *,
    timeout_s: float,
    archive_factory: Callable[[str, float], object] | None,
):
    if archive_factory is not None:
        return archive_factory(archive_path, timeout_s)
    if RemoteZip is None:
        raise RuntimeError("remotezip is required for AVeriTeC preparation")
    url = _source_url(spec, str(spec["huggingface_revision"]), archive_path, raw=False)
    return RemoteZip(url, timeout=timeout_s)


def _close_context(value: object) -> None:
    close = getattr(value, "close", None)
    if close is not None:
        close()


def _make_runtime_item(
    runtime: Mapping[str, object], *, corpus_payload: bytes, corpus_relpath: str, record_count: int
) -> dict[str, object]:
    claim = runtime["claim"]
    if not isinstance(claim, str):
        raise ValueError("runtime claim is not text")
    item = dict(runtime)
    item.update(
        {
            "claim_sha256": _sha256(claim.encode("utf-8")),
            "corpus_relpath": corpus_relpath,
            "corpus_sha256": _sha256(corpus_payload),
            "corpus_bytes": len(corpus_payload),
            "corpus_records": record_count,
        }
    )
    _assert_runtime_safe(item)
    return item


def _assert_runtime_safe(value: object) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if key in FORBIDDEN_GOLD_KEYS:
                raise ValueError(f"runtime manifest contains forbidden field: {key}")
            _assert_runtime_safe(nested)
    elif isinstance(value, list):
        for nested in value:
            _assert_runtime_safe(nested)


def _manifest(
    items: list[dict[str, object]],
    *,
    revision: str,
    source_metadata_sha256: str,
    seed: int,
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": "AVeriTeC",
        "revision": revision,
        "source_metadata_sha256": source_metadata_sha256,
        "seed": seed,
        "items": items,
    }


def _gold_manifest(
    items: list[dict[str, object]],
    *,
    revision: str,
    source_metadata_sha256: str,
    runtime_manifest_sha256: str,
    seed: int,
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": "AVeriTeC",
        "revision": revision,
        "source_metadata_sha256": source_metadata_sha256,
        "runtime_manifest_sha256": runtime_manifest_sha256,
        "seed": seed,
        "items": items,
    }


def _prepare_source_identity(
    spec: Mapping[str, object],
    *,
    timeout_s: float,
    fetcher: Callable[[str, float], bytes],
) -> list[dict[str, object]]:
    """Verify LFS pointers and return identity records for the receipt."""

    identities: list[dict[str, object]] = []
    revision = str(spec["huggingface_revision"])
    for path, raw_entry in spec["knowledge_store"].items():  # type: ignore[union-attr]
        entry = dict(raw_entry)
        url = _source_url(spec, revision, str(path), raw=True)
        pointer = fetcher(url, timeout_s)
        verify_lfs_pointer_bytes(
            pointer,
            expected_sha256=str(entry["lfs_oid_sha256"]),
            expected_size=int(entry["size"]),
        )
        identities.append(
            {
                "archive_path": path,
                "lfs_oid_sha256": entry["lfs_oid_sha256"],
                "size": entry["size"],
            }
        )
    return identities


def prepare_dataset(
    source_spec: Path | str,
    output_root: Path | str,
    runtime_manifest_root: Path | str,
    scorer_manifest_root: Path | str,
    *,
    seed: int = 20260817,
    calibration_per_label: int = 8,
    dev_per_label: int = 20,
    stability_per_label: int = 5,
    remote_timeout_s: float = 60.0,
    max_member_uncompressed_bytes: int = DEFAULT_MAX_MEMBER_UNCOMPRESSED_BYTES,
    fetcher: Callable[[str, float], bytes] | None = None,
    archive_factory: Callable[[str, float], object] | None = None,
) -> dict[str, object]:
    """Prepare corpora and manifests, returning a redacted preparation summary."""

    source_spec_path = Path(source_spec)
    output_root = Path(output_root)
    runtime_root = Path(runtime_manifest_root)
    scorer_root = Path(scorer_manifest_root)
    if runtime_root.resolve() == scorer_root.resolve():
        raise ValueError("runtime and scorer manifest roots must be separate")
    if max_member_uncompressed_bytes < 1:
        raise ValueError("max_member_uncompressed_bytes must be positive")
    spec_value, source_bytes = _load_json_bytes(source_spec_path)
    if not isinstance(spec_value, dict):
        raise ValueError("source spec must be a JSON object")
    spec = spec_value
    _validate_source_spec(spec)
    revision = str(spec["huggingface_revision"])
    metadata_sha = _sha256(source_bytes)
    fetch = fetcher or (lambda url, timeout: _fetch_url_bytes(url, timeout_s=timeout))

    train_rows = _load_split_rows(
        "train",
        spec,
        source_spec_dir=source_spec_path.parent,
        timeout_s=remote_timeout_s,
        fetcher=fetch,
    )
    dev_rows = _load_split_rows(
        "dev",
        spec,
        source_spec_dir=source_spec_path.parent,
        timeout_s=remote_timeout_s,
        fetcher=fetch,
    )
    calibration_ids = select_balanced_ids(
        train_rows, split="train", per_label=calibration_per_label, seed=seed
    )
    dev_ids = select_balanced_ids(dev_rows, split="dev", per_label=dev_per_label, seed=seed)
    stability_ids = select_stability_ids(
        dev_rows, dev_ids, per_label=stability_per_label, seed=seed
    )
    lfs_identities = _prepare_source_identity(
        spec, timeout_s=remote_timeout_s, fetcher=fetch
    )
    selections: dict[str, tuple[str, list[int]]] = {
        "calibration": ("train", calibration_ids),
        "dev": ("dev", dev_ids),
        "stability": ("dev", stability_ids),
    }
    unique_selected: list[tuple[str, int]] = []
    for split, ids in selections.values():
        for original_id in ids:
            pair = (split, original_id)
            if pair not in unique_selected:
                unique_selected.append(pair)

    grouped: dict[str, list[tuple[str, int, str]]] = defaultdict(list)
    for split, original_id in unique_selected:
        archive_path, member_path = archive_member(split, original_id)
        if archive_path not in spec["knowledge_store"]:
            raise ValueError(f"source spec has no identity for selected archive: {archive_path}")
        grouped[archive_path].append((split, original_id, member_path))

    stage_parent = output_root.parent if output_root.parent != Path("") else Path(".")
    stage_parent.mkdir(parents=True, exist_ok=True)
    stage_dir = Path(tempfile.mkdtemp(prefix=".averitec-prepare-", dir=stage_parent))
    stage_corpora = stage_dir / "corpora"
    corpus_meta: dict[tuple[str, int], dict[str, object]] = {}
    receipt_members: list[dict[str, object]] = []
    try:
        for archive_path, members in grouped.items():
            archive = _archive_context(
                spec,
                archive_path,
                timeout_s=remote_timeout_s,
                archive_factory=archive_factory,
            )
            try:
                allowed = {member_path for _split, _id, member_path in members}
                for split, original_id, member_path in members:
                    info, normalised_member, payload = _stream_selected_member_with_info(
                        archive,
                        member_path,
                        allowed=allowed,
                        max_uncompressed_bytes=max_member_uncompressed_bytes,
                    )
                    source_records = _parse_jsonl_member(payload, member_name=normalised_member)
                    evidence_payload, record_count = _serialised_corpus(
                        normalize_member(split, original_id, source_records)
                    )
                    corpus_relpath = f"{split}-{original_id}.jsonl"
                    staged_path = stage_corpora / corpus_relpath
                    _atomic_write(staged_path, evidence_payload)
                    _atomic_write(
                        staged_path.with_suffix(staged_path.suffix + ".sha256"),
                        (_sha256(evidence_payload) + "\n").encode("ascii"),
                    )
                    corpus_meta[(split, original_id)] = {
                        "payload": evidence_payload,
                        "record_count": record_count,
                        "corpus_relpath": corpus_relpath,
                    }
                    receipt_members.append(
                        {
                            "split": split,
                            "original_id": original_id,
                            "archive_path": archive_path,
                            "archive_lfs_oid_sha256": spec["knowledge_store"][archive_path][  # type: ignore[index]
                                "lfs_oid_sha256"
                            ],
                            "archive_size": spec["knowledge_store"][archive_path]["size"],  # type: ignore[index]
                            "member_path": normalised_member,
                            "member_crc32": int(getattr(info, "CRC", 0)),
                            "member_compressed_bytes": int(
                                getattr(info, "compress_size", len(payload))
                            ),
                            "member_uncompressed_bytes": int(
                                getattr(info, "file_size", len(payload))
                            ),
                            "member_sha256": _sha256(payload),
                            "corpus_relpath": corpus_relpath,
                            "corpus_sha256": _sha256(evidence_payload),
                            "corpus_bytes": len(evidence_payload),
                            "corpus_records": record_count,
                        }
                    )
            finally:
                _close_context(archive)

        # Every selected pair must have a complete corpus before any public manifest is replaced.
        if set(corpus_meta) != set(unique_selected):
            raise ValueError("preparation did not produce every selected corpus")

        rows_by_split = {"train": train_rows, "dev": dev_rows}
        runtime_by_kind: dict[str, list[dict[str, object]]] = {}
        gold_by_kind: dict[str, list[dict[str, object]]] = {}
        for kind, (split, ids) in selections.items():
            runtime_items: list[dict[str, object]] = []
            gold_items: list[dict[str, object]] = []
            for original_id in ids:
                runtime, gold = split_runtime_and_gold(
                    split, original_id, rows_by_split[split][original_id]
                )
                metadata = corpus_meta[(split, original_id)]
                runtime_items.append(
                    _make_runtime_item(
                        runtime,
                        corpus_payload=metadata["payload"],  # type: ignore[arg-type]
                        corpus_relpath=str(metadata["corpus_relpath"]),
                        record_count=int(metadata["record_count"]),
                    )
                )
                gold_items.append(gold)
            runtime_by_kind[kind] = runtime_items
            if kind != "stability":
                gold_by_kind[kind] = gold_items

        manifests: dict[str, tuple[dict[str, object], Path, Path]] = {}
        for kind, items in runtime_by_kind.items():
            runtime_name = f"averitec_{kind}_runtime.json"
            runtime_value = _manifest(
                items,
                revision=revision,
                source_metadata_sha256=metadata_sha,
                seed=seed,
            )
            runtime_path = runtime_root / runtime_name
            runtime_digest = _sha256(_canonical_json_bytes(runtime_value))
            manifests[f"{kind}_runtime"] = (runtime_value, runtime_path, runtime_root)
            if kind != "stability":
                gold_name = f"averitec_{kind}_gold.json"
                gold_value = _gold_manifest(
                    gold_by_kind[kind],
                    revision=revision,
                    source_metadata_sha256=metadata_sha,
                    runtime_manifest_sha256=runtime_digest,
                    seed=seed,
                )
                manifests[f"{kind}_gold"] = (gold_value, scorer_root / gold_name, scorer_root)

        # Commit staged corpora and all manifests only after all validation above succeeds.
        for path in stage_corpora.glob("*"):
            target = output_root / "corpora" / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(path, target)
        for value, path, _root in manifests.values():
            _atomic_json(path, value)

        totals = {
            "selected_member_count": len(receipt_members),
            "compressed_bytes": sum(item["member_compressed_bytes"] for item in receipt_members),
            "uncompressed_bytes": sum(
                item["member_uncompressed_bytes"] for item in receipt_members
            ),
            "corpus_bytes": sum(item["corpus_bytes"] for item in receipt_members),
            "corpus_records": sum(item["corpus_records"] for item in receipt_members),
        }
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "dataset": "AVeriTeC",
            "revision": revision,
            "source_metadata_sha256": metadata_sha,
            "seed": seed,
            "selected_members": sorted(
                receipt_members, key=lambda item: (str(item["split"]), int(item["original_id"]))
            ),
            "totals": totals,
            "lfs_identities": lfs_identities,
        }
        receipt_path = output_root / "preparation_receipt.json"
        receipt_digest = _atomic_json(receipt_path, receipt)
        return {
            "dataset": "AVeriTeC",
            "revision": revision,
            "source_metadata_sha256": metadata_sha,
            "calibration_count": len(calibration_ids),
            "dev_count": len(dev_ids),
            "stability_count": len(stability_ids),
            "receipt_sha256": receipt_digest,
            "runtime_manifest_paths": [
                str(path)
                for _value, path, _root in manifests.values()
                if "runtime" in path.name
            ],
            "scorer_manifest_paths": [
                str(path)
                for _value, path, _root in manifests.values()
                if "gold" in path.name
            ],
        }
    finally:
        # The directory should be empty after successful corpus moves; remove only our staging tree.
        for child in sorted(stage_dir.rglob("*"), reverse=True):
            if child.is_file() or child.is_symlink():
                child.unlink(missing_ok=True)
            elif child.is_dir():
                child.rmdir()
        stage_dir.rmdir()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-spec", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--runtime-manifest-root", type=Path, required=True)
    parser.add_argument("--scorer-manifest-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--calibration-per-label", type=int, default=8)
    parser.add_argument("--dev-per-label", type=int, default=20)
    parser.add_argument("--stability-per-label", type=int, default=5)
    parser.add_argument("--remote-timeout-s", type=float, default=60.0)
    parser.add_argument(
        "--max-member-uncompressed-bytes",
        type=int,
        default=DEFAULT_MAX_MEMBER_UNCOMPRESSED_BYTES,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = prepare_dataset(
            args.source_spec,
            args.output_root,
            args.runtime_manifest_root,
            args.scorer_manifest_root,
            seed=args.seed,
            calibration_per_label=args.calibration_per_label,
            dev_per_label=args.dev_per_label,
            stability_per_label=args.stability_per_label,
            remote_timeout_s=args.remote_timeout_s,
            max_member_uncompressed_bytes=args.max_member_uncompressed_bytes,
        )
    except (OSError, ValueError, RuntimeError, URLError) as exc:
        build_parser().error(str(exc))
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
