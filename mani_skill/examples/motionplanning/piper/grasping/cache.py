"""Content-addressed, process-safe cache for object-local grasp proposals."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import pathlib
import uuid
from contextlib import contextmanager
from typing import Any

import numpy as np

from .contracts import FRAME_CONVENTION_VERSION, GraspCandidate, GraspProviderName
from .geometry import GEOMETRY_PREPROCESS_VERSION, ResolvedObjectGeometry

CACHE_SCHEMA_VERSION = "piper_grasp_cache_v1"


def dependency_files_hash(paths: list[pathlib.Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted((path.resolve() for path in paths), key=str):
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(str(path).encode())
        contents = path.read_bytes()
        digest.update(len(contents).to_bytes(8, "little"))
        digest.update(contents)
    return digest.hexdigest()


def cache_key(
    geometry: ResolvedObjectGeometry,
    *,
    provider: GraspProviderName,
    provider_version: str,
    provider_config_fingerprint: str,
    gripper_geometry_hash: str,
) -> str:
    payload = {
        "schema": CACHE_SCHEMA_VERSION,
        "frame": FRAME_CONVENTION_VERSION,
        "mesh_preprocess": GEOMETRY_PREPROCESS_VERSION,
        "object": geometry.object_spec.stable_id,
        "geometry": geometry.canonical_geometry_hash,
        "provider": GraspProviderName(provider).value,
        "provider_version": provider_version,
        "provider_config": provider_config_fingerprint,
        "gripper_geometry": gripper_geometry_hash,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _candidate_payload(candidates: list[GraspCandidate]) -> dict[str, np.ndarray]:
    return {
        "candidate_ids": np.asarray([item.candidate_id for item in candidates]),
        "sources": np.asarray([item.source.value for item in candidates]),
        "object_T_tcp": np.stack([item.object_T_tcp for item in candidates]),
        "required_width": np.asarray(
            [item.required_width for item in candidates], dtype=np.float64
        ),
        "proposal_score": np.asarray(
            [item.proposal_score for item in candidates], dtype=np.float64
        ),
        "contact_points": np.stack([item.contact_points for item in candidates]),
        "metadata_json": np.asarray(
            [json.dumps(item.metadata, sort_keys=True) for item in candidates]
        ),
    }


def _payload_hash(payload: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(payload):
        array = np.ascontiguousarray(payload[key])
        digest.update(key.encode())
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


class GraspCache:
    def __init__(self, root: pathlib.Path):
        self.root = pathlib.Path(root)

    def _paths(self, key: str) -> tuple[pathlib.Path, pathlib.Path]:
        if len(key) != 64 or any(
            character not in "0123456789abcdef" for character in key
        ):
            raise ValueError("cache key must be a lowercase SHA-256 digest")
        directory = self.root / key[:2]
        return directory / f"{key}.npz", directory / f"{key}.json"

    @contextmanager
    def _lock(self, key: str, *, exclusive: bool):
        npz_path, _ = self._paths(key)
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = npz_path.with_suffix(".lock")
        with lock_path.open("a+b") as handle:
            fcntl.flock(
                handle.fileno(),
                fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH,
            )
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def store(
        self,
        key: str,
        candidates: list[GraspCandidate],
        *,
        manifest: dict[str, Any],
        failure: str | None = None,
    ) -> None:
        if bool(candidates) == bool(failure):
            raise ValueError("cache entry must contain candidates or one failure")
        npz_path, json_path = self._paths(key)
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        payload = _candidate_payload(candidates) if candidates else {}
        payload_sha256 = _payload_hash(payload)
        metadata = {
            **manifest,
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "frame_convention_version": FRAME_CONVENTION_VERSION,
            "candidate_count": len(candidates),
            "failure": failure,
            "canonical_candidate_payload_sha256": payload_sha256,
        }
        token = f"{os.getpid()}.{uuid.uuid4().hex}"
        temporary_npz = npz_path.parent / f".{key}.npz.{token}.tmp"
        temporary_json = json_path.parent / f".{key}.json.{token}.tmp"
        try:
            with temporary_npz.open("wb") as handle:
                np.savez_compressed(handle, **payload)
                handle.flush()
                os.fsync(handle.fileno())
            with temporary_json.open("w") as handle:
                json.dump(metadata, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            with self._lock(key, exclusive=True):
                if npz_path.exists() or json_path.exists():
                    existing, existing_manifest = self._load_unlocked(key)
                    existing_hash = existing_manifest[
                        "canonical_candidate_payload_sha256"
                    ]
                    if (
                        existing_hash != payload_sha256
                        or bool(existing) != bool(candidates)
                        or existing_manifest["failure"] != failure
                    ):
                        raise RuntimeError(
                            f"non-deterministic cache collision for {key}"
                        )
                    return
                os.replace(temporary_npz, npz_path)
                os.replace(temporary_json, json_path)
                _, verified = self._load_unlocked(key)
                if verified["canonical_candidate_payload_sha256"] != payload_sha256:
                    raise RuntimeError(f"cache verification failed for {key}")
        finally:
            temporary_npz.unlink(missing_ok=True)
            temporary_json.unlink(missing_ok=True)

    def load(self, key: str) -> tuple[list[GraspCandidate], dict[str, Any]]:
        with self._lock(key, exclusive=False):
            return self._load_unlocked(key)

    def _load_unlocked(self, key: str) -> tuple[list[GraspCandidate], dict[str, Any]]:
        npz_path, json_path = self._paths(key)
        if not npz_path.is_file() or not json_path.is_file():
            raise FileNotFoundError(f"Incomplete grasp cache entry: {key}")
        manifest = json.loads(json_path.read_text())
        with np.load(npz_path, allow_pickle=False) as archive:
            payload = {name: archive[name] for name in archive.files}
        if _payload_hash(payload) != manifest["canonical_candidate_payload_sha256"]:
            raise RuntimeError(f"Corrupt grasp cache payload: {key}")
        if manifest["failure"] is not None:
            return [], manifest
        required = {
            "candidate_ids",
            "sources",
            "object_T_tcp",
            "required_width",
            "proposal_score",
            "contact_points",
            "metadata_json",
        }
        if set(payload) != required:
            raise RuntimeError(f"Unexpected grasp cache arrays: {sorted(payload)}")
        candidates = [
            GraspCandidate(
                candidate_id=str(payload["candidate_ids"][index]),
                source=str(payload["sources"][index]),
                object_T_tcp=payload["object_T_tcp"][index],
                required_width=float(payload["required_width"][index]),
                proposal_score=float(payload["proposal_score"][index]),
                contact_points=payload["contact_points"][index],
                metadata=json.loads(str(payload["metadata_json"][index])),
            )
            for index in range(len(payload["candidate_ids"]))
        ]
        return candidates, manifest
