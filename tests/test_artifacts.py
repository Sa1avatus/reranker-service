import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from reranker_service.artifacts import (
    ArtifactError,
    ArtifactManifest,
    ArtifactStore,
    artifact_directory,
    safe_model_directory_name,
)

REVISION = "a" * 40


def _write_artifact(root: Path, **overrides: object) -> tuple[Path, dict[str, object]]:
    directory = artifact_directory(
        root, "BAAI/bge-reranker-v2-m3", REVISION, "onnx_pairwise", "fp32"
    )
    directory.mkdir(parents=True)
    files = {"model": "model.onnx", "tokenizer": "tokenizer.json"}
    contents = {"model": b"onnx-model", "tokenizer": b'{"version":"1.0"}'}
    for name, relative in files.items():
        (directory / relative).write_bytes(contents[name])
    manifest: dict[str, object] = {
        "version": 1,
        "model_id": "BAAI/bge-reranker-v2-m3",
        "requested_revision": "main",
        "resolved_revision": REVISION,
        "backend": "onnx_pairwise",
        "precision": "fp32",
        "files": files,
        "checksums": {
            name: hashlib.sha256(content).hexdigest() for name, content in contents.items()
        },
        "onnx_opset": 17,
        "export_metadata": {"exporter": "test"},
        "created_at": datetime.now(UTC).isoformat(),
        "runtime_requirements": {"onnxruntime": "1.26.0"},
        "supported_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "preferred_provider": "CUDAExecutionProvider",
        "tokenizer_metadata": {"type": "Tokenizer"},
        "max_length": 1024,
        "score_transform": "sigmoid",
    }
    manifest.update(overrides)
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return directory, manifest


def test_artifact_store_loads_verified_files(tmp_path: Path) -> None:
    directory, _ = _write_artifact(tmp_path)

    loaded = ArtifactStore(tmp_path).load(
        model_id="BAAI/bge-reranker-v2-m3",
        resolved_revision=REVISION,
        backend="onnx_pairwise",
        precision="fp32",
    )

    assert loaded.directory == directory.resolve()
    assert loaded.manifest.requested_revision == "main"
    assert loaded.manifest.score_transform == "sigmoid"
    assert loaded.files["model"].read_bytes() == b"onnx-model"


def test_artifact_store_rejects_checksum_mismatch(tmp_path: Path) -> None:
    directory, _ = _write_artifact(tmp_path)
    (directory / "model.onnx").write_bytes(b"tampered")

    with pytest.raises(ArtifactError, match="checksum"):
        ArtifactStore(tmp_path).load(
            model_id="BAAI/bge-reranker-v2-m3",
            resolved_revision=REVISION,
            backend="onnx_pairwise",
            precision="fp32",
        )


@pytest.mark.parametrize("unsafe_path", ["../model.onnx", "/model.onnx", r"..\\model.onnx"])
def test_manifest_rejects_unsafe_file_paths(tmp_path: Path, unsafe_path: str) -> None:
    _, manifest = _write_artifact(tmp_path)
    manifest["files"] = {"model": unsafe_path, "tokenizer": "tokenizer.json"}

    with pytest.raises(ValidationError, match="artifact file paths"):
        ArtifactManifest.model_validate(manifest)


def test_artifact_store_rejects_unknown_manifest_version(tmp_path: Path) -> None:
    _, manifest = _write_artifact(tmp_path, version=2)

    with pytest.raises(ArtifactError, match="manifest"):
        ArtifactStore(tmp_path).load(
            model_id="BAAI/bge-reranker-v2-m3",
            resolved_revision=REVISION,
            backend="onnx_pairwise",
            precision="fp32",
        )


def test_model_directory_name_is_safe_and_deterministic() -> None:
    assert safe_model_directory_name("BAAI/bge-reranker-v2-m3") == "BAAI%2Fbge-reranker-v2-m3"
    with pytest.raises(ValueError, match="canonical"):
        safe_model_directory_name(" ../model")
