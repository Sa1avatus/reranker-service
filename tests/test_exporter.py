from pathlib import Path

import pytest

from reranker_service.exporter import ExportRequest, prepare_artifact

REVISION = "c" * 40


def _request(tmp_path: Path) -> ExportRequest:
    return ExportRequest(
        model_id="BAAI/bge-reranker-v2-m3",
        requested_revision="main",
        artifact_root=tmp_path,
        allowed_models=frozenset({"BAAI/bge-reranker-v2-m3"}),
        precision="fp32",
        opset=18,
        max_length=1024,
        score_transform="sigmoid",
    )


def test_prepare_artifact_writes_verified_immutable_manifest(tmp_path: Path) -> None:
    calls = 0

    def fake_export(request: ExportRequest, revision: str, output: Path) -> None:
        nonlocal calls
        calls += 1
        assert request.model_id == "BAAI/bge-reranker-v2-m3"
        assert revision == REVISION
        (output / "model.onnx").write_bytes(b"onnx")
        (output / "tokenizer.json").write_text("{}", encoding="utf-8")

    request = _request(tmp_path)
    loaded = prepare_artifact(request, resolver=lambda _: REVISION, exporter=fake_export)
    cached = prepare_artifact(request, resolver=lambda _: REVISION, exporter=fake_export)

    assert calls == 1
    assert loaded.directory == cached.directory
    assert loaded.manifest.requested_revision == "main"
    assert loaded.manifest.resolved_revision == REVISION
    assert loaded.manifest.runtime_requirements["cuda"] == "12.8"
    assert loaded.manifest.score_transform == "sigmoid"


def test_prepare_artifact_rejects_non_immutable_resolver_output(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="non-immutable"):
        prepare_artifact(_request(tmp_path), resolver=lambda _: "main")
