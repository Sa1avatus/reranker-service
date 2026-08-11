import argparse
import json
import os
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from .artifacts import (
    ArtifactError,
    ArtifactManifest,
    ArtifactStore,
    LoadedArtifact,
    artifact_directory,
    sha256_file,
)
from .revisions import FULL_REVISION_PATTERN, resolve_immutable_revision


@dataclass(frozen=True)
class ExportRequest:
    model_id: str
    requested_revision: str
    artifact_root: Path
    allowed_models: frozenset[str]
    precision: Literal["fp32", "fp16"]
    opset: int
    max_length: int
    score_transform: Literal["identity", "sigmoid"]
    score_output_name: str | None = None
    score_output_index: int | None = None
    export_device: Literal["cpu", "cuda"] = "cpu"


ArtifactExporter = Callable[[ExportRequest, str, Path], None]
RevisionResolver = Callable[[ExportRequest], str]


def resolve_revision(request: ExportRequest) -> str:
    if request.model_id not in request.allowed_models:
        raise ValueError("model repository is not in the exporter allowlist")
    return resolve_immutable_revision(request.model_id, request.requested_revision)


def _export_with_optimum(request: ExportRequest, resolved_revision: str, output: Path) -> None:
    try:
        from optimum.exporters.onnx import main_export
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("ONNX export requires the exporter dependencies") from exc
    main_export(
        model_name_or_path=request.model_id,
        output=output,
        task="text-classification",
        opset=request.opset,
        device=request.export_device,
        dtype=request.precision,
        monolith=True,
        revision=resolved_revision,
        trust_remote_code=False,
        do_validation=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        request.model_id,
        revision=resolved_revision,
        trust_remote_code=False,
        use_fast=True,
    )
    tokenizer.save_pretrained(output)


def prepare_artifact(
    request: ExportRequest,
    *,
    resolver: RevisionResolver = resolve_revision,
    exporter: ArtifactExporter = _export_with_optimum,
) -> LoadedArtifact:
    resolved_revision = resolver(request).lower()
    if not FULL_REVISION_PATTERN.fullmatch(resolved_revision):
        raise RuntimeError("revision resolver returned a non-immutable revision")
    target = artifact_directory(
        request.artifact_root,
        request.model_id,
        resolved_revision,
        "onnx_pairwise",
        request.precision,
    )
    store = ArtifactStore(request.artifact_root)
    if target.exists():
        return store.load(
            model_id=request.model_id,
            resolved_revision=resolved_revision,
            backend="onnx_pairwise",
            precision=request.precision,
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    root = request.artifact_root.resolve(strict=True)
    if not target.parent.resolve(strict=True).is_relative_to(root):
        raise ArtifactError("artifact export directory escapes the configured root")
    staging = Path(tempfile.mkdtemp(prefix=".export-", dir=target.parent))
    try:
        exporter(request, resolved_revision, staging)
        model_path = staging / "model.onnx"
        tokenizer_path = staging / "tokenizer.json"
        if not model_path.is_file() or not tokenizer_path.is_file():
            raise ArtifactError("exporter did not produce model.onnx and tokenizer.json")
        files = {"model": "model.onnx", "tokenizer": "tokenizer.json"}
        manifest = ArtifactManifest(
            model_id=request.model_id,
            requested_revision=request.requested_revision,
            resolved_revision=resolved_revision,
            precision=request.precision,
            files=files,
            checksums={
                "model": sha256_file(model_path),
                "tokenizer": sha256_file(tokenizer_path),
            },
            onnx_opset=request.opset,
            export_metadata={
                "exporter": "optimum",
                "export_device": request.export_device,
                "validation": True,
            },
            created_at=datetime.now(UTC),
            runtime_requirements={
                "onnxruntime-gpu": "1.26.0",
                "cuda": "12.8",
                "cudnn": "9.x",
            },
            supported_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            preferred_provider="CUDAExecutionProvider",
            tokenizer_metadata={"file": "tokenizer.json", "fast": True},
            max_length=request.max_length,
            score_transform=request.score_transform,
            score_output_name=request.score_output_name,
            score_output_index=request.score_output_index,
        )
        (staging / "manifest.json").write_text(
            json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        staging.replace(target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return store.load(
        model_id=request.model_id,
        resolved_revision=resolved_revision,
        backend="onnx_pairwise",
        precision=request.precision,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export an allowlisted reranker to ONNX")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path(os.environ.get("RERANKER_ARTIFACT_ROOT", "/models/artifacts")),
    )
    parser.add_argument(
        "--allowlist",
        default=os.environ.get("RERANKER_MODEL_ALLOWLIST", "BAAI/bge-reranker-v2-m3"),
    )
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--score-transform", choices=("identity", "sigmoid"), required=True)
    parser.add_argument("--score-output-name")
    parser.add_argument("--score-output-index", type=int)
    parser.add_argument("--export-device", choices=("cpu", "cuda"), default="cpu")
    return parser


def main() -> None:
    args = _parser().parse_args()
    request = ExportRequest(
        model_id=args.model_id,
        requested_revision=args.revision,
        artifact_root=args.artifact_root,
        allowed_models=frozenset(x.strip() for x in args.allowlist.split(",") if x.strip()),
        precision=args.precision,
        opset=args.opset,
        max_length=args.max_length,
        score_transform=args.score_transform,
        score_output_name=args.score_output_name,
        score_output_index=args.score_output_index,
        export_device=args.export_device,
    )
    artifact = prepare_artifact(request)
    print(artifact.directory)
