import hashlib
import json
import re
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Literal
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .revisions import FULL_REVISION_PATTERN

SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
ArtifactScalar = str | int | float | bool | None


class ArtifactError(RuntimeError):
    """Raised when a local model artifact is absent, unsafe, or invalid."""


class ArtifactManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = 1
    model_id: str = Field(min_length=1)
    requested_revision: str = Field(min_length=1)
    resolved_revision: str
    backend: Literal["onnx_pairwise"] = "onnx_pairwise"
    precision: Literal["fp32", "fp16"]
    files: dict[str, str]
    checksums: dict[str, str]
    onnx_opset: int = Field(ge=11, le=30)
    export_metadata: dict[str, ArtifactScalar] = Field(default_factory=dict)
    created_at: datetime
    runtime_requirements: dict[str, str] = Field(default_factory=dict)
    supported_providers: list[str] = Field(min_length=1)
    preferred_provider: str
    tokenizer_metadata: dict[str, ArtifactScalar] = Field(default_factory=dict)
    max_length: int = Field(ge=64, le=8192)
    score_transform: Literal["identity", "sigmoid"] = "identity"
    score_output_name: str | None = None
    score_output_index: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_contract(self) -> "ArtifactManifest":
        if not FULL_REVISION_PATTERN.fullmatch(self.resolved_revision):
            raise ValueError("resolved_revision must be a lowercase 40-character commit SHA")
        if set(self.files) != set(self.checksums):
            raise ValueError("checksums must exactly match the declared files")
        if not {"model", "tokenizer"}.issubset(self.files):
            raise ValueError("manifest must declare model and tokenizer files")
        for digest in self.checksums.values():
            if not SHA256_PATTERN.fullmatch(digest):
                raise ValueError("checksums must be lowercase SHA-256 values")
        for relative_path in self.files.values():
            _validate_relative_path(relative_path)
        if self.preferred_provider not in self.supported_providers:
            raise ValueError("preferred_provider must be included in supported_providers")
        return self


class LoadedArtifact(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    manifest: ArtifactManifest
    directory: Path
    files: dict[str, Path]


def _validate_relative_path(value: str) -> None:
    if not value or "\\" in value:
        raise ValueError("artifact file paths must use non-empty POSIX relative paths")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("artifact file paths must stay inside the artifact directory")


def safe_model_directory_name(model_id: str) -> str:
    if not model_id or model_id.strip() != model_id:
        raise ValueError("model_id must be a non-empty canonical repository id")
    encoded = quote(model_id, safe="-_.")
    if encoded in {"", ".", ".."}:
        raise ValueError("model_id cannot be represented safely")
    return encoded


def artifact_directory(
    root: Path,
    model_id: str,
    resolved_revision: str,
    backend: Literal["onnx_pairwise"],
    precision: Literal["fp32", "fp16"],
) -> Path:
    if not FULL_REVISION_PATTERN.fullmatch(resolved_revision):
        raise ValueError("resolved_revision must be a lowercase 40-character commit SHA")
    return root / safe_model_directory_name(model_id) / resolved_revision / backend / precision


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact_file:
        for chunk in iter(lambda: artifact_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def load(
        self,
        *,
        model_id: str,
        resolved_revision: str,
        backend: Literal["onnx_pairwise"],
        precision: Literal["fp32", "fp16"],
    ) -> LoadedArtifact:
        expected_directory = artifact_directory(
            self.root, model_id, resolved_revision, backend, precision
        )
        manifest_path = expected_directory / "manifest.json"
        try:
            root = self.root.resolve(strict=True)
            directory = expected_directory.resolve(strict=True)
            if not directory.is_relative_to(root):
                raise ArtifactError("artifact directory escapes the configured root")
            resolved_manifest = manifest_path.resolve(strict=True)
            if not resolved_manifest.is_relative_to(directory) or not resolved_manifest.is_file():
                raise ArtifactError("artifact manifest escapes the artifact directory")
            manifest_data = json.loads(resolved_manifest.read_text(encoding="utf-8"))
            manifest = ArtifactManifest.model_validate(manifest_data)
        except ArtifactError:
            raise
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise ArtifactError("artifact manifest is missing or invalid") from exc

        if (
            manifest.model_id != model_id
            or manifest.resolved_revision != resolved_revision
            or manifest.backend != backend
            or manifest.precision != precision
        ):
            raise ArtifactError("artifact manifest identity does not match the requested artifact")

        files: dict[str, Path] = {}
        for name, relative_path in manifest.files.items():
            candidate = directory / relative_path
            try:
                resolved = candidate.resolve(strict=True)
            except OSError as exc:
                raise ArtifactError("artifact file is missing") from exc
            if not resolved.is_relative_to(directory) or not resolved.is_file():
                raise ArtifactError("artifact file escapes the artifact directory")
            if sha256_file(resolved) != manifest.checksums[name]:
                raise ArtifactError("artifact checksum verification failed")
            files[name] = resolved
        return LoadedArtifact(manifest=manifest, directory=directory, files=files)
