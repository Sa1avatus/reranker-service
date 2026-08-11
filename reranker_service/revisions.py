import re

FULL_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
REVISION_REFERENCE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,199}$")


def validate_revision_reference(revision: str) -> str:
    if not REVISION_REFERENCE_PATTERN.fullmatch(revision):
        raise ValueError("revision contains unsupported characters")
    if "\\" in revision or "//" in revision:
        raise ValueError("revision contains an unsafe path separator")
    if any(part in {"", ".", ".."} for part in revision.split("/")):
        raise ValueError("revision contains an unsafe path segment")
    return revision


def resolve_immutable_revision(model_id: str, revision: str) -> str:
    validate_revision_reference(revision)
    normalized = revision.lower()
    if FULL_REVISION_PATTERN.fullmatch(normalized):
        return normalized
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("revision resolution requires huggingface-hub") from exc
    info = HfApi().model_info(model_id, revision=revision, token=False)
    resolved = str(info.sha).lower()
    if not FULL_REVISION_PATTERN.fullmatch(resolved):
        raise RuntimeError("model repository did not resolve to an immutable commit SHA")
    return resolved
