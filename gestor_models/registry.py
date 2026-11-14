"""Model discovery and descriptor registry (encapsulated models + legacy stages)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sys

from .config import LEGACY_STAGE_ORDER, REPO_ROOT, read_json_file

_MODEL_VERSION_RE = re.compile(r'model_version\s*:\s*str\s*=\s*["\']([^"\']+)["\']')


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Descriptor for an encapsulated model folder or legacy stage directory."""

    folder: str
    name: str
    model_version: str
    kind: str  # "encapsulated" or "legacy"
    root_dir: Path
    entrypoint: Path
    self_check_args: tuple[str, ...]
    train_cmd: tuple[str, ...] | None
    heartbeat_path: Path | None
    log_path: Path
    description: str

    def to_dict(self) -> dict[str, object]:
        return {
            "folder": self.folder,
            "name": self.name,
            "model_version": self.model_version,
            "kind": self.kind,
            "entrypoint": str(self.entrypoint.relative_to(REPO_ROOT)).replace("\\", "/"),
            "has_self_check": bool(self.self_check_args),
            "has_pretrain": bool(self.train_cmd),
            "log_path": str(self.log_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "description": self.description,
        }


def _detect_legacy_model_version(folder_dir: Path, fallback: str) -> str:
    cfg_path = folder_dir / "src" / "config.py"
    if not cfg_path.is_file():
        return fallback
    try:
        text = cfg_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return fallback
    match = _MODEL_VERSION_RE.search(text)
    return match.group(1).strip() if match else fallback


def _detect_legacy_train_cmd(folder: str, folder_dir: Path) -> tuple[str, ...] | None:
    if folder == "etapa_2_version_especial" and (folder_dir / "src" / "builder.py").is_file():
        return (sys.executable, str(folder_dir / "src" / "builder.py"), "--once", "--rebuild")
    if folder == "etapa_1_version_especial" and (folder_dir / "src" / "train.py").is_file():
        return (sys.executable, str(folder_dir / "src" / "train.py"))
    return None


def discover_models() -> dict[str, ModelSpec]:
    """Discover encapsulated models (with model_manifest.json) and legacy stage folders."""
    registry: dict[str, ModelSpec] = {}

    for child in sorted(REPO_ROOT.iterdir(), key=lambda p: p.name.lower()):
        if not child.is_dir():
            continue
        manifest_path = child / "model_manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = read_json_file(manifest_path)
        if not isinstance(manifest, dict):
            manifest = {}
        entry_rel = str(manifest.get("entrypoint") or "serve.py")
        entrypoint = child / entry_rel
        hb_rel = manifest.get("heartbeat_file")
        log_rel = str(manifest.get("log_file") or "logs/live.log")
        self_check_raw = manifest.get("self_check_args") or []
        train_raw = manifest.get("train_args") or []
        train_cmd = tuple([sys.executable, str(entrypoint), *(str(a) for a in train_raw)]) if train_raw else None

        registry[child.name] = ModelSpec(
            folder=child.name,
            name=str(manifest.get("name") or child.name),
            model_version=str(manifest.get("model_version") or child.name),
            kind="encapsulated",
            root_dir=child,
            entrypoint=entrypoint,
            self_check_args=tuple(str(a) for a in self_check_raw),
            train_cmd=train_cmd,
            heartbeat_path=(child / str(hb_rel)) if hb_rel else None,
            log_path=child / log_rel,
            description=str(manifest.get("description") or ""),
        )

    for folder, default_name, default_mv in LEGACY_STAGE_ORDER:
        if folder in registry:
            continue
        folder_dir = REPO_ROOT / folder
        entrypoint = folder_dir / "src" / "serve.py"
        if not entrypoint.is_file():
            continue
        hb_candidate = folder_dir / "cache" / "heartbeat.json"
        registry[folder] = ModelSpec(
            folder=folder,
            name=default_name,
            model_version=_detect_legacy_model_version(folder_dir, default_mv),
            kind="legacy",
            root_dir=folder_dir,
            entrypoint=entrypoint,
            self_check_args=(),
            train_cmd=_detect_legacy_train_cmd(folder, folder_dir),
            heartbeat_path=hb_candidate if hb_candidate.parent.exists() else None,
            log_path=folder_dir / "logs" / "live.log",
            description=f"Carpeta historica ({folder}/src/serve.py)",
        )

    return registry


def resolve_model(registry: dict[str, ModelSpec], query: str) -> ModelSpec:
    """Resolve user-provided folder, name, or unique model_version."""
    norm = query.strip().lower()
    for folder, spec in registry.items():
        if folder.lower() == norm:
            return spec

    canon = re.sub(r"[\s\-_]+", "", norm)
    name_matches = [
        spec
        for spec in registry.values()
        if re.sub(r"[\s\-_]+", "", spec.name.lower()) == canon
        or re.sub(r"[\s\-_]+", "", spec.folder.lower()) == canon
    ]
    if len(name_matches) == 1:
        return name_matches[0]

    mv_matches = [spec for spec in registry.values() if spec.model_version.lower() == norm]
    if len(mv_matches) == 1:
        return mv_matches[0]
    if len(mv_matches) > 1:
        folders = ", ".join(s.folder for s in mv_matches)
        raise SystemExit(
            f"Hi ha multiples carpetes amb model_version='{query}': {folders}. "
            f"Especifica directament el nom de la carpeta."
        )

    available = ", ".join(registry.keys())
    raise SystemExit(f"Model '{query}' no trobat. Disponibles: {available}")
