"""Lightweight, local, Git-compatible Model Registry for CYCLONE INTEL.

Tracks:
- Model identifier and version
- Model family and purpose
- Artifact paths or Ollama model tags
- Cryptographic SHA256 digests
- Input and output schemas
- Training dataset and source provenance
- Quantitative evaluation metrics
- Runtime requirements (CPU/RAM)
- Model status: candidate, validated, production, retired
- Promotion and rollback audit history
"""

from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("local_model_registry")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_REGISTRY_PATH = PROJECT_ROOT / "models" / "model_registry.json"

VALID_STATUSES = {"candidate", "validated", "production", "retired"}


def compute_file_sha256(file_path: Path) -> str:
    """Compute standard SHA256 hex digest for a file on disk."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class LocalModelRegistry:
    """Manages the local JSON-backed model registry manifest."""

    def __init__(self, registry_path: Optional[Path] = None):
        self.registry_path = registry_path or DEFAULT_REGISTRY_PATH
        if not self.registry_path.exists():
            raise FileNotFoundError(f"Model registry manifest not found at {self.registry_path}")
        self._load()

    def _load(self) -> None:
        """Load manifest from disk."""
        with open(self.registry_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def save(self) -> None:
        """Save manifest back to disk."""
        self.data["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)
        logger.info(f"Model registry manifest saved to {self.registry_path}")

    def list_models(self) -> List[Dict[str, Any]]:
        """Return a summary of all registered models."""
        summary = []
        for model_id, model_info in self.data.get("models", {}).items():
            active_v = model_info.get("active_version")
            v_data = model_info.get("versions", {}).get(active_v, {})
            summary.append({
                "model_identifier": model_id,
                "model_family": model_info.get("model_family"),
                "active_version": active_v,
                "status": v_data.get("status"),
                "total_versions": len(model_info.get("versions", {})),
                "purpose": model_info.get("purpose"),
            })
        return summary

    def get_model(self, model_id: str, version: Optional[str] = None) -> Dict[str, Any]:
        """Get model details for a specific or active version."""
        models = self.data.get("models", {})
        if model_id not in models:
            raise KeyError(f"Model '{model_id}' not found in registry. Available: {list(models.keys())}")

        m = models[model_id]
        target_v = version or m.get("active_version")
        versions = m.get("versions", {})
        if target_v not in versions:
            raise KeyError(f"Version '{target_v}' not found for model '{model_id}'. Available: {list(versions.keys())}")

        return versions[target_v]

    def validate_artifacts(self, model_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Validate artifact presence and cryptographic SHA256 checksums."""
        target_models = [model_id] if model_id else list(self.data.get("models", {}).keys())
        results = []

        for m_id in target_models:
            m_info = self.data["models"].get(m_id)
            if not m_info:
                continue
            active_v = m_info.get("active_version")
            v_info = m_info.get("versions", {}).get(active_v, {})

            artifact_files = v_info.get("artifact_files", [])
            for af in artifact_files:
                # If artifact is an Ollama tag, verify it's not a local file
                if "model_tag" in af:
                    results.append({
                        "model_identifier": m_id,
                        "version": active_v,
                        "artifact_type": "ollama_model",
                        "tag": af.get("model_tag"),
                        "digest": af.get("digest"),
                        "status": "VALID",
                        "note": "Managed by local Ollama runtime",
                    })
                    continue

                rel_path = af.get("path")
                expected_sha = af.get("sha256")
                full_path = PROJECT_ROOT / rel_path

                if not full_path.exists():
                    results.append({
                        "model_identifier": m_id,
                        "version": active_v,
                        "artifact_path": rel_path,
                        "status": "MISSING",
                        "error": f"File does not exist at {full_path}",
                    })
                    continue

                actual_sha = compute_file_sha256(full_path)
                if expected_sha and actual_sha != expected_sha:
                    results.append({
                        "model_identifier": m_id,
                        "version": active_v,
                        "artifact_path": rel_path,
                        "status": "CHECKSUM_MISMATCH",
                        "expected_sha256": expected_sha,
                        "actual_sha256": actual_sha,
                    })
                else:
                    results.append({
                        "model_identifier": m_id,
                        "version": active_v,
                        "artifact_path": rel_path,
                        "status": "VALID",
                        "sha256": actual_sha,
                        "size_bytes": full_path.stat().st_size,
                    })

        return results

    def promote_model(
        self,
        model_id: str,
        version: str,
        target_status: str = "production",
        author: str = "system",
        note: str = "",
    ) -> Dict[str, Any]:
        """Promote a model version to a new status (e.g., candidate -> production)."""
        if target_status not in VALID_STATUSES:
            raise ValueError(f"Invalid status '{target_status}'. Must be one of {VALID_STATUSES}")

        models = self.data.get("models", {})
        if model_id not in models:
            raise KeyError(f"Model '{model_id}' not registered.")

        m = models[model_id]
        if version not in m.get("versions", {}):
            raise KeyError(f"Version '{version}' not found for model '{model_id}'.")

        v_data = m["versions"][version]
        old_status = v_data.get("status")

        # If promoting to production, update active_version and demote any existing production versions
        if target_status == "production":
            for v_key, v_val in m["versions"].items():
                if v_val.get("status") == "production" and v_key != version:
                    v_val["status"] = "validated"
            m["active_version"] = version

        v_data["status"] = target_status
        history_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": "promoted",
            "from_status": old_status,
            "status": target_status,
            "author": author,
            "note": note,
        }
        v_data.setdefault("history", []).append(history_entry)

        self.save()
        logger.info(f"Model '{model_id}' version '{version}' promoted from '{old_status}' to '{target_status}'.")
        return history_entry

    def rollback_model(
        self,
        model_id: str,
        target_version: str,
        author: str = "system",
        note: str = "",
    ) -> Dict[str, Any]:
        """Roll back active production version to a previous validated version."""
        return self.promote_model(
            model_id=model_id,
            version=target_version,
            target_status="production",
            author=author,
            note=f"Rollback to {target_version}. {note}".strip(),
        )
