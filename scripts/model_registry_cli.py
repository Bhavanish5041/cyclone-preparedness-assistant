#!/usr/bin/env python3
"""CLI utility to inspect, validate, and manage models in the local CYCLONE INTEL Model Registry.

Commands:
    list                     List all registered models and active versions
    show <model_id>          Display detailed metadata and schema for a model
    validate [model_id]      Verify presence and SHA256 checksums of model artifacts
    promote <model_id> <v>   Promote a model version to production/validated/candidate
    rollback <model_id> <v>  Roll back active production model to a previous version
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.registry.local_registry import LocalModelRegistry


def main():
    parser = argparse.ArgumentParser(
        description="CYCLONE INTEL Local Model Registry Management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. list
    subparsers.add_parser("list", help="List all registered models and their active versions")

    # 2. show
    show_parser = subparsers.add_parser("show", help="Show full metadata for a specific model")
    show_parser.add_argument("model_id", type=str, help="Model identifier")
    show_parser.add_argument("--version", "-v", type=str, default=None, help="Specific version (default: active)")

    # 3. validate
    val_parser = subparsers.add_parser("validate", help="Validate artifact files and cryptographic SHA256 checksums")
    val_parser.add_argument("model_id", type=str, nargs="?", default=None, help="Optional model identifier")

    # 4. promote
    prom_parser = subparsers.add_parser("promote", help="Promote a model version to a target lifecycle state")
    prom_parser.add_argument("model_id", type=str, help="Model identifier")
    prom_parser.add_argument("version", type=str, help="Version string to promote")
    prom_parser.add_argument("--status", "-s", type=str, default="production", choices=["candidate", "validated", "production", "retired"])
    prom_parser.add_argument("--note", "-n", type=str, default="", help="Audit note explaining promotion reason")
    prom_parser.add_argument("--author", "-a", type=str, default="cli_operator", help="Operator author name")

    # 5. rollback
    roll_parser = subparsers.add_parser("rollback", help="Roll back active production version to a previous version")
    roll_parser.add_argument("model_id", type=str, help="Model identifier")
    roll_parser.add_argument("version", type=str, help="Target version to restore to production")
    roll_parser.add_argument("--note", "-n", type=str, default="Emergency operational rollback", help="Audit note")
    roll_parser.add_argument("--author", "-a", type=str, default="cli_operator", help="Operator author name")

    args = parser.parse_args()
    registry = LocalModelRegistry()

    if args.command == "list":
        models = registry.list_models()
        print("\n=== CYCLONE INTEL LOCAL MODEL REGISTRY ===")
        print(f"{'Model Identifier':<35} {'Family':<20} {'Active Ver':<12} {'Status':<12} {'Total Ver':<10}")
        print("-" * 92)
        for m in models:
            print(f"{m['model_identifier']:<35} {m['model_family']:<20} {m['active_version']:<12} {m['status']:<12} {m['total_versions']:<10}")
        print("-" * 92)
        print(f"Total Models: {len(models)}\n")

    elif args.command == "show":
        try:
            details = registry.get_model(args.model_id, version=args.version)
            print(f"\n=== METADATA: {args.model_id} (Version: {details.get('version')}) ===")
            print(json.dumps(details, indent=2))
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "validate":
        print("\n=== ARTIFACT VALIDATION REPORT ===")
        results = registry.validate_artifacts(model_id=args.model_id)
        all_valid = True
        for r in results:
            status = r["status"]
            icon = "✅" if status == "VALID" else "❌"
            m_id = r["model_identifier"]
            if "artifact_path" in r:
                target = r["artifact_path"]
            elif "tag" in r:
                target = f"Ollama tag: {r['tag']} (Digest: {r.get('digest')})"
            else:
                target = "Unknown artifact"

            print(f" {icon} [{status}] {m_id} -> {target}")
            if status != "VALID":
                all_valid = False
                print(f"    Details: {r.get('error') or r.get('actual_sha256')}")

        print("=" * 60)
        if all_valid:
            print("Status: ALL ARTIFACTS VERIFIED AND VALID ✅\n")
            sys.exit(0)
        else:
            print("Status: ARTIFACT VALIDATION FAILED ❌\n", file=sys.stderr)
            sys.exit(1)

    elif args.command == "promote":
        try:
            entry = registry.promote_model(
                model_id=args.model_id,
                version=args.version,
                target_status=args.status,
                author=args.author,
                note=args.note,
            )
            print(f"Successfully promoted '{args.model_id}' to status '{args.status}'.")
            print(json.dumps(entry, indent=2))
        except Exception as e:
            print(f"Error promoting model: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "rollback":
        try:
            entry = registry.rollback_model(
                model_id=args.model_id,
                target_version=args.version,
                author=args.author,
                note=args.note,
            )
            print(f"Successfully rolled back '{args.model_id}' to version '{args.version}' (production).")
            print(json.dumps(entry, indent=2))
        except Exception as e:
            print(f"Error rolling back model: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
