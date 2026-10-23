#!/usr/bin/env python3
"""Run data inventory, file inspection, manifest export, and markdown report generation.

Usage:
    python scripts/inspect_data.py
    python scripts/inspect_data.py --raw-dir data/raw --output-dir reports
"""

import argparse
import logging
from pathlib import Path
import sys

# Ensure src/ is accessible regardless of invocation directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.inventory import DataInventory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run reproducible data inventory and quality assessment for raw datasets."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw",
        help="Path to the raw data directory (default: data/raw)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "reports",
        help="Directory to store manifests and markdown reports (default: reports)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("=" * 80)
    print("CYCLONE PREPAREDNESS ASSISTANT — RAW DATA INVENTORY PIPELINE")
    print("=" * 80)
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Raw data dir : {args.raw_dir}")
    print(f"Output dir   : {args.output_dir}")
    print("-" * 80)

    try:
        inventory = DataInventory(raw_dir=args.raw_dir, project_root=PROJECT_ROOT)
        print("Scanning raw datasets recursively...")
        records = inventory.scan()
        print(f"Discovered and inspected {len(records)} files.\n")

        # Export manifests
        args.output_dir.mkdir(parents=True, exist_ok=True)
        json_path = inventory.export_manifest_json(args.output_dir / "data_manifest.json")
        csv_path = inventory.export_manifest_csv(args.output_dir / "data_manifest.csv")
        md_path = inventory.generate_markdown_report(args.output_dir / "data_inventory.md")

        print("Manifests and Reports Generated:")
        print(f"  • JSON Manifest : {json_path.relative_to(PROJECT_ROOT)}")
        print(f"  • CSV Manifest  : {csv_path.relative_to(PROJECT_ROOT)}")
        print(f"  • Markdown Report: {md_path.relative_to(PROJECT_ROOT)}\n")

        # Summary terminal output
        total_size_mb = sum(r["size_mb"] for r in records)
        print("-" * 80)
        print(f"Total Files       : {len(records)}")
        print(f"Total Size        : {total_size_mb:.2f} MB (~{total_size_mb / 1024:.2f} GB)")
        print(f"Duplicate Groups  : {len(inventory.duplicates)}")
        if inventory.duplicates:
            for sha, paths in inventory.duplicates.items():
                print(f"  • Duplicate pair ({sha[:12]}...):")
                for p in paths:
                    print(f"      - {p}")

        empty_files = [r for r in records if r["is_empty"]]
        print(f"Empty/Corrupt     : {len(empty_files)}")
        for ef in empty_files:
            print(f"  • {ef['relative_path']} (0 bytes)")

        print("=" * 80)
        print("Inventory pipeline completed successfully.")
        print("=" * 80)
        return 0

    except Exception as exc:
        logging.exception(f"Fatal error during inventory pipeline execution: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
