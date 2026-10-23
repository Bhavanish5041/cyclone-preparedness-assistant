"""Unit tests for data inventory, inspection, hash verification, and manifest export."""

import hashlib
import json
from pathlib import Path
import zipfile

import pandas as pd
import pytest

from src.data.inventory import (
    EMPTY_FILE_SHA256,
    DataInventory,
    compute_sha256,
    detect_duplicates,
    inspect_csv,
    inspect_excel,
    inspect_file,
    inspect_zip,
    verify_shapefile_sidecars,
)


@pytest.fixture
def temp_workspace(tmp_path: Path):
    """Create a temporary test workspace with mock data files."""
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    return tmp_path, raw_dir


def test_sha256_known_content(tmp_path: Path):
    """Test SHA-256 computation against known string hash."""
    test_file = tmp_path / "hello.txt"
    test_file.write_text("hello world\n", encoding="utf-8")

    expected_hash = hashlib.sha256(b"hello world\n").hexdigest()
    computed_hash = compute_sha256(test_file)

    assert computed_hash == expected_hash
    assert len(computed_hash) == 64


def test_sha256_empty_file(tmp_path: Path):
    """Test SHA-256 computation on an empty file."""
    empty_file = tmp_path / "empty.dat"
    empty_file.touch()

    computed_hash = compute_sha256(empty_file)
    assert computed_hash == EMPTY_FILE_SHA256


def test_sha256_missing_file(tmp_path: Path):
    """Test that missing file raises FileNotFoundError."""
    missing = tmp_path / "nonexistent.csv"
    with pytest.raises(FileNotFoundError):
        compute_sha256(missing)


def test_duplicate_file_detection(tmp_path: Path):
    """Test detection of duplicate files using SHA-256 hashing."""
    f1 = tmp_path / "file1.csv"
    f2 = tmp_path / "file2.csv"
    f3 = tmp_path / "file3.csv"

    content_a = "facility_id,name,lat,lon\n1,Centre A,12.9,74.8\n"
    content_b = "facility_id,name,lat,lon\n2,Centre B,13.1,75.0\n"

    f1.write_text(content_a, encoding="utf-8")
    f2.write_text(content_a, encoding="utf-8")  # Duplicate of f1
    f3.write_text(content_b, encoding="utf-8")  # Distinct

    records = [
        {"relative_path": "file1.csv", "sha256": compute_sha256(f1)},
        {"relative_path": "file2.csv", "sha256": compute_sha256(f2)},
        {"relative_path": "file3.csv", "sha256": compute_sha256(f3)},
    ]

    duplicates = detect_duplicates(records)
    assert len(duplicates) == 1
    sha_dup = compute_sha256(f1)
    assert sha_dup in duplicates
    assert set(duplicates[sha_dup]) == {"file1.csv", "file2.csv"}


def test_empty_file_handling(tmp_path: Path):
    """Test that an empty 0-byte file is gracefully inspected and marked empty."""
    empty_pdf = tmp_path / "REPORT-EMPTY.pdf"
    empty_pdf.touch()

    rec = inspect_file(empty_pdf, project_root=tmp_path)
    assert rec["is_empty"] is True
    assert rec["size_bytes"] == 0
    assert rec["sha256"] == EMPTY_FILE_SHA256
    assert rec["inspection_status"] == "empty_or_invalid"
    assert "empty/corrupted" in rec["error"]


def test_csv_inspection_utf8(tmp_path: Path):
    """Test CSV inspection on clean UTF-8 data."""
    csv_file = tmp_path / "test_health.csv"
    csv_file.write_text(
        "id,facility_name,district,beds\n"
        "101,Mangalore PHC,Dakshina Kannada,10\n"
        "102,Bantwal CHC,Dakshina Kannada,30\n",
        encoding="utf-8",
    )

    info = inspect_csv(csv_file)
    assert info["status"] == "valid"
    assert info["encoding"] == "utf-8"
    assert info["row_count"] == 2
    assert info["column_count"] == 4
    assert info["columns"] == ["id", "facility_name", "district", "beds"]


def test_csv_inspection_latin1_fallback(tmp_path: Path):
    """Test CSV inspection gracefully falling back to Latin-1 on byte 0x96 (en-dash)."""
    csv_file = tmp_path / "test_latin1.csv"
    # Byte 0x96 in Windows-1252 / Latin-1 represents en-dash
    data = b"id,hospital_name,services\n1,Specialty Clinic \x96 Mangalore,Ambulance\n"
    csv_file.write_bytes(data)

    info = inspect_csv(csv_file)
    assert info["status"] == "valid"
    assert info["encoding"] in {"latin1", "cp1252", "iso-8859-1"}
    assert info["row_count"] == 1
    assert "hospital_name" in info["columns"]


def test_excel_inspection(tmp_path: Path):
    """Test Excel inspection with multiple worksheets."""
    xlsx_file = tmp_path / "test_census.xlsx"
    with pd.ExcelWriter(xlsx_file, engine="openpyxl") as writer:
        df1 = pd.DataFrame({"village_id": [1, 2], "population": [1000, 2000]})
        df2 = pd.DataFrame({"town_id": [10], "households": [500]})
        df1.to_excel(writer, sheet_name="Villages", index=False)
        df2.to_excel(writer, sheet_name="Towns", index=False)

    info = inspect_excel(xlsx_file)
    assert info["status"] == "valid"
    assert info["sheet_count"] == 2
    assert set(info["sheet_names"]) == {"Villages", "Towns"}
    assert "village_id" in info["sheets"]["Villages"]["columns"]
    assert "town_id" in info["sheets"]["Towns"]["columns"]


def test_zip_inspection_without_extraction(tmp_path: Path):
    """Test ZIP inspection without extracting members to disk."""
    zip_path = tmp_path / "test_archive.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("table1.csv", "a,b,c\n1,2,3\n")
        zf.writestr("subfolder/table2.csv", "x,y,z\n4,5,6\n")

    info = inspect_zip(zip_path)
    assert info["status"] == "valid"
    assert info["member_count"] == 2
    member_names = [m["filename"] for m in info["members"]]
    assert "table1.csv" in member_names
    assert "subfolder/table2.csv" in member_names
    assert info["total_uncompressed_bytes"] > 0
    # Ensure members were NOT extracted to disk
    assert not (tmp_path / "table1.csv").exists()


def test_corrupt_zip_handling(tmp_path: Path):
    """Test graceful handling of a malformed or corrupted ZIP file."""
    corrupt_zip = tmp_path / "corrupt.zip"
    corrupt_zip.write_bytes(b"PK\x03\x04this is not a valid zip archive contents")

    info = inspect_zip(corrupt_zip)
    assert info["status"] in {"corrupt", "error"}
    assert "Bad ZIP file" in info["error"] or "error" in info["error"].lower()


def test_shapefile_sidecar_verification(tmp_path: Path):
    """Test verification of shapefile sidecar files."""
    shp_base = tmp_path / "test_boundary"
    shp_file = tmp_path / "test_boundary.shp"
    shp_file.touch()

    # Missing shx, dbf, prj
    present, missing = verify_shapefile_sidecars(shp_file)
    assert ".shp" in present
    assert {".shx", ".dbf", ".prj"}.issubset(missing)

    # Create remaining sidecars
    (tmp_path / "test_boundary.shx").touch()
    (tmp_path / "test_boundary.dbf").touch()
    (tmp_path / "test_boundary.prj").touch()

    present_all, missing_none = verify_shapefile_sidecars(shp_file)
    assert {".shp", ".shx", ".dbf", ".prj"}.issubset(present_all)
    assert len(missing_none) == 0


def test_full_inventory_and_manifest_generation(temp_workspace):
    """Test end-to-end DataInventory scan and manifest/report export on mock workspace."""
    project_root, raw_dir = temp_workspace

    # Create mock datasets
    (raw_dir / "valid_facility.csv").write_text("id,name\n1,Facility 1\n", encoding="utf-8")
    (raw_dir / "valid_duplicate.csv").write_text("id,name\n1,Facility 1\n", encoding="utf-8")
    (raw_dir / "empty_report.pdf").touch()

    inventory = DataInventory(raw_dir=raw_dir, project_root=project_root)
    records = inventory.scan()

    assert len(records) == 3
    assert len(inventory.duplicates) == 1

    reports_dir = project_root / "reports"
    json_path = inventory.export_manifest_json(reports_dir / "manifest.json")
    csv_path = inventory.export_manifest_csv(reports_dir / "manifest.csv")
    md_path = inventory.generate_markdown_report(reports_dir / "inventory.md")

    assert json_path.exists()
    assert csv_path.exists()
    assert md_path.exists()

    # Verify JSON structure
    with json_path.open() as f:
        manifest_data = json.load(f)
    assert "metadata" in manifest_data
    assert "datasets" in manifest_data
    assert len(manifest_data["datasets"]) == 3

    # Verify CSV structure
    df_manifest = pd.read_csv(csv_path)
    assert "relative_path" in df_manifest.columns
    assert "sha256" in df_manifest.columns
    assert "is_duplicate" in df_manifest.columns
    assert "suitable_for_analysis" in df_manifest.columns
    assert len(df_manifest) == 3

    # Verify Markdown report
    md_content = md_path.read_text(encoding="utf-8")
    assert "Executive Summary" in md_content
    assert "Asset Manifest" in md_content
    assert "Exact Duplicate Files Detected" in md_content
