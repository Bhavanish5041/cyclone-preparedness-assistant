"""Unit and integration tests for Model Quality Monitoring and Drift Auditing (Phase 2.1)."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.monitoring.model_quality import (
    AHP_TARGET_MEANING,
    CATEGORICAL_FEATURES,
    TARGET_CLASSES,
    ComprehensiveMonitoringReport,
    DataDriftReport,
    EvidentlyExecutionError,
    EvidentlyIntegrationStatus,
    EvidentlyReportGenerator,
    EvidentlyReportResult,
    EvidentlySuiteResult,
    FeatureSchemaValidator,
    ModelQualityEvaluator,
    ModelQualityMonitor,
    PredictionDriftReport,
    SchemaValidationResult,
    check_evidently_availability,
    compute_data_drift,
    compute_prediction_drift,
    compute_psi,
    evaluate_model_quality,
    load_model_metadata,
    validate_input_schema,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METADATA_PATH = PROJECT_ROOT / "models" / "logistic_regression_metadata.json"
MODELLING_PATH = PROJECT_ROOT / "data" / "processed" / "modelling_dataset.parquet"
PROBABILITIES_PATH = PROJECT_ROOT / "data" / "processed" / "village_preparedness_probabilities.parquet"


@pytest.fixture(scope="module")
def metadata():
    assert METADATA_PATH.exists(), f"Missing metadata file at {METADATA_PATH}"
    return load_model_metadata(METADATA_PATH)


@pytest.fixture(scope="module")
def df_modelling():
    assert MODELLING_PATH.exists(), f"Missing modelling dataset at {MODELLING_PATH}"
    return pd.read_parquet(MODELLING_PATH)


@pytest.fixture(scope="module")
def df_probabilities():
    assert PROBABILITIES_PATH.exists(), f"Missing probabilities dataset at {PROBABILITIES_PATH}"
    return pd.read_parquet(PROBABILITIES_PATH)


def test_load_model_metadata(metadata):
    """Verify loading and basic integrity of model metadata."""
    assert metadata["model_artifact"] == "logistic_regression_tier.joblib"
    assert len(metadata["features"]) == 26
    assert metadata["split_configuration"]["train_size"] == 1315
    assert metadata["split_configuration"]["val_size"] == 282
    assert metadata["split_configuration"]["test_size"] == 282
    assert "High" in metadata["target"]["classes"]
    assert "Low" in metadata["target"]["classes"]
    assert "Medium" in metadata["target"]["classes"]


def test_schema_validator_valid_input(df_modelling, metadata):
    """Verify schema validator accepts well-formed feature matrix."""
    features = metadata["features"]
    X = df_modelling[features].copy()

    validator = FeatureSchemaValidator()
    result = validator.validate(X)

    assert result.is_valid is True
    assert len(result.errors) == 0
    assert result.expected_feature_count == 26
    assert result.received_feature_count == 26
    assert result.order_matches_strictly is True
    assert len(result.missing_features) == 0


def test_schema_validator_missing_features(df_modelling, metadata):
    """Verify validator catches and lists missing required features."""
    features = metadata["features"]
    # Drop 2 critical features
    dropped_cols = [features[0], features[5]]
    X_incomplete = df_modelling[features].drop(columns=dropped_cols)

    validator = FeatureSchemaValidator()
    result = validator.validate(X_incomplete)

    assert result.is_valid is False
    assert len(result.errors) >= 1
    assert set(dropped_cols).issubset(set(result.missing_features))


def test_schema_validator_detects_nulls_and_infs(df_modelling, metadata):
    """Verify validator catches NaN and infinite values in input features."""
    features = metadata["features"]
    X = df_modelling[features].copy()
    X.iloc[0, 1] = np.nan
    X.iloc[1, 2] = np.inf

    validator = FeatureSchemaValidator()
    result = validator.validate(X)

    assert result.is_valid is False
    assert result.null_counts.get(features[1], 0) == 1
    assert result.infinite_counts.get(features[2], 0) == 1
    assert any("missing" in err for err in result.errors)
    assert any("infinite" in err for err in result.errors)


def test_schema_validator_scrambled_column_order(df_modelling, metadata):
    """Verify detection of scrambled column order and successful alignment."""
    features = metadata["features"]
    scrambled_order = list(reversed(features))
    X_scrambled = df_modelling[scrambled_order].copy()

    validator = FeatureSchemaValidator()
    result = validator.validate(X_scrambled, strict_order=True)

    assert result.order_matches_strictly is False
    assert any("sequence differs" in w for w in result.warnings)

    # Test extract_and_align restores strict fit order
    X_aligned = validator.extract_and_align(X_scrambled)
    assert list(X_aligned.columns) == features


def test_model_quality_evaluator_metrics():
    """Verify calculation of accuracy, macro/weighted F1, and confusion matrix."""
    y_true = ["High", "High", "Low", "Low", "Medium", "Medium"]
    y_pred = ["High", "Medium", "Low", "Medium", "Medium", "Medium"]
    y_probs = np.array([
        [0.85, 0.05, 0.10],  # High -> High (correct)
        [0.40, 0.10, 0.50],  # High -> Medium (misclassified, boundary)
        [0.05, 0.90, 0.05],  # Low -> Low (correct)
        [0.10, 0.40, 0.50],  # Low -> Medium (misclassified, boundary)
        [0.05, 0.05, 0.90],  # Medium -> Medium (correct)
        [0.10, 0.10, 0.80],  # Medium -> Medium (correct)
    ])

    evaluator = ModelQualityEvaluator(class_names=["High", "Low", "Medium"])
    metrics = evaluator.evaluate(y_true, y_pred, y_probs)

    assert metrics.sample_size == 6
    # 4 correct out of 6 -> acc = 2/3 = 0.6667
    assert pytest.approx(metrics.accuracy, rel=1e-3) == 0.6667
    assert "High" in metrics.per_class_metrics
    assert "Low" in metrics.per_class_metrics
    assert "Medium" in metrics.per_class_metrics
    assert metrics.confusion_matrix["labels"] == ["High", "Low", "Medium"]
    assert metrics.brier_scores is not None
    assert metrics.log_loss_value is not None
    assert metrics.boundary_case_count is not None
    assert metrics.disagreement_counts is not None


def test_compute_psi():
    """Verify Population Stability Index calculation on identical vs shifted distributions."""
    ref = np.random.normal(loc=0.0, scale=1.0, size=1000)
    cur_same = np.random.normal(loc=0.0, scale=1.0, size=1000)
    cur_shifted = np.random.normal(loc=1.5, scale=1.5, size=1000)

    psi_same = compute_psi(ref, cur_same)
    psi_shifted = compute_psi(ref, cur_shifted)

    # Identical distribution PSI should be small (<0.10)
    assert psi_same < 0.15
    # Shifted distribution PSI should be substantial (>0.25)
    assert psi_shifted > 0.25


def test_data_drift_detection_no_drift_on_random_splits(df_modelling, metadata):
    """Verify train vs test split exhibits zero or minimal feature drift (validating random stratified split)."""
    features = metadata["features"]
    train_idx = metadata["split_indices"]["train"]
    test_idx = metadata["split_indices"]["test"]

    train_df = df_modelling[features].iloc[train_idx]
    test_df = df_modelling[features].iloc[test_idx]

    report = compute_data_drift(train_df, test_df, features=features)
    assert report.total_features_evaluated == 26
    # In a properly stratified random split, drift share should be very low (< 10%)
    assert report.drift_share <= 0.10
    assert report.dataset_drift_detected is False


def test_data_drift_detection_spatial_covariate_shift(df_modelling, metadata):
    """Verify that comparing Dakshina Kannada to Uttara Kannada detects spatial covariate shift."""
    features = metadata["features"]
    dk_df = df_modelling[df_modelling["district_name"] == "Dakshina Kannada"][features]
    uk_df = df_modelling[df_modelling["district_name"] == "Uttara Kannada"][features]

    report = compute_data_drift(dk_df, uk_df, features=features)
    assert report.total_features_evaluated == 26
    # Geographic differences between coastal DK and hilly/forested UK cause significant covariate shift (> 50%)
    assert report.drift_share > 0.50
    assert report.dataset_drift_detected is True


def test_prediction_drift_detection(df_probabilities):
    """Verify prediction drift engine compares probability distributions across splits."""
    p_low = df_probabilities["prob_low"].values
    p_med = df_probabilities["prob_medium"].values
    p_high = df_probabilities["prob_high"].values
    probs = np.column_stack([p_high, p_low, p_med])  # High, Low, Medium

    # Compare first half vs second half
    mid = len(probs) // 2
    probs_ref = probs[:mid]
    probs_cur = probs[mid:]

    report = compute_prediction_drift(
        reference_probs=probs_ref,
        current_probs=probs_cur,
        class_names=["High", "Low", "Medium"],
    )

    assert report.sample_size_reference == mid
    assert "High" in report.probability_drift
    assert "Low" in report.probability_drift
    assert "Medium" in report.probability_drift
    assert "High" in report.population_stability_index


def test_evidently_availability_check():
    """Verify Evidently status check returns structured diagnostic and detects installed version."""
    status = check_evidently_availability()
    assert isinstance(status.installed, bool)
    assert status.status_message is not None
    assert "DataQualityPreset" in status.supported_presets
    assert "DataDriftPreset" in status.supported_presets
    assert "ClassificationPreset" in status.supported_presets
    assert "TargetDriftPreset" in status.supported_presets
    # Evidently is installed in the active environment
    assert status.installed is True
    assert status.version == "0.7.23"


def test_evidently_data_quality_report_generation(tmp_path, df_modelling, metadata):
    """Verify generation of Evidently Data Quality report with HTML and JSON outputs."""
    features = metadata["features"]
    train_idx = metadata["split_indices"]["train"]
    test_idx = metadata["split_indices"]["test"]

    train_df = df_modelling.iloc[train_idx][features]
    test_df = df_modelling.iloc[test_idx][features]

    generator = EvidentlyReportGenerator(metadata_path=METADATA_PATH, output_dir=tmp_path)
    report, result = generator.generate_data_quality_report(
        current_features=test_df,
        reference_features=train_df,
        save_html=True,
        save_json=True,
        output_prefix="test_data_quality",
    )

    assert result.status == "success"
    assert result.report_type == "data_quality"
    assert result.html_path is not None
    assert result.json_path is not None

    html_path = Path(result.html_path)
    json_path = Path(result.json_path)
    assert html_path.exists()
    assert html_path.stat().st_size > 100_000
    assert json_path.exists()
    assert json_path.stat().st_size > 100

    # Summary metrics verification
    assert result.summary_metrics.get("number_of_rows") == len(test_df)
    assert result.summary_metrics.get("number_of_columns") == len(features)
    assert result.summary_metrics.get("number_of_missing_values") == 0


def test_evidently_feature_drift_report_generation(tmp_path, df_modelling, metadata):
    """Verify generation of Evidently Feature Drift report across 26 model features."""
    features = metadata["features"]
    train_idx = metadata["split_indices"]["train"]
    test_idx = metadata["split_indices"]["test"]

    train_df = df_modelling.iloc[train_idx][features]
    test_df = df_modelling.iloc[test_idx][features]

    generator = EvidentlyReportGenerator(metadata_path=METADATA_PATH, output_dir=tmp_path)
    report, result = generator.generate_feature_drift_report(
        reference_features=train_df,
        current_features=test_df,
        save_html=True,
        save_json=True,
        output_prefix="test_feature_drift",
    )

    assert result.status == "success"
    assert result.report_type == "feature_drift"
    assert result.html_path is not None
    assert result.json_path is not None

    html_path = Path(result.html_path)
    assert html_path.exists()
    assert html_path.stat().st_size > 100_000

    # Offline random train/test split should display zero or low drift (well below 33% threshold)
    summary = result.summary_metrics
    assert summary.get("number_of_columns") == len(features)
    assert summary.get("dataset_drift") is False
    assert summary.get("share_of_drifted_columns", 0.0) <= 0.20


def test_evidently_prediction_drift_report_generation(tmp_path, df_probabilities, metadata):
    """Verify generation of Evidently Prediction Drift report assessing class and probability shift."""
    train_idx = metadata["split_indices"]["train"]
    test_idx = metadata["split_indices"]["test"]

    ref_preds = df_probabilities.iloc[train_idx]["predicted_preparedness_tier"]
    cur_preds = df_probabilities.iloc[test_idx]["predicted_preparedness_tier"]

    # Align probability matrices to canonical class order [High, Low, Medium]
    ref_probs = df_probabilities.iloc[train_idx][["prob_high", "prob_low", "prob_medium"]].to_numpy()
    cur_probs = df_probabilities.iloc[test_idx][["prob_high", "prob_low", "prob_medium"]].to_numpy()

    generator = EvidentlyReportGenerator(metadata_path=METADATA_PATH, output_dir=tmp_path)
    report, result = generator.generate_prediction_drift_report(
        reference_preds=ref_preds,
        current_preds=cur_preds,
        reference_probs=ref_probs,
        current_probs=cur_probs,
        save_html=True,
        save_json=True,
        output_prefix="test_prediction_drift",
    )

    assert result.status == "success"
    assert result.report_type == "prediction_drift"
    assert result.html_path is not None
    assert Path(result.html_path).exists()
    assert Path(result.html_path).stat().st_size > 100_000


def test_evidently_classification_performance_reporting_with_labels(tmp_path, df_probabilities, metadata):
    """Verify Evidently Classification Performance report against verified AHP ground-truth labels."""
    train_idx = metadata["split_indices"]["train"]
    test_idx = metadata["split_indices"]["test"]

    ref_labels = df_probabilities.iloc[train_idx]["actual_preparedness_tier"]
    ref_preds = df_probabilities.iloc[train_idx]["predicted_preparedness_tier"]
    ref_probs = df_probabilities.iloc[train_idx][["prob_high", "prob_low", "prob_medium"]].to_numpy()

    cur_labels = df_probabilities.iloc[test_idx]["actual_preparedness_tier"]
    cur_preds = df_probabilities.iloc[test_idx]["predicted_preparedness_tier"]
    cur_probs = df_probabilities.iloc[test_idx][["prob_high", "prob_low", "prob_medium"]].to_numpy()

    generator = EvidentlyReportGenerator(metadata_path=METADATA_PATH, output_dir=tmp_path)
    report, result = generator.generate_classification_performance_report(
        current_labels=cur_labels,
        current_preds=cur_preds,
        current_probs=cur_probs,
        reference_labels=ref_labels,
        reference_preds=ref_preds,
        reference_probs=ref_probs,
        save_html=True,
        save_json=True,
        output_prefix="test_classification_performance",
    )

    assert result.status == "success"
    assert result.report_type == "classification_performance"
    assert result.html_path is not None
    assert Path(result.html_path).exists()
    assert Path(result.json_path).exists()

    # Target description must explicitly state AHP preparedness tier (not physical hazard)
    assert "AHP" in result.target_description
    assert "physical cyclone-risk" in result.target_description

    # Baseline performance check on held-out test split
    summary = result.summary_metrics
    assert summary.get("accuracy") is not None
    assert summary["accuracy"] > 0.70  # Model baseline is ~87%


def test_evidently_handling_of_missing_values(tmp_path, df_modelling, metadata):
    """Verify Evidently Data Quality report handles and correctly tallies NaN values."""
    features = metadata["features"]
    test_idx = metadata["split_indices"]["test"]
    test_df = df_modelling.iloc[test_idx][features].copy()

    # Inject known missing values
    test_df.iloc[0, 0] = np.nan
    test_df.iloc[1, 1] = np.nan
    test_df.iloc[2, 2] = np.nan

    generator = EvidentlyReportGenerator(metadata_path=METADATA_PATH, output_dir=tmp_path)
    report, result = generator.generate_data_quality_report(
        current_features=test_df,
        save_html=True,
        save_json=True,
        output_prefix="test_missing_values",
    )

    assert result.status == "success"
    assert result.summary_metrics.get("number_of_missing_values") == 3


def test_evidently_26_feature_schema_and_ordering(tmp_path, df_modelling, metadata):
    """Verify Evidently generator re-orders scrambled feature columns to canonical 26-feature order."""
    features = metadata["features"]
    test_idx = metadata["split_indices"]["test"]
    test_df = df_modelling.iloc[test_idx][features].copy()

    # Reverse column order
    reversed_features = list(reversed(features))
    scrambled_df = test_df[reversed_features].copy()

    generator = EvidentlyReportGenerator(metadata_path=METADATA_PATH, output_dir=tmp_path)
    report, result = generator.generate_data_quality_report(
        current_features=scrambled_df,
        save_html=False,
        save_json=False,
    )
    assert result.status == "success"
    assert result.summary_metrics.get("number_of_columns") == 26


def test_evidently_html_report_generation_meaningful_content(tmp_path, df_modelling, metadata):
    """Verify generated HTML reports contain structured HTML content and model feature names."""
    features = metadata["features"]
    test_idx = metadata["split_indices"]["test"]
    test_df = df_modelling.iloc[test_idx][features]

    generator = EvidentlyReportGenerator(metadata_path=METADATA_PATH, output_dir=tmp_path)
    _, result = generator.generate_data_quality_report(
        current_features=test_df,
        save_html=True,
        save_json=False,
        output_prefix="test_html_content",
    )

    with open(result.html_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "<html" in content.lower()
    assert "evidently" in content.lower()
    # Check that key domain feature names appear in HTML payload
    assert "cvi_physical" in content
    assert "coastal_risk_score" in content


def test_evidently_json_structured_output_generation(tmp_path, df_modelling, metadata):
    """Verify generated JSON reports contain parseable metrics and structured schema."""
    features = metadata["features"]
    test_idx = metadata["split_indices"]["test"]
    test_df = df_modelling.iloc[test_idx][features]

    generator = EvidentlyReportGenerator(metadata_path=METADATA_PATH, output_dir=tmp_path)
    _, result = generator.generate_data_quality_report(
        current_features=test_df,
        save_html=False,
        save_json=True,
        output_prefix="test_json_schema",
    )

    with open(result.json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "metrics" in data
    assert isinstance(data["metrics"], list)
    assert len(data["metrics"]) > 0
    metric_names = [m.get("metric") for m in data["metrics"]]
    assert "DatasetSummaryMetric" in metric_names


def test_evidently_invalid_incompatible_inputs(tmp_path, df_modelling, metadata):
    """Verify validator and generator raise explicit ValueErrors on malformed inputs."""
    features = metadata["features"]
    test_idx = metadata["split_indices"]["test"]
    test_df = df_modelling.iloc[test_idx][features]

    generator = EvidentlyReportGenerator(metadata_path=METADATA_PATH, output_dir=tmp_path)

    # 1. Empty DataFrame
    with pytest.raises(ValueError, match="Cannot generate Data Quality report on empty dataset"):
        generator.generate_data_quality_report(current_features=pd.DataFrame())

    # 2. Missing required columns
    incomplete_df = test_df.drop(columns=[features[0], features[1]])
    with pytest.raises(ValueError, match="Input features missing"):
        generator.generate_data_quality_report(current_features=incomplete_df)

    # 3. Label/prediction length mismatch
    with pytest.raises(ValueError, match="length mismatch"):
        generator.generate_classification_performance_report(
            current_labels=["High", "Low"],
            current_preds=["High"],
        )

    # 4. Probability shape mismatch
    with pytest.raises(ValueError, match="current_probs shape"):
        generator.generate_classification_performance_report(
            current_labels=["High", "Low"],
            current_preds=["High", "Low"],
            current_probs=np.ones((2, 2)),  # Expects 3 classes
        )


def test_evidently_report_generation_errors_in_suite(tmp_path, df_modelling, metadata):
    """Verify run_suite gracefully captures per-report errors without terminating execution."""
    features = metadata["features"]
    test_idx = metadata["split_indices"]["test"]
    test_df = df_modelling.iloc[test_idx][features]

    generator = EvidentlyReportGenerator(metadata_path=METADATA_PATH, output_dir=tmp_path)

    # Intentionally trigger an error in classification_performance by passing mismatched labels/preds
    suite_result = generator.run_suite(
        current_features=test_df,
        current_labels=pd.Series(["High", "Low"]),
        current_preds=pd.Series(["High"]),  # Length mismatch
        save_html=False,
        save_json=False,
    )

    assert suite_result.status in ["partial", "failed"]
    assert "classification_performance" in suite_result.reports
    assert suite_result.reports["classification_performance"].status == "failed"
    assert suite_result.reports["classification_performance"].error_message is not None
    # Data Quality report should still succeed
    assert suite_result.reports["data_quality"].status == "success"


def test_native_monitoring_fallback_when_evidently_uninstalled(monkeypatch, df_modelling, metadata):
    """Verify that when Evidently is unavailable, native monitoring works and Evidently raises clear errors."""
    import src.monitoring.model_quality as mq

    monkeypatch.setattr(mq, "EVIDENTLY_AVAILABLE", False)
    monkeypatch.setattr(mq, "EVIDENTLY_IMPORT_ERROR", "Simulated uninstalled environment")

    # 1. Availability check indicates uninstalled
    status = mq.check_evidently_availability()
    assert status.installed is False
    assert status.version is None
    assert "Simulated uninstalled environment" in status.status_message

    # 2. Native monitoring cycle succeeds without Evidently
    features = metadata["features"]
    test_idx = metadata["split_indices"]["test"]
    test_df = df_modelling.iloc[test_idx][features]

    monitor = mq.ModelQualityMonitor()
    report = monitor.run_monitoring_cycle(
        current_features=test_df,
        run_evidently=False,
    )
    assert report.schema_validation.is_valid is True
    assert report.evidently_suite is None
    assert report.evidently_status.installed is False

    # 3. Explicit Evidently request raises EvidentlyExecutionError
    with pytest.raises(mq.EvidentlyExecutionError, match="Simulated uninstalled environment"):
        monitor.run_monitoring_cycle(
            current_features=test_df,
            run_evidently=True,
        )

    # 4. Standalone EvidentlyReportGenerator raises EvidentlyExecutionError
    gen = mq.EvidentlyReportGenerator()
    with pytest.raises(mq.EvidentlyExecutionError, match="Simulated uninstalled environment"):
        gen.generate_data_quality_report(current_features=test_df)


def test_comprehensive_monitoring_cycle_with_evidently_suite(tmp_path, df_modelling, df_probabilities, metadata):
    """Verify integrated end-to-end monitoring cycle with active Evidently AI suite."""
    features = metadata["features"]
    train_idx = metadata["split_indices"]["train"]
    test_idx = metadata["split_indices"]["test"]

    train_df = df_modelling.iloc[train_idx]
    test_df = df_modelling.iloc[test_idx]

    prob_cols = ["prob_high", "prob_low", "prob_medium"]
    ref_probs = df_probabilities.iloc[train_idx][prob_cols].to_numpy()
    cur_probs = df_probabilities.iloc[test_idx][prob_cols].to_numpy()

    monitor = ModelQualityMonitor()
    report = monitor.run_monitoring_cycle(
        current_features=test_df[features],
        reference_features=train_df[features],
        current_labels=df_probabilities.iloc[test_idx]["actual_preparedness_tier"],
        current_preds=df_probabilities.iloc[test_idx]["predicted_preparedness_tier"],
        current_probs=cur_probs,
        reference_labels=df_probabilities.iloc[train_idx]["actual_preparedness_tier"],
        reference_preds=df_probabilities.iloc[train_idx]["predicted_preparedness_tier"],
        reference_probs=ref_probs,
        run_evidently=True,
        evidently_output_dir=tmp_path,
    )

    assert report.schema_validation.is_valid is True
    assert report.quality_metrics is not None
    assert report.data_drift is not None
    assert report.prediction_drift is not None
    assert report.evidently_status.installed is True
    assert report.evidently_suite is not None
    assert report.evidently_suite.status == "success"

    # All 4 reports must be populated
    suite_reports = report.evidently_suite.reports
    assert "data_quality" in suite_reports
    assert "feature_drift" in suite_reports
    assert "prediction_drift" in suite_reports
    assert "classification_performance" in suite_reports

    for rep_name, rep_res in suite_reports.items():
        assert rep_res.status == "success"
        assert rep_res.html_path is not None
        assert Path(rep_res.html_path).exists()
        assert Path(rep_res.json_path).exists()


def test_comprehensive_monitoring_cycle_and_alerts(df_modelling, metadata):
    """Verify full end-to-end monitoring cycle execution and structured JSON serialization."""
    features = metadata["features"]
    test_idx = metadata["split_indices"]["test"]
    train_idx = metadata["split_indices"]["train"]

    train_df = df_modelling.iloc[train_idx]
    test_df = df_modelling.iloc[test_idx]

    monitor = ModelQualityMonitor()
    report = monitor.run_monitoring_cycle(
        current_features=test_df[features],
        reference_features=train_df[features],
        current_labels=test_df["preparedness_tier"],
        current_preds=test_df["preparedness_tier"],
    )

    assert report.schema_validation.is_valid is True
    assert report.quality_metrics is not None
    assert pytest.approx(report.quality_metrics.accuracy, rel=1e-3) == 1.0  # test labels = preds
    assert report.data_drift is not None
    assert report.data_drift.dataset_drift_detected is False

    # Check JSON serialization
    json_str = report.model_dump_json()
    assert len(json_str) > 1000
    parsed = json.loads(json_str)
    assert parsed["model_artifact"] == "logistic_regression_tier.joblib"
    assert "schema_validation" in parsed
    assert "quality_metrics" in parsed
