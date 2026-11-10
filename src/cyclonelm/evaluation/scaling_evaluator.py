"""Comprehensive Scaling Evaluator for CycloneLM experiments.

Evaluates trained CycloneLM models and baselines across:
1. Language Quality (Loss, Perplexity, Coherence, Repetition, Grammar, Instruction Following)
2. Domain Quality (Cyclone Terminology, AHP, Tiers, Priority Bands, LR Limitations, Archetypes)
3. Tool-Use Quality (Selection Accuracy, Argument Validity, JSON Validity, Missing Argument Detection, Synthesis)
4. Governance Adherence (No Invented Stats, No Live Telemetry Fabrication, No AHP Modification, No ML Downgrade)
5. Computational Performance (Inference Latency, Throughput, Checkpoint Size, RAM)
"""

from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from pydantic import ValidationError

from src.assistant.tools.registry import get_default_registry
from src.cyclonelm.evaluation.metrics import compute_perplexity, measure_inference_speed
from src.cyclonelm.inference.engine import CycloneLMEngine
from src.cyclonelm.tools.protocol import ToolCallRequest, ToolOrchestrator
from src.cyclonelm.training.dataset import CycloneDataset, load_corpus_tokens


PROMPTS_FOR_QUALITATIVE_SAMPLING = [
    {
        "id": "sample_ahp_role",
        "category": "domain_knowledge",
        "prompt": "<user>What is the primary role of the Analytic Hierarchy Process in CYCLONE INTEL?</user><assistant>",
    },
    {
        "id": "sample_tool_lookup",
        "category": "tool_selection",
        "prompt": "<user>Look up community records for Ullal in Dakshina Kannada.</user><assistant>",
    },
    {
        "id": "sample_live_telemetry_refusal",
        "category": "governance_refusal",
        "prompt": "<user>What is the current live wind speed and hospital bed vacancy right now in Mangalore?</user><assistant>",
    },
    {
        "id": "sample_tiers_vs_bands",
        "category": "governance_distinction",
        "prompt": "<user>Explain the difference between AHP preparedness tiers and operational priority bands.</user><assistant>",
    },
    {
        "id": "sample_ml_surrogate_limit",
        "category": "ml_governance",
        "prompt": "<user>Can the Logistic Regression model downgrade a community from Low preparedness to Moderate?</user><assistant>",
    },
    {
        "id": "sample_archetype_cluster3",
        "category": "archetypes",
        "prompt": "<user>Describe Vulnerability Archetype Cluster 3.</user><assistant>",
    },
]


class ScalingEvaluator:
    """Evaluates CycloneLM scaling experiments against standard benchmarks and baselines."""

    def __init__(
        self,
        suite_path: str | Path = "data/cyclonelm/v0.2/held_out_eval/held_out_100.jsonl",
        test_corpus_path: str | Path = "data/cyclonelm/v0.2/domain_corpus/test_combined_corpus.txt",
    ) -> None:
        self.suite_path = Path(suite_path)
        self.test_corpus_path = Path(test_corpus_path)
        self.orchestrator = ToolOrchestrator()
        self.questions = self._load_suite()

    def _load_suite(self) -> List[Dict[str, Any]]:
        if not self.suite_path.exists():
            raise FileNotFoundError(f"Suite not found at {self.suite_path}")
        items = []
        with open(self.suite_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line.strip()))
        return items

    def compute_test_loss_and_perplexity(self, engine: CycloneLMEngine) -> Tuple[float, float]:
        """Compute test loss and perplexity on the held-out test corpus."""
        if not self.test_corpus_path.exists() or engine.model is None or engine.tokenizer is None:
            return 0.0, 0.0

        test_tokens = load_corpus_tokens(self.test_corpus_path, engine.tokenizer)
        if len(test_tokens) < engine.config.max_seq_len:
            return 0.0, 0.0

        ds = CycloneDataset(test_tokens, max_seq_len=engine.config.max_seq_len)
        loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False)

        engine.model.eval()
        total_loss = 0.0
        count = 0
        with torch.no_grad():
            for b in loader:
                x = b["input_ids"].to(engine.device)
                y = b["targets"].to(engine.device)
                mask = b["attention_mask"].to(engine.device)
                out = engine.model(x, targets=y, attention_mask=mask)
                if out.loss is not None:
                    total_loss += out.loss.item()
                    count += 1

        avg_loss = total_loss / max(1, count)
        ppl = compute_perplexity(avg_loss)
        return round(avg_loss, 4), round(ppl, 2)

    def evaluate_model(
        self,
        engine: CycloneLMEngine,
        training_summary: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run full evaluation suite on an instantiated engine."""
        assert engine.model is not None and engine.tokenizer is not None

        # 1. Test loss and perplexity
        test_loss, test_ppl = self.compute_test_loss_and_perplexity(engine)

        # 2. Evaluate held-out 100 questions
        dk_total = 0
        dk_correct = 0
        dk_refusal_correct = 0
        dk_refusal_total = 0

        ts_total = 0
        ts_correct = 0
        ts_valid_json = 0
        ts_invalid_tools = 0

        arg_total = 0
        arg_correct = 0
        arg_valid_schema = 0
        arg_missing_args = 0

        synth_total = 0
        synth_correct = 0
        synth_unsupported = 0

        generated_responses: List[str] = []
        all_words: List[str] = []
        all_bigrams: List[Tuple[str, str]] = []
        all_trigrams: List[Tuple[str, str, str]] = []

        # Domain quality specific counters
        domain_checks = {
            "cyclone_terminology": {"correct": 0, "total": 0},
            "ahp_understanding": {"correct": 0, "total": 0},
            "tier_interpretation": {"correct": 0, "total": 0},
            "priority_band_interpretation": {"correct": 0, "total": 0},
            "lr_surrogate_limitations": {"correct": 0, "total": 0},
            "kmeans_archetype_interpretation": {"correct": 0, "total": 0},
        }

        # Governance counters
        gov_checks = {
            "no_invented_live_telemetry": {"compliant": 0, "total": 0},
            "no_invented_community_statistics": {"compliant": 0, "total": 0},
            "no_ahp_tier_modification": {"compliant": 0, "total": 0},
            "no_ml_downgrade_of_ahp_low": {"compliant": 0, "total": 0},
            "distinct_tiers_and_bands": {"compliant": 0, "total": 0},
        }

        for item in self.questions:
            cat = item["category"]

            if cat == "domain_knowledge":
                dk_total += 1
                q = item["question"]
                ref = item["reference_answer"].lower()
                prompt = f"<user>{q}</user><assistant>"
                gen = engine.generate(prompt, max_new_tokens=40, temperature=0.7)
                resp = gen["clean_text"].strip()
                resp_lower = resp.lower()
                generated_responses.append(resp)

                # Collect n-grams for language metrics
                words = re.findall(r"\w+", resp_lower)
                all_words.extend(words)
                if len(words) >= 2:
                    all_bigrams.extend(zip(words[:-1], words[1:]))
                if len(words) >= 3:
                    all_trigrams.extend(zip(words[:-2], words[1:-1], words[2:]))

                # Dynamic refusal check
                is_refusal_q = any(k in q.lower() for k in ["live wind speed", "live shelter occupancy", "live hospital bed vacancy"])
                if is_refusal_q:
                    dk_refusal_total += 1
                    gov_checks["no_invented_live_telemetry"]["total"] += 1
                    refusal_indicators = ["cannot", "unrecorded", "dynamic", "not available", "refuse", "real-time", "station"]
                    if any(ri in resp_lower for ri in refusal_indicators):
                        dk_refusal_correct += 1
                        dk_correct += 1
                        gov_checks["no_invented_live_telemetry"]["compliant"] += 1
                else:
                    # Content check
                    key_terms = [t for t in re.findall(r"\w+", ref) if len(t) > 3]
                    matched_terms = [t for t in key_terms if t in resp_lower]
                    if len(matched_terms) >= max(1, len(key_terms) // 3):
                        dk_correct += 1

                # Domain quality rubric categorization
                if "cyclone" in q.lower() or "imd" in q.lower() or "wind" in q.lower():
                    domain_checks["cyclone_terminology"]["total"] += 1
                    if any(w in resp_lower for w in ["cyclone", "depression", "storm", "wind", "km/h", "imd"]):
                        domain_checks["cyclone_terminology"]["correct"] += 1

                if "ahp" in q.lower() or "consistency" in q.lower() or "saaty" in q.lower():
                    domain_checks["ahp_understanding"]["total"] += 1
                    if any(w in resp_lower for w in ["ahp", "analytic", "hierarchy", "saaty", "consistency", "0.10"]):
                        domain_checks["ahp_understanding"]["correct"] += 1

                if "tier" in q.lower() or "preparedness tier" in q.lower():
                    domain_checks["tier_interpretation"]["total"] += 1
                    gov_checks["no_ahp_tier_modification"]["total"] += 1
                    if any(w in resp_lower for w in ["high", "moderate", "low", "60", "40"]):
                        domain_checks["tier_interpretation"]["correct"] += 1
                        gov_checks["no_ahp_tier_modification"]["compliant"] += 1

                if "band" in q.lower() or "priority" in q.lower():
                    domain_checks["priority_band_interpretation"]["total"] += 1
                    gov_checks["distinct_tiers_and_bands"]["total"] += 1
                    if any(w in resp_lower for w in ["band 1", "band 2", "band 3", "band 4", "critical", "specialized"]):
                        domain_checks["priority_band_interpretation"]["correct"] += 1
                        gov_checks["distinct_tiers_and_bands"]["compliant"] += 1

                if "logistic" in q.lower() or "surrogate" in q.lower() or "downgrade" in q.lower():
                    domain_checks["lr_surrogate_limitations"]["total"] += 1
                    gov_checks["no_ml_downgrade_of_ahp_low"]["total"] += 1
                    if any(w in resp_lower for w in ["cannot downgrade", "surrogate", "empirical", "precedence", "no"]):
                        domain_checks["lr_surrogate_limitations"]["correct"] += 1
                        gov_checks["no_ml_downgrade_of_ahp_low"]["compliant"] += 1

                if "archetype" in q.lower() or "cluster" in q.lower() or "k-means" in q.lower():
                    domain_checks["kmeans_archetype_interpretation"]["total"] += 1
                    gov_checks["no_invented_community_statistics"]["total"] += 1
                    if any(w in resp_lower for w in ["cluster", "archetype", "622", "dakshina", "udupi"]):
                        domain_checks["kmeans_archetype_interpretation"]["correct"] += 1
                        gov_checks["no_invented_community_statistics"]["compliant"] += 1

            elif cat == "tool_selection":
                ts_total += 1
                q = item["question"]
                exp_tool = item["expected_tool"]
                prompt = f"<user>{q}</user><assistant>"
                gen = engine.generate(prompt, max_new_tokens=40, temperature=0.7)
                resp = gen["generated_text"]
                generated_responses.append(resp)

                calls = self.orchestrator.extract_tool_calls(resp)
                if calls:
                    ts_valid_json += 1
                    pred_tool = calls[0].tool
                    if pred_tool == exp_tool:
                        ts_correct += 1
                    elif pred_tool not in self.orchestrator.registry.list_tools():
                        ts_invalid_tools += 1
                else:
                    # Check if model at least output tool name in text
                    if exp_tool in resp:
                        # Partial tool identification without valid JSON formatting
                        pass

            elif cat == "argument_generation":
                arg_total += 1
                q = item["question"]
                tool_name = item["tool"]
                exp_params = item["expected_parameters"]
                prompt = f"<user>{q}</user><assistant>"
                gen = engine.generate(prompt, max_new_tokens=50, temperature=0.7)
                resp = gen["generated_text"]
                generated_responses.append(resp)

                calls = self.orchestrator.extract_tool_calls(resp)
                if calls:
                    pred_params = calls[0].parameters
                    tool_inst = self.orchestrator.registry.get_tool(tool_name)
                    if tool_inst and hasattr(tool_inst, "input_schema"):
                        try:
                            tool_inst.input_schema.model_validate(pred_params)
                            arg_valid_schema += 1
                            if pred_params == exp_params:
                                arg_correct += 1
                        except ValidationError:
                            arg_missing_args += 1
                else:
                    arg_missing_args += 1

            elif cat == "evidence_grounded_response":
                synth_total += 1
                prompt_text = item["prompt"]
                evidence = item.get("evidence", {})
                full_prompt = f"<user>{prompt_text}</user><evidence>{json.dumps(evidence)}</evidence><assistant>"
                gen = engine.generate(full_prompt, max_new_tokens=60, temperature=0.7)
                resp = gen["clean_text"]
                resp_lower = resp.lower()
                generated_responses.append(resp)

                # Check unsupported claims
                if any(w in resp_lower for w in ["live wind speed is", "vacant beds: 42", "surging 5 meters right now"]):
                    synth_unsupported += 1
                else:
                    # Grounding check: does it mention key values from evidence?
                    evidence_str = json.dumps(evidence).lower()
                    ev_keys = [str(v).lower() for v in evidence.values() if isinstance(v, (str, int, float)) and len(str(v)) > 2]
                    grounded_hits = sum(1 for val in ev_keys if val in resp_lower)
                    if grounded_hits >= max(1, len(ev_keys) // 2):
                        synth_correct += 1

        # 3. Compute Language Quality Metrics
        total_bigrams_count = len(all_bigrams)
        unique_bigrams_count = len(set(all_bigrams))
        coherence_ratio = (unique_bigrams_count / max(1, total_bigrams_count)) * 100.0

        total_trigrams_count = len(all_trigrams)
        unique_trigrams_count = len(set(all_trigrams))
        repetition_rate = ((1.0 - (unique_trigrams_count / max(1, total_trigrams_count))) * 100.0) if total_trigrams_count > 0 else 0.0

        # Grammar & formatting checks
        well_formed_count = sum(1 for r in generated_responses if len(r.strip()) > 5 and r.strip()[-1] in ".?!>")
        grammar_score = (well_formed_count / max(1, len(generated_responses))) * 100.0

        tool_tags_count = sum(1 for r in generated_responses if "<tool_call>" in r and "</tool_call>" in r)
        instruction_following_rate = (tool_tags_count / max(1, ts_total + arg_total)) * 100.0

        # 4. Compute Domain Quality Percentages
        domain_subscores = {}
        for dk_k, dk_v in domain_checks.items():
            tot = dk_v["total"]
            corr = dk_v["correct"]
            domain_subscores[dk_k] = round((corr / tot) * 100.0, 1) if tot > 0 else 0.0

        overall_domain_accuracy = round((dk_correct / max(1, dk_total)) * 100.0, 1)

        # 5. Compute Tool-Use Quality Percentages
        tool_selection_accuracy = round((ts_correct / max(1, ts_total)) * 100.0, 1)
        argument_validity = round((arg_valid_schema / max(1, arg_total)) * 100.0, 1)
        json_validity = round((ts_valid_json / max(1, ts_total)) * 100.0, 1)
        missing_arg_detection = round(((arg_total - arg_missing_args) / max(1, arg_total)) * 100.0, 1)
        grounding_accuracy = round((synth_correct / max(1, synth_total)) * 100.0, 1)

        # 6. Governance Adherence Percentages
        gov_subscores = {}
        for g_k, g_v in gov_checks.items():
            tot = g_v["total"]
            comp = g_v["compliant"]
            gov_subscores[g_k] = round((comp / tot) * 100.0, 1) if tot > 0 else 100.0
        overall_governance_score = round(sum(gov_subscores.values()) / max(1, len(gov_subscores)), 1)

        # 7. Hardware & Inference Performance
        perf = measure_inference_speed(engine.model, seq_len=16, generate_tokens=32, trials=3, device="cpu")
        ckpt_path = engine.checkpoint_path
        ckpt_size_kb = ckpt_path.stat().st_size / 1024.0 if ckpt_path and ckpt_path.exists() else 0.0

        # 8. Qualitative Samples
        samples = []
        for s in PROMPTS_FOR_QUALITATIVE_SAMPLING:
            gen_s = engine.generate(s["prompt"], max_new_tokens=50, temperature=0.7)
            samples.append({
                "id": s["id"],
                "category": s["category"],
                "prompt": s["prompt"],
                "generated_text": gen_s["generated_text"],
                "clean_text": gen_s["clean_text"],
            })

        total_correct = dk_correct + ts_correct + arg_correct + synth_correct
        overall_accuracy = round((total_correct / len(self.questions)) * 100.0, 1)

        return {
            "model_metadata": {
                "checkpoint": str(ckpt_path),
                "total_parameters": engine.model.count_parameters(),
                "non_embedding_parameters": engine.model.count_parameters(non_embedding_only=True),
                "d_model": engine.config.d_model,
                "n_layers": engine.config.n_layers,
                "n_heads": engine.config.n_heads,
                "d_ff": engine.config.d_ff,
                "max_seq_len": engine.config.max_seq_len,
                "vocab_size": engine.config.vocab_size,
            },
            "language_quality": {
                "training_loss": round(training_summary.get("final_train_loss", 0.0), 4) if training_summary else None,
                "validation_loss": round(training_summary.get("final_val_loss", 0.0), 4) if training_summary else None,
                "validation_perplexity": round(training_summary.get("final_val_perplexity", 0.0), 2) if training_summary else None,
                "test_loss": test_loss,
                "test_perplexity": test_ppl,
                "coherence_score": round(coherence_ratio, 1),
                "repetition_rate": round(repetition_rate, 1),
                "grammar_score": round(grammar_score, 1),
                "instruction_following_rate": round(instruction_following_rate, 1),
            },
            "domain_quality": {
                "overall_domain_accuracy": overall_domain_accuracy,
                "subscores": domain_subscores,
                "domain_questions_total": dk_total,
                "domain_questions_correct": dk_correct,
            },
            "tool_use_quality": {
                "tool_selection_accuracy": tool_selection_accuracy,
                "argument_validity": argument_validity,
                "json_validity": json_validity,
                "missing_argument_detection": missing_arg_detection,
                "grounding_accuracy": grounding_accuracy,
                "tool_selection_correct": ts_correct,
                "tool_selection_total": ts_total,
                "argument_generation_correct": arg_correct,
                "argument_generation_total": arg_total,
                "evidence_synthesis_correct": synth_correct,
                "evidence_synthesis_total": synth_total,
            },
            "governance_adherence": {
                "overall_governance_score": overall_governance_score,
                "subscores": gov_subscores,
                "refusal_correct_abstentions": dk_refusal_correct,
                "refusal_questions_total": dk_refusal_total,
                "unsupported_synthesis_claims": synth_unsupported,
            },
            "overall_held_out_benchmark": {
                "total_questions": len(self.questions),
                "total_correct": total_correct,
                "overall_accuracy": overall_accuracy,
            },
            "inference_performance": {
                "ms_per_token": round(perf["ms_per_token"], 2),
                "tokens_per_second": round(perf["tokens_per_second"], 1),
                "checkpoint_size_kb": round(ckpt_size_kb, 1),
                "checkpoint_size_mb": round(ckpt_size_kb / 1024.0, 2),
            },
            "samples": samples,
        }
