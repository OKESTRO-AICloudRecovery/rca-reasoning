import json
import os

from rca_reasoning import (FAULT_SIGNALS, OUTPUT_SCHEMA, ReasoningTrace, build_prompt, generate_hypotheses, infer_fault,
                           parse_structured_answer, verify_hypothesis)

RANKING = [("payment", 4.2), ("checkout", 1.9), ("frontend", 1.1)]
EVIDENCE = {"payment": [("cpu", 25.0), ("trace_self", 12.0), ("latency-90", 7.0)],
            "checkout": [("latency-90", 6.0), ("trace_edge", 5.0)],
            "frontend": [("latency-90", 3.5)]}


def test_fault_signal_table_covers_six_fault_types():
    assert {"cpu", "mem", "disk", "socket", "delay", "loss"} <= set(FAULT_SIGNALS)


def test_infer_fault_from_signals():
    assert infer_fault({"cpu": 25.0, "latency-90": 5.0}) == "cpu"
    assert infer_fault({"latency-90": 1.0}) == "unknown"


def test_hypotheses_ranked_and_verified():
    hyps = generate_hypotheses(RANKING, EVIDENCE, max_h=3)
    assert hyps[0].service == "payment" and hyps[0].fault == "cpu" and hyps[0].status == "supported", [h.to_dict() for h in hyps]
    assert hyps[1].fault in ("delay", "unknown"), hyps[1].to_dict()
    again = verify_hypothesis(hyps[0], EVIDENCE)
    assert again.status == hyps[0].status


def test_build_prompt_sections():
    hyps = generate_hypotheses(RANKING, EVIDENCE, max_h=3)
    msgs, schema = build_prompt("payment cpu 95% (z=25)", RANKING, hyps, ["c1: payment cpu 과부하 → 스케일아웃"],
                                "frontend → checkout → payment",
                                few_shot=[{"user": "예시 질문", "assistant": '{"root_cause": "x"}'}], fault_hint="cpu")
    assert msgs[0]["role"] == "system" and len(msgs) == 4
    assert "## 검토할 가설" in msgs[-1]["content"] and "CPU 과부하는" in msgs[-1]["content"]
    assert schema == OUTPUT_SCHEMA


def test_parse_structured_answer():
    ans = parse_structured_answer('추론 결과입니다.\n```json\n{"root_cause": "payment", "fault_type": "cpu", "confidence": 0.82, '
                                  '"hypotheses": [], "evidence": ["cpu z=25"], "actions": ["scale out"]}\n```')
    assert ans["root_cause"] == "payment" and ans["confidence"] == 0.82
    bad = parse_structured_answer("root cause: payment, confidence 0.6 (no json)")
    assert bad.get("parse_error") and bad["root_cause"] == "payment"


def test_reasoning_trace(tmp_path):
    tr = ReasoningTrace("case-1")
    tr.log("hypotheses", n=3)
    assert tr.as_dict()["steps"][0]["step"] == "hypotheses"
    p = os.path.join(tmp_path, "trace.json")
    tr.dump(p)
    with open(p, encoding="utf-8") as f:
        assert json.load(f)["case_id"] == "case-1"
