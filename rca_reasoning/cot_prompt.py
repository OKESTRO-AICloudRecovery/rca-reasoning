"""근본원인 추론용 구조화 Chain-of-Thought 프롬프트 구성 및 다중 가설 생성·검증 모듈.

장애 유형별 템플릿으로 단계별 추론을 유도하는 프롬프트를 동적으로 구성하고, 근본원인 후보마다
가설(서비스 × 장애 유형)을 만들어 관측 근거와 대조하는 점검표로 검증한다. 모델 응답은 구조화 출력
(JSON: root_cause, hypotheses, evidence, confidence, actions)으로 받아 파싱하며, 추론 단계는 추적 로그에 남긴다.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# 장애 유형별 기대 신호: must(원인 서비스 자체에서 보여야 함) / support(함께 보이면 가설 강화) / propagate(호출자 쪽 증상)
FAULT_SIGNALS: Dict[str, Dict[str, List[str]]] = {
    "cpu":    {"must": ["cpu"], "support": ["trace_self", "latency-90", "latency-50"], "propagate": ["trace_edge", "latency-90"]},
    "mem":    {"must": ["mem"], "support": ["trace_self", "log_error", "log_new_template"], "propagate": ["trace_edge", "error"]},
    "disk":   {"must": ["diskio"], "support": ["trace_self", "latency-90"], "propagate": ["trace_edge"]},
    "socket": {"must": ["socket"], "support": ["error", "trace_error", "log_error"], "propagate": ["trace_edge_error", "error"]},
    "delay":  {"must": ["trace_edge", "latency-90"], "support": ["edge_consensus", "latency-50"], "propagate": ["trace_edge"]},
    "loss":   {"must": ["trace_edge_error", "error"], "support": ["log_error", "trace_error"], "propagate": ["error", "log_error"]},
    "unknown": {"must": [], "support": ["trace_self", "log_new_template", "cpu", "mem"], "propagate": ["trace_edge"]},
}
FAULT_LABEL = {"cpu": "CPU 과부하", "mem": "메모리 누수", "disk": "디스크 부하", "socket": "소켓 고갈",
               "delay": "네트워크 지연", "loss": "패킷 손실", "unknown": "미상"}

OUTPUT_SCHEMA: Dict[str, object] = {
    "type": "object",
    "required": ["root_cause", "fault_type", "confidence", "hypotheses", "evidence", "actions"],
    "properties": {
        "root_cause": {"type": "string", "description": "근본 원인 서비스"},
        "fault_type": {"type": "string", "enum": list(FAULT_LABEL)},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "hypotheses": {"type": "array", "items": {"type": "object", "properties": {
            "service": {"type": "string"}, "fault_type": {"type": "string"}, "verdict": {"type": "string", "enum": ["accepted", "rejected", "uncertain"]},
            "reason": {"type": "string"}}}},
        "evidence": {"type": "array", "items": {"type": "string"}},
        "actions": {"type": "array", "items": {"type": "string"}},
    },
}

SYSTEM_KO = ("당신은 클라우드 운영 AI 어시스턴트의 장애 분석 엔진이다. 메트릭·로그·트레이스 관측 근거와 유사 인시던트 사례를 "
             "바탕으로 근본 원인 서비스를 단계별로 추론하고, 반드시 지정된 JSON 형식으로만 답한다.")
STEPS_KO = ["1단계 — 증상 정리: 어떤 서비스에서 어떤 지표가 정상 대비 얼마나 벗어났는지 요약한다.",
            "2단계 — 가설 검토: 제시된 각 가설(서비스 × 장애 유형)에 대해 기대 신호 점검표와 근거를 대조해 채택/기각/불확실을 판단한다.",
            "3단계 — 전파 관계 확인: 호출 그래프에서 증상이 나타난 서비스가 원인 서비스의 호출자인지 확인하고, 호출자 증상은 원인 서비스로 귀속한다.",
            "4단계 — 유사 사례 대조: 검색된 과거 인시던트의 원인·조치가 현재 상황과 맞는지 비교한다.",
            "5단계 — 결론: 근본 원인 서비스·장애 유형·신뢰도(0~1)·근거·권장 조치를 JSON으로 출력한다."]
FAULT_HINT_KO = {
    "cpu": "CPU 과부하는 원인 서비스의 CPU 사용량과 자체 처리시간이 함께 오르고, 호출자에는 지연만 전파된다.",
    "mem": "메모리 누수는 원인 서비스의 메모리가 단조 증가하며 재시작·GC 관련 신규 로그가 동반되기 쉽다.",
    "disk": "디스크 부하는 디스크 I/O와 자체 처리시간이 오르고 CPU는 크게 변하지 않는 경우가 많다.",
    "socket": "소켓 고갈은 연결 오류·오류 로그가 급증하고 호출 구간 오류율이 함께 오른다.",
    "delay": "네트워크 지연은 특정 호출 구간의 지연만 오르고 서비스 자체 처리시간은 변하지 않는다.",
    "loss": "패킷 손실은 호출 구간 오류율·재시도·타임아웃 로그가 늘고 지연도 불규칙하게 오른다.",
}


@dataclass
class Hypothesis:
    service: str
    fault: str
    prior: float                                   # 후보 순위 점수(정규화)
    checklist: List[Tuple[str, str, bool, float]] = field(default_factory=list)  # (구분, 신호, 충족, z)
    support: float = 0.0
    status: str = "pending"                        # supported | weak | rejected | pending

    def to_dict(self) -> Dict[str, object]:
        return {"service": self.service, "fault_type": self.fault, "prior": round(self.prior, 4), "support": round(self.support, 4),
                "status": self.status, "checklist": [{"group": g, "signal": s, "matched": m, "z": round(z, 2)} for g, s, m, z in self.checklist]}


def _kind_z(evidence: Mapping[str, Sequence], service: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for e in evidence.get(service, []):
        kind, z = (e.kind, e.z) if hasattr(e, "kind") else (e[0], e[1])
        out[kind] = max(out.get(kind, 0.0), float(z))
    return out


def infer_fault(kind_z: Mapping[str, float], z_threshold: float = 3.0) -> str:
    """서비스의 이상 신호 조합으로 가장 그럴듯한 장애 유형을 고른다."""
    best, best_s = "unknown", 0.0
    for fault, sig in FAULT_SIGNALS.items():
        if fault == "unknown":
            continue
        must = [kind_z.get(k, 0.0) for k in sig["must"]]
        if not must or max(must) < z_threshold:
            continue
        s = max(must) + 0.3 * sum(min(kind_z.get(k, 0.0), 20.0) for k in sig["support"])
        if s > best_s:
            best, best_s = fault, s
    return best


def generate_hypotheses(candidates: Sequence[Tuple[str, float]], evidence: Mapping[str, Sequence],
                        max_h: int = 3, fault_hint: Optional[str] = None, z_threshold: float = 3.0) -> List[Hypothesis]:
    """근본원인 후보 상위 max_h 개 → (서비스, 장애 유형) 가설. 유형은 힌트가 없으면 신호 조합으로 추정."""
    top = list(candidates[:max_h])
    total = sum(max(s, 0.0) for _, s in top) or 1.0
    hyps = []
    for svc, score in top:
        kz = _kind_z(evidence, svc)
        fault = fault_hint or infer_fault(kz, z_threshold)
        hyps.append(verify_hypothesis(Hypothesis(svc, fault, max(score, 0.0) / total), evidence, z_threshold))
    return hyps


def verify_hypothesis(h: Hypothesis, evidence: Mapping[str, Sequence], z_threshold: float = 3.0) -> Hypothesis:
    """기대 신호 점검표를 채우고 지지도를 계산한다. must 미충족이면 기각."""
    kz = _kind_z(evidence, h.service)
    sig = FAULT_SIGNALS.get(h.fault, FAULT_SIGNALS["unknown"])
    h.checklist = []
    for group, w in (("must", 1.0), ("support", 0.5)):
        for k in sig[group]:
            z = kz.get(k, 0.0)
            h.checklist.append((group, k, z >= z_threshold, z))
    must = [m for g, _, m, _ in h.checklist if g == "must"]
    sup = [m for g, _, m, _ in h.checklist if g == "support"]
    must_ok = (any(must) if must else True)
    h.support = round((0.6 * (sum(must) / len(must)) if must else 0.6) + 0.4 * (sum(sup) / len(sup) if sup else 0.0), 4)
    h.status = "rejected" if not must_ok else ("supported" if h.support >= 0.5 else "weak")
    return h


def build_prompt(observations: str, candidates: Sequence[Tuple[str, float]], hypotheses: Sequence[Hypothesis],
                 similar_cases: Iterable[str] = (), graph_context: str = "", few_shot: Iterable[Mapping[str, str]] = (),
                 fault_hint: Optional[str] = None, lang: str = "ko") -> Tuple[List[Dict[str, str]], Dict[str, object]]:
    """메시지 목록(system/user, few-shot 포함)과 출력 JSON 스키마를 돌려준다."""
    parts: List[str] = []
    parts.append("## 관측 근거\n" + observations.strip())
    parts.append("## 근본원인 후보 (분석 엔진 순위)\n" + "\n".join(f"{i}. {s} (점수 {v:.3f})" for i, (s, v) in enumerate(candidates, 1)))
    hl = []
    for h in hypotheses:
        chk = ", ".join(f"{k}{'✓' if m else '✗'}" for _, k, m, _ in h.checklist)
        hl.append(f"- {h.service} / {FAULT_LABEL.get(h.fault, h.fault)}: 지지도 {h.support:.2f} ({h.status}) — 점검 {chk}")
    parts.append("## 검토할 가설\n" + ("\n".join(hl) if hl else "- 없음"))
    if graph_context:
        parts.append("## 서비스 의존 관계\n" + graph_context.strip())
    cases = [c for c in similar_cases if c]
    if cases:
        parts.append("## 유사 인시던트 사례\n" + "\n".join(f"- {c}" for c in cases))
    hint = FAULT_HINT_KO.get(fault_hint or "")
    steps = "## 추론 절차\n" + "\n".join(STEPS_KO) + (f"\n참고: {hint}" if hint else "")
    parts.append(steps)
    parts.append("## 출력 형식\n다음 JSON 스키마를 따르는 객체 하나만 출력한다.\n" + json.dumps(OUTPUT_SCHEMA, ensure_ascii=False))
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_KO}]
    for ex in few_shot:
        messages.append({"role": "user", "content": ex["user"]})
        messages.append({"role": "assistant", "content": ex["assistant"]})
    messages.append({"role": "user", "content": "\n\n".join(parts)})
    return messages, OUTPUT_SCHEMA


_JSON_BLOCK = re.compile(r"\{.*\}", re.S)


def parse_structured_answer(text: str) -> Dict[str, object]:
    """모델 응답에서 JSON 객체를 꺼낸다. 실패하면 정규식으로 root_cause/confidence만 복구한다."""
    m = _JSON_BLOCK.search(text or "")
    if m:
        raw = m.group(0)
        for cand in (raw, raw[: raw.rfind("}") + 1]):
            try:
                d = json.loads(cand)
                if isinstance(d, dict):
                    d.setdefault("confidence", 0.0)
                    return d
            except json.JSONDecodeError:
                continue
    rc = re.search(r"root[_ ]cause\W+([A-Za-z0-9_.-]+)", text or "", re.I)
    cf = re.search(r"confidence\W+([01](?:\.\d+)?)", text or "", re.I)
    return {"root_cause": rc.group(1) if rc else "", "confidence": float(cf.group(1)) if cf else 0.0, "parse_error": True}


class ReasoningTrace:
    """추론 단계 기록 (프롬프트 구성 → 가설 → 검증 → 응답 파싱)."""

    def __init__(self, case_id: str):
        self.case_id = case_id
        self.steps: List[Dict[str, object]] = []
        self.started = time.time()

    def log(self, step: str, **data: object) -> None:
        self.steps.append({"t": round(time.time() - self.started, 4), "step": step, **data})

    def as_dict(self) -> Dict[str, object]:
        return {"case_id": self.case_id, "started": self.started, "steps": self.steps}

    def dump(self, path: str) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(self.as_dict(), ensure_ascii=False, default=str) + "\n")
