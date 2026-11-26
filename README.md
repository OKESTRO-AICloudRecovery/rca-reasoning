# rca-reasoning

LLM이 장애 근본 원인을 **단계별로 추론하고 가설을 검증하도록** 프롬프트를 구성하고, 응답을 구조화된 결과로 받아 내는 패키지입니다.

[![CI](https://github.com/OKESTRO-AICloudRecovery/rca-reasoning/actions/workflows/ci.yml/badge.svg)](https://github.com/OKESTRO-AICloudRecovery/rca-reasoning/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)

관측 근거와 후보 순위를 그대로 LLM에 주면 그럴듯하지만 근거와 맞지 않는 결론이 나오기 쉽습니다.
이 패키지는 장애 유형마다 "원인 서비스에서 반드시 보여야 할 신호 / 함께 보이면 강화되는 신호 / 호출자에 번지는 증상"을 정의해 두고,
후보 서비스마다 가설(서비스 × 장애 유형)을 만들어 점검표로 먼저 검증한 뒤, 그 결과를 5단계 추론 절차와 함께 프롬프트에 넣습니다.
모델 응답은 JSON 스키마로 받아 파싱하고, 형식이 깨져도 원인과 신뢰도만큼은 복구합니다. 특정 LLM이나 SDK에 묶이지 않으며 외부 의존 패키지가 없습니다.

## 기능

- **장애 유형별 신호 정의** (`FAULT_SIGNALS`): CPU 과부하·메모리 누수·디스크 부하·소켓 고갈·네트워크 지연·패킷 손실
- **가설 생성·검증** (`generate_hypotheses`, `verify_hypothesis`, `infer_fault`): 후보 순위와 서비스별 근거(z-점수)로 가설을 만들고 지지도·채택 여부 판정
- **프롬프트 구성** (`build_prompt`): 관측 근거 → 후보 순위 → 검토할 가설 → 서비스 의존 관계 → 유사 인시던트 → 추론 절차 → 출력 형식, few-shot 예시와 장애 유형 힌트 선택
- **구조화 출력** (`OUTPUT_SCHEMA`, `parse_structured_answer`): root_cause · fault_type · confidence · hypotheses · evidence · actions
- **추론 추적** (`ReasoningTrace`): 단계별 기록을 남겨 디버깅·감사에 사용

## 설치

```bash
pip install git+https://github.com/OKESTRO-AICloudRecovery/rca-reasoning.git
```

## 사용 예

```python
from rca_reasoning import ReasoningTrace, build_prompt, generate_hypotheses, parse_structured_answer

ranking = [("payment", 4.2), ("checkout", 1.9), ("frontend", 1.1)]          # 근본원인 분석 엔진의 후보 순위
evidence = {"payment": [("cpu", 25.0), ("trace_self", 12.0)],                # 서비스별 (특징 종류, z-점수)
            "checkout": [("latency-90", 6.0), ("trace_edge", 5.0)]}

trace = ReasoningTrace("INC-2026-001")
hyps = generate_hypotheses(ranking, evidence, max_h=3)
trace.log("hypotheses", top=hyps[0].to_dict())

messages, schema = build_prompt(
    observations="payment CPU 95% (z=25), checkout p90 지연 3.1배",
    candidates=ranking, hypotheses=hyps,
    similar_cases=["INC-101: payment CPU 과부하 → 레플리카 확장"],
    graph_context="frontend → checkout → payment",
    fault_hint=hyps[0].fault,
)
# messages 는 OpenAI 호환 chat 형식([{"role": ..., "content": ...}]) — 어떤 LLM 서버에도 그대로 보낼 수 있다
reply = '{"root_cause": "payment", "fault_type": "cpu", "confidence": 0.8, "hypotheses": [], "evidence": [], "actions": []}'
answer = parse_structured_answer(reply)
trace.log("answer", **answer)
trace.dump("trace.json")
```

입력으로 [MSO-RCA](https://github.com/OKESTRO-AICloudRecovery/MSO-RCA)의 순위·근거, [trace-knowledge-graph](https://github.com/OKESTRO-AICloudRecovery/trace-knowledge-graph)의 그래프 텍스트, [incident-retriever](https://github.com/OKESTRO-AICloudRecovery/incident-retriever)의 유사 사례를 그대로 쓸 수 있습니다.

## 테스트

```bash
pip install -e ".[dev]"
pytest -q
```

## 라이선스

[Apache License 2.0](LICENSE)

## 사사

이 소프트웨어는 2025년도 정부(과학기술정보통신부)의 재원으로 정보통신기획평가원의 지원을 받아 수행된 연구의 결과물입니다.
(RS-2025-25441560, 클라우드 장애극복을 위한 AI 어시스턴트 기반 운영·관리 자동화 기술개발)

---

**English.** Structured chain-of-thought prompting for LLM-based root cause analysis: per-fault expected-signal tables, hypothesis generation and checklist verification for each candidate service, a five-step reasoning prompt (symptoms, hypotheses, propagation, similar cases, conclusion), JSON-schema output parsing with fallback recovery, and a reasoning trace log. Model- and SDK-agnostic, pure Python.
