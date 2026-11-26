"""rca_reasoning: LLM 근본원인 추론용 구조화 CoT 프롬프트와 다중 가설 검증.

  cot_prompt  장애 유형별 신호 정의, 후보별 가설(서비스 × 장애 유형) 생성·검증, 5단계 추론 프롬프트 구성,
              구조화 출력(JSON) 파싱, 추론 단계 추적 로그
"""
from .cot_prompt import (FAULT_SIGNALS, OUTPUT_SCHEMA, Hypothesis, ReasoningTrace, build_prompt, generate_hypotheses,
                         infer_fault, parse_structured_answer, verify_hypothesis)

__version__ = "2.0.0"
