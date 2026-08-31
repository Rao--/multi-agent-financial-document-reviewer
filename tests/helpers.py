from __future__ import annotations

import copy
import json
from collections.abc import Iterable
from typing import Any

import httpx

from financial_reviewer.local.model import LocalModelError
from financial_reviewer.local.model import OllamaModel


SYNTHETIC_NAME = "SYNTHETIC PERSON ALPHA"
SYNTHETIC_EMPLOYEE_ID = "SYN-EMP-0001"
SYNTHETIC_EMPLOYER = "SYNTHETIC LABS LLC"
SYNTHETIC_EIN = "00-0000001"
SYNTHETIC_INCOME_SOURCE = "$6,250.00"
MODEL_OUTPUT_SENTINEL = "MODEL-OUTPUT-MUST-NOT-APPEAR"


def pay_stub_payload(document_text: str) -> dict[str, Any]:
    lines = document_text.splitlines()

    def source(line_number: int) -> dict[str, Any]:
        return {
            "line_number": line_number,
            "quote": lines[line_number - 1],
            "confidence": 0.99,
        }

    return {
        "document_type": "pay_stub",
        "fields": {
            "employee_name": {
                "status": "supported",
                "value": SYNTHETIC_NAME,
                "source": source(3),
            },
            "employee_id": {
                "status": "supported",
                "value": SYNTHETIC_EMPLOYEE_ID,
                "source": source(4),
            },
            "employer_name": {
                "status": "supported",
                "value": SYNTHETIC_EMPLOYER,
                "source": source(5),
            },
            "employer_ein": {
                "status": "supported",
                "value": SYNTHETIC_EIN,
                "source": source(6),
            },
            "monthly_income": {
                "status": "supported",
                "value": "6250.00",
                "source": source(7),
            },
            "pay_period_months": {
                "status": "supported",
                "value": 12,
                "source": source(8),
            },
            "pay_period_year": {
                "status": "supported",
                "value": 2025,
                "source": source(9),
            },
        },
    }


def pay_stub_json(document_text: str) -> str:
    return json.dumps(pay_stub_payload(document_text), separators=(",", ":"))


def unsupported_pay_stub_json(document_text: str) -> str:
    payload = copy.deepcopy(pay_stub_payload(document_text))
    payload["fields"]["employee_id"] = {
        "status": "unsupported",
        "reason": "not_present",
    }
    return json.dumps(payload, separators=(",", ":"))


def mismatched_evidence_json(document_text: str) -> str:
    payload = copy.deepcopy(pay_stub_payload(document_text))
    payload["fields"]["monthly_income"]["source"]["quote"] = (
        "Monthly Income: $9,999.99"
    )
    return json.dumps(payload, separators=(",", ":"))


class ScriptedOllama:
    """MockTransport script installed only around synthetic workflow tests."""

    def __init__(self, outputs: Iterable[str | LocalModelError]) -> None:
        self._outputs = list(outputs)
        self.calls: list[dict[str, Any]] = []

    def _handle(self, request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        self.calls.append(payload)
        if not self._outputs:
            raise AssertionError("unexpected model call")
        result = self._outputs.pop(0)
        if isinstance(result, LocalModelError):
            raise result
        return httpx.Response(
            200,
            json={"model": "qwen2.5:3b", "done": True, "response": result},
        )

    def install(self, monkeypatch: Any) -> None:
        transport = httpx.MockTransport(self._handle)

        def test_client_context(_model: OllamaModel, _settings: object) -> httpx.Client:
            return httpx.Client(
                transport=transport,
                trust_env=False,
                follow_redirects=False,
            )

        monkeypatch.setattr(OllamaModel, "_client_context", test_client_context)
