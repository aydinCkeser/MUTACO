"""Crash-safe write-ahead physical-call ledger for future SBI-zoo datasets."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Mapping

from mutaco import canonical_json_bytes, stable_hash
from mutaco.atomic_io import atomic_write_bytes

from .sbi_zoo_failures import ScientificBudgetViolation


LEDGER_SCHEMA = "mutaco-sbi-zoo-scientific-call-ledger-v1"
def _event_path(root: Path, slot: str, ordinal: int, status: str) -> Path:
    return root / "slots" / slot / f"event-{ordinal:02d}-{status}.json"


def _append_event(path: Path, document: Mapping[str, Any]) -> None:
    if path.exists():
        raise ScientificBudgetViolation(f"append-only call event already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    body = dict(document)
    body["event_identity"] = stable_hash("sbi-zoo-call-event-v1", body)
    atomic_write_bytes(path, canonical_json_bytes(body) + b"\n")


@dataclass
class WriteAheadScientificCallLedger:
    """Append-only ledger where every invocation is counted before execution."""

    root: Path
    declaration_identity: str
    maximum_calls: int

    def __post_init__(self) -> None:
        if self.maximum_calls not in (40, 100, 140, 480, 17_280, 17_420):
            raise ValueError("call cap is not an approved SBI-zoo cap")
        self.root = Path(self.root)
        self._active_attempts: set[str] = set()
        header = self.root / "ledger.json"
        expected = {
            "schema_version": LEDGER_SCHEMA,
            "declaration_identity": self.declaration_identity,
            "maximum_calls": self.maximum_calls,
            "ledger_identity": self.identity,
            "automatic_retry": False,
            "attempts_include_failures_and_unknown_outcomes": True,
        }
        if header.exists():
            if json.loads(header.read_text(encoding="utf-8")) != expected:
                raise ScientificBudgetViolation("call ledger header identity mismatch")
        else:
            self.root.mkdir(parents=True, exist_ok=True)
            atomic_write_bytes(header, canonical_json_bytes(expected) + b"\n")

    @property
    def identity(self) -> str:
        return stable_hash(
            "sbi-zoo-call-ledger-v1",
            {
                "declaration_identity": self.declaration_identity,
                "maximum_calls": self.maximum_calls,
            },
        )

    def _events(self, slot: str) -> list[dict[str, Any]]:
        directory = self.root / "slots" / slot
        if not directory.exists():
            return []
        result = []
        for path in sorted(directory.glob("event-*.json")):
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ScientificBudgetViolation("call ledger event is not an object")
            identity = value.pop("event_identity", None)
            if identity != stable_hash("sbi-zoo-call-event-v1", value):
                raise ScientificBudgetViolation("call ledger event identity mismatch")
            result.append(value)
        return result

    def attempted_count(self) -> int:
        return sum(
            1
            for path in self.root.glob("slots/*/event-01-attempted.json")
            if path.is_file()
        )

    def slot_status(self, slot: str) -> str | None:
        events = self._events(slot)
        if not events:
            return None
        statuses = [event.get("status") for event in events]
        transitions = {
            ("intent",): "intent",
            ("intent", "attempted"): "attempted_outcome_unknown",
            ("intent", "attempted", "attempted_outcome_unknown"): "attempted_outcome_unknown",
            ("intent", "attempted", "success"): "success",
            ("intent", "attempted", "failed"): "failed",
        }
        try:
            return transitions[tuple(statuses)]
        except KeyError as exc:
            raise ScientificBudgetViolation(
                f"invalid call ledger state sequence {statuses}"
            ) from exc

    def terminal_payload(self, slot: str) -> Mapping[str, Any] | None:
        """Return the durable terminal payload, never inferring one for unknown state."""

        events = self._events(slot)
        if not events or events[-1].get("status") not in {
            "success",
            "failed",
            "attempted_outcome_unknown",
        }:
            return None
        payload = events[-1].get("payload")
        if not isinstance(payload, Mapping):
            raise ScientificBudgetViolation("terminal call event payload is malformed")
        return payload

    def intent_payload(self, slot: str) -> Mapping[str, Any]:
        events = self._events(slot)
        if not events or events[0].get("status") != "intent":
            raise ScientificBudgetViolation("scientific call intent is missing")
        payload = events[0].get("payload")
        if not isinstance(payload, Mapping):
            raise ScientificBudgetViolation("scientific call intent payload is malformed")
        return payload

    def persist_intent(self, slot: str, payload: Mapping[str, Any]) -> None:
        if self.slot_status(slot) is not None:
            raise ScientificBudgetViolation("scientific call slot is not new")
        body = {
            "schema_version": LEDGER_SCHEMA,
            "declaration_identity": self.declaration_identity,
            "ledger_identity": self.identity,
            "slot": slot,
            "status": "intent",
            "payload": dict(payload),
        }
        _append_event(_event_path(self.root, slot, 0, "intent"), body)

    def mark_attempted(self, slot: str) -> None:
        if self.slot_status(slot) != "intent":
            raise ScientificBudgetViolation("only an intent may become attempted")
        if self.attempted_count() >= self.maximum_calls:
            raise ScientificBudgetViolation("scientific simulator-call cap exceeded")
        body = {
            "schema_version": LEDGER_SCHEMA,
            "declaration_identity": self.declaration_identity,
            "ledger_identity": self.identity,
            "slot": slot,
            "status": "attempted",
            "counts_toward_cap": True,
        }
        _append_event(_event_path(self.root, slot, 1, "attempted"), body)
        self._active_attempts.add(slot)

    def resolve(self, slot: str, status: str, payload: Mapping[str, Any]) -> None:
        if status not in ("success", "failed", "attempted_outcome_unknown"):
            raise ValueError(
                "attempt resolution must be success, failed, or attempted_outcome_unknown"
            )
        statuses = tuple(event.get("status") for event in self._events(slot))
        if statuses != ("intent", "attempted") or slot not in self._active_attempts:
            raise ScientificBudgetViolation("only a just-attempted slot may be resolved")
        body = {
            "schema_version": LEDGER_SCHEMA,
            "declaration_identity": self.declaration_identity,
            "ledger_identity": self.identity,
            "slot": slot,
            "status": status,
            "payload": dict(payload),
            "retry_calls": 0,
        }
        _append_event(_event_path(self.root, slot, 2, status), body)
        self._active_attempts.discard(slot)

    def invoke_once(
        self,
        slot: str,
        *,
        intent: Mapping[str, Any],
        invoke: Callable[[], Any],
        persist_outcome: Callable[[Any], Mapping[str, Any]],
    ) -> Any:
        """Persist intent/attempt before one invocation and never retry it."""

        self.persist_intent(slot, intent)
        self.mark_attempted(slot)
        try:
            result = invoke()
        except Exception as exc:
            self.resolve(
                slot,
                "failed",
                {
                    "failure_stage": "simulator_invocation",
                    "failure_type": type(exc).__name__,
                    "failure_explanation": str(exc),
                },
            )
            raise
        try:
            persisted = persist_outcome(result)
            self.resolve(slot, "success", persisted)
        except Exception as exc:
            # Invocation has returned, so any subsequent failure is never a
            # simulator failure. Best-effort explicit recording is useful, but
            # attempted-without-terminal already replays as outcome-unknown.
            try:
                self.resolve(
                    slot,
                    "attempted_outcome_unknown",
                    {
                        "failure_stage": "post_invocation_durability",
                        "failure_type": type(exc).__name__,
                        "failure_explanation": str(exc),
                    },
                )
            except Exception:
                pass
            finally:
                self._active_attempts.discard(slot)
            raise
        return result

    def invoke_unattempted_once(
        self,
        slot: str,
        *,
        intent: Mapping[str, Any],
        invoke: Callable[[], Any],
        persist_outcome: Callable[[Any], Mapping[str, Any]],
    ) -> Any:
        """Invoke a new or intent-only slot, never a scientifically attempted slot.

        An intent-only slot is durably proven unattempted and is therefore the
        sole restart state from which invocation is permitted.  Success,
        failure, and explicit or implicit outcome-unknown states remain
        non-retryable.
        """

        status = self.slot_status(slot)
        if status is None:
            self.persist_intent(slot, intent)
        elif status == "intent":
            if dict(self.intent_payload(slot)) != dict(intent):
                raise ScientificBudgetViolation("restart intent identity mismatch")
        else:
            raise ScientificBudgetViolation(
                f"scientific call slot is non-retryable after status {status}"
            )
        self.mark_attempted(slot)
        try:
            result = invoke()
        except Exception as exc:
            self.resolve(
                slot,
                "failed",
                {
                    "failure_stage": "simulator_invocation",
                    "failure_type": type(exc).__name__,
                    "failure_explanation": str(exc),
                },
            )
            raise
        try:
            persisted = persist_outcome(result)
            self.resolve(slot, "success", persisted)
        except Exception as exc:
            try:
                self.resolve(
                    slot,
                    "attempted_outcome_unknown",
                    {
                        "failure_stage": "post_invocation_durability",
                        "failure_type": type(exc).__name__,
                        "failure_explanation": str(exc),
                    },
                )
            except Exception:
                pass
            finally:
                self._active_attempts.discard(slot)
            raise
        return result

    def accounting(self) -> dict[str, Any]:
        statuses = (
            [
                self.slot_status(path.name)
                for path in sorted((self.root / "slots").glob("*"))
            ]
            if (self.root / "slots").exists()
            else []
        )
        return {
            "maximum_calls": self.maximum_calls,
            "attempted_calls": self.attempted_count(),
            "successful_calls": statuses.count("success"),
            "failed_calls": statuses.count("failed"),
            "attempted_outcome_unknown": statuses.count("attempted_outcome_unknown"),
            "retry_calls": 0,
        }
