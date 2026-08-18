from __future__ import annotations

import json
import os
import sqlite3
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from threading import RLock
from typing import Literal

from pydantic import Field

from evidence_route.budget import BudgetExceeded, PriceConfig
from evidence_route.contracts import StrictModel, Usage


class BillingStateError(RuntimeError):
    """Raised when a paid call cannot be safely resumed."""


class RequestFingerprintMismatch(RuntimeError):
    """Raised when a call id is reused for a different request."""


class CallState(StrEnum):
    RESERVED = "reserved"
    SENT = "sent"
    COMPLETED = "completed"
    USAGE_MISSING = "usage_missing"
    BILLING_UNCERTAIN = "billing_uncertain"


class ResumeDecision(StrictModel):
    action: Literal["send", "reuse", "reuse_and_stop"]
    call_id: str
    payload: dict[str, object] | None = None


class RunCallSummary(StrictModel):
    call_ids: list[str]
    usage: Usage
    actual_cost_micro_cny: int | None = Field(default=None, ge=0)
    known_actual_cost_micro_cny: int = Field(ge=0)
    committed_cost_micro_cny: int = Field(ge=0)
    cost_is_lower_bound: bool
    fresh_call_count: int = Field(ge=0)
    cache_hit_count: int = Field(ge=0)
    transport_attempts: int = Field(ge=0)
    requested_aliases: list[str]
    response_model_ids_raw: list[str]
    usage_sources: list[Literal["provider", "missing"]]
    identity_verified: bool
    billing_uncertain: bool


_SENSITIVE_KEYS = {"authorization", "api_key", "token", "secret", "password"}
_INITIALIZE_LOCK = RLock()


def redact_payload(value: object) -> object:
    if isinstance(value, dict):
        return {
            str(key): "[REDACTED]"
            if str(key).lower() in _SENSITIVE_KEYS
            else redact_payload(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_payload(item) for item in value]
    return value


def _timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


class TraceWriter:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, payload: object) -> None:
        encoded = json.dumps(
            redact_payload(payload), ensure_ascii=False, separators=(",", ":")
        )
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def atomic_write_json(path: Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


class SQLiteRunStore:
    """Transactional call cache and budget reservation ledger."""

    def __init__(
        self, path: Path, *, activity_id: str, cap_cny: float, pricing: PriceConfig
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.activity_id = activity_id
        self.cap_micro_cny = int(Decimal(str(cap_cny)) * Decimal(1_000_000))
        if self.cap_micro_cny <= 0:
            raise ValueError("cap_cny must be positive")
        self.pricing = pricing
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path, timeout=5.0, isolation_level=None, check_same_thread=False
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA busy_timeout=5000")
        return connection

    def _initialize(self) -> None:
        with _INITIALIZE_LOCK:
            self._initialize_locked()

    def _initialize_locked(self) -> None:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS store_metadata (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    schema_version INTEGER NOT NULL CHECK (schema_version = 1),
                    activity_id TEXT NOT NULL,
                    cap_micro_cny INTEGER NOT NULL CHECK (cap_micro_cny > 0),
                    currency TEXT NOT NULL CHECK (currency = 'CNY'),
                    price_config_id TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS calls (
                    call_id TEXT PRIMARY KEY,
                    request_sha256 TEXT NOT NULL CHECK (length(request_sha256) = 64),
                    activity_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    node TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    logical_attempt INTEGER NOT NULL,
                    state TEXT NOT NULL CHECK (state IN (
                        'reserved', 'sent', 'completed', 'usage_missing', 'billing_uncertain'
                    )),
                    reserved_micro_cny INTEGER NOT NULL CHECK (reserved_micro_cny >= 0),
                    actual_micro_cny INTEGER CHECK (actual_micro_cny >= 0),
                    payload_json TEXT,
                    usage_json TEXT,
                    usage_source TEXT CHECK (usage_source IN ('provider', 'missing')),
                    requested_alias TEXT,
                    response_model_id_raw TEXT,
                    identity_verified INTEGER CHECK (identity_verified IN (0, 1)),
                    transport_attempts INTEGER NOT NULL DEFAULT 0 CHECK (transport_attempts >= 0),
                    cache_hits INTEGER NOT NULL DEFAULT 0 CHECK (cache_hits >= 0),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE (run_id, node, task_id, logical_attempt)
                );
                CREATE INDEX IF NOT EXISTS idx_calls_run_id ON calls(run_id);
                CREATE INDEX IF NOT EXISTS idx_calls_activity_id ON calls(activity_id);
                """
            )
            row = connection.execute("SELECT * FROM store_metadata WHERE singleton = 1").fetchone()
            expected = (
                self.activity_id,
                self.cap_micro_cny,
                self.pricing.currency,
                self.pricing.config_id,
            )
            if row is None:
                connection.execute(
                    "INSERT INTO store_metadata VALUES (1, 1, ?, ?, ?, ?)", expected
                )
            elif (
                row["activity_id"],
                row["cap_micro_cny"],
                row["currency"],
                row["price_config_id"],
            ) != expected:
                raise ValueError("run store metadata does not match requested activity")
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _check_fingerprint(row: sqlite3.Row, request_sha256: str) -> None:
        if row["request_sha256"] != request_sha256:
            raise RequestFingerprintMismatch("call id is bound to a different request")

    def reserve_call(
        self,
        call_id: str,
        *,
        request_sha256: str,
        run_id: str,
        node: str,
        task_id: str,
        logical_attempt: int,
        max_input_tokens: int,
        max_output_tokens: int,
    ) -> int:
        projected = self.pricing.estimate_micro_cny(
            Usage(
                input_tokens=max_input_tokens,
                output_tokens=max_output_tokens,
                total_tokens=max_input_tokens + max_output_tokens,
                complete=True,
            )
        )
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM calls WHERE call_id = ?", (call_id,)).fetchone()
            if row is not None:
                self._check_fingerprint(row, request_sha256)
                state = CallState(row["state"])
                if state in {CallState.SENT, CallState.BILLING_UNCERTAIN}:
                    raise BillingStateError(f"{state.value} call has unknown billing")
                connection.commit()
                return row["reserved_micro_cny"]
            conflict = connection.execute(
                """SELECT call_id FROM calls
                   WHERE run_id = ? AND node = ? AND task_id = ? AND logical_attempt = ?""",
                (run_id, node, task_id, logical_attempt),
            ).fetchone()
            if conflict is not None:
                raise BillingStateError("logical call slot already belongs to another call")
            committed = connection.execute(
                """SELECT COALESCE(SUM(
                    CASE WHEN state = 'completed' THEN COALESCE(actual_micro_cny, 0)
                         ELSE reserved_micro_cny END), 0) AS total
                   FROM calls WHERE activity_id = ?""",
                (self.activity_id,),
            ).fetchone()["total"]
            if committed + projected > self.cap_micro_cny:
                raise BudgetExceeded("next call would exceed estimated_cost_cap")
            now = _timestamp()
            connection.execute(
                """INSERT INTO calls (
                    call_id, request_sha256, activity_id, run_id, node, task_id, logical_attempt,
                    state, reserved_micro_cny, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'reserved', ?, ?, ?)""",
                (
                    call_id,
                    request_sha256,
                    self.activity_id,
                    run_id,
                    node,
                    task_id,
                    logical_attempt,
                    projected,
                    now,
                    now,
                ),
            )
            connection.commit()
            return projected
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def mark_sent(self, call_id: str) -> None:
        self._transition(call_id, CallState.RESERVED, CallState.SENT, increment_transport=True)

    def mark_retryable_not_billed(self, call_id: str) -> None:
        self._transition(call_id, CallState.SENT, CallState.RESERVED)

    def mark_billing_uncertain(self, call_id: str) -> None:
        self._transition(call_id, CallState.SENT, CallState.BILLING_UNCERTAIN)

    def _transition(
        self,
        call_id: str,
        expected: CallState,
        target: CallState,
        *,
        increment_transport: bool = False,
    ) -> None:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT state FROM calls WHERE call_id = ?", (call_id,)
            ).fetchone()
            if row is None:
                raise KeyError(call_id)
            if row["state"] != expected.value:
                raise BillingStateError(
                    f"cannot transition {row['state']} call to {target.value}"
                )
            now = _timestamp()
            if increment_transport:
                connection.execute(
                    "UPDATE calls SET state = ?, transport_attempts = transport_attempts + 1, "
                    "updated_at = ? WHERE call_id = ?",
                    (target.value, now, call_id),
                )
            else:
                connection.execute(
                    "UPDATE calls SET state = ?, updated_at = ? WHERE call_id = ?",
                    (target.value, now, call_id),
                )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def complete_call(
        self,
        call_id: str,
        *,
        request_sha256: str,
        payload: dict[str, object],
        usage: Usage,
        usage_source: Literal["provider", "missing"],
        requested_alias: str,
        response_model_id_raw: str,
        identity_verified: bool,
    ) -> dict[str, object]:
        if usage_source == "provider" and not usage.complete:
            raise ValueError("provider usage source requires complete usage")
        if usage_source == "missing" and usage.complete:
            raise ValueError("missing usage source requires incomplete usage")
        connection = self._connect()
        over_budget = False
        stored_payload: dict[str, object]
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM calls WHERE call_id = ?", (call_id,)).fetchone()
            if row is None:
                raise KeyError(call_id)
            self._check_fingerprint(row, request_sha256)
            state = CallState(row["state"])
            if state in {CallState.COMPLETED, CallState.USAGE_MISSING}:
                stored_payload = json.loads(row["payload_json"])
                connection.commit()
                return stored_payload
            stored_payload = redact_payload(payload)  # type: ignore[assignment]
            actual = self.pricing.estimate_micro_cny(usage) if usage.complete else None
            if actual is not None and actual > row["reserved_micro_cny"]:
                over_budget = True
            now = _timestamp()
            connection.execute(
                """UPDATE calls SET state = ?, actual_micro_cny = ?, payload_json = ?,
                    usage_json = ?, usage_source = ?, requested_alias = ?,
                    response_model_id_raw = ?, identity_verified = ?,
                    updated_at = ? WHERE call_id = ?""",
                (
                    CallState.COMPLETED.value if usage.complete else CallState.USAGE_MISSING.value,
                    actual,
                    json.dumps(stored_payload, ensure_ascii=False, separators=(",", ":")),
                    usage.model_dump_json(),
                    usage_source,
                    requested_alias,
                    response_model_id_raw,
                    int(identity_verified),
                    now,
                    call_id,
                ),
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        if over_budget:
            raise BudgetExceeded("provider response exceeded reserved budget")
        return stored_payload

    def get_completed(self, call_id: str) -> dict[str, object] | None:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM calls WHERE call_id = ?", (call_id,)).fetchone()
            if row is None or row["state"] != CallState.COMPLETED.value:
                connection.commit()
                return None
            connection.execute(
                "UPDATE calls SET cache_hits = cache_hits + 1, updated_at = ? WHERE call_id = ?",
                (_timestamp(), call_id),
            )
            connection.commit()
            return json.loads(row["payload_json"])
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def resume_decision(self, call_id: str, *, request_sha256: str) -> ResumeDecision:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM calls WHERE call_id = ?", (call_id,)).fetchone()
            if row is None:
                connection.commit()
                return ResumeDecision(action="send", call_id=call_id)
            self._check_fingerprint(row, request_sha256)
            state = CallState(row["state"])
            if state == CallState.COMPLETED:
                decision = ResumeDecision(
                    action="reuse", call_id=call_id, payload=json.loads(row["payload_json"])
                )
            elif state == CallState.USAGE_MISSING:
                decision = ResumeDecision(
                    action="reuse_and_stop",
                    call_id=call_id,
                    payload=json.loads(row["payload_json"]),
                )
            elif state == CallState.RESERVED:
                decision = ResumeDecision(action="send", call_id=call_id)
            else:
                raise BillingStateError(f"{state.value} call has unknown billing")
            connection.commit()
            return decision
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def summarize_run(self, run_id: str) -> RunCallSummary:
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT * FROM calls WHERE activity_id = ? AND run_id = ? "
                "ORDER BY created_at, call_id",
                (self.activity_id, run_id),
            ).fetchall()
        finally:
            connection.close()
        input_tokens = output_tokens = total_tokens = 0
        all_complete = True
        known_actual = committed = 0
        actual_exact = True
        fresh = 0
        cache_hits = transport = 0
        aliases: list[str] = []
        model_ids: list[str] = []
        usage_sources: list[Literal["provider", "missing"]] = []
        identity_values: list[bool] = []
        billing_uncertain = False
        for row in rows:
            fresh += 1
            cache_hits += row["cache_hits"]
            transport += row["transport_attempts"]
            if row["requested_alias"] and row["requested_alias"] not in aliases:
                aliases.append(row["requested_alias"])
            if row["response_model_id_raw"] and row["response_model_id_raw"] not in model_ids:
                model_ids.append(row["response_model_id_raw"])
            if row["usage_source"]:
                usage_sources.append(row["usage_source"])
            if row["identity_verified"] is not None:
                identity_values.append(bool(row["identity_verified"]))
            if row["actual_micro_cny"] is not None:
                known_actual += row["actual_micro_cny"]
                committed += row["actual_micro_cny"]
            else:
                committed += row["reserved_micro_cny"]
            if row["state"] != CallState.COMPLETED.value:
                actual_exact = False
            if row["state"] == CallState.BILLING_UNCERTAIN.value:
                billing_uncertain = True
            if row["usage_json"]:
                usage = Usage.model_validate_json(row["usage_json"])
                input_tokens += usage.input_tokens
                output_tokens += usage.output_tokens
                total_tokens += usage.total_tokens
                all_complete = all_complete and usage.complete
        usage = Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            complete=all_complete,
        )
        return RunCallSummary(
            call_ids=[row["call_id"] for row in rows],
            usage=usage,
            actual_cost_micro_cny=known_actual if actual_exact else None,
            known_actual_cost_micro_cny=known_actual,
            committed_cost_micro_cny=committed,
            cost_is_lower_bound=not actual_exact,
            fresh_call_count=fresh,
            cache_hit_count=cache_hits,
            transport_attempts=transport,
            requested_aliases=aliases,
            response_model_ids_raw=model_ids,
            usage_sources=usage_sources,
            identity_verified=bool(identity_values) and all(identity_values),
            billing_uncertain=billing_uncertain,
        )
