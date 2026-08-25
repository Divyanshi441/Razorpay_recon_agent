from __future__ import annotations

import argparse
import io
import json
import os
import random
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


AMOUNT_TOLERANCE = 1.00
DEFAULT_RECORDS = 80
DEFAULT_SEED = 42


@dataclass
class Explanation:
    likely_cause: str
    recommended_action: str
    confidence: str
    source: str


RULE_BASED_EXPLANATIONS = {
    "EXCEPTION_AMOUNT_MISMATCH": Explanation(
        likely_cause=(
            "The bank payout amount does not equal the expected net settlement amount. "
            "Possible reasons include fee/tax differences, partial settlement, chargeback offset, "
            "manual adjustment, or a wrong reference in the payout file."
        ),
        recommended_action=(
            "Compare gross amount, fees, taxes, and expected net against the bank row. "
            "Check for refunds, chargebacks, manual adjustments, or incorrect mapping. "
            "If still unresolved, escalate to finance ops with payment_id and UTR."
        ),
        confidence="medium",
        source="rule_based",
    ),
    "EXCEPTION_MISSING_SETTLEMENT": Explanation(
        likely_cause=(
            "The payment exists in the gateway file but no matching bank payout row was found. "
            "This usually means delayed settlement, incomplete bank export, payout hold, or payout failure."
        ),
        recommended_action=(
            "Check whether the bank export covers the expected settlement date, and verify if the payout is pending, held, or failed. "
            "Escalate only after the allowed settlement lag is exceeded."
        ),
        confidence="high",
        source="rule_based",
    ),
    "EXCEPTION_DUPLICATE_GATEWAY": Explanation(
        likely_cause=(
            "The same payment_id appears more than once in the gateway input. "
            "This usually comes from duplicate ingestion, replayed jobs, or repeated exports."
        ),
        recommended_action=(
            "Deduplicate the gateway data on payment_id, identify the upstream source of duplicate rows, "
            "and keep one canonical record with an audit note."
        ),
        confidence="high",
        source="rule_based",
    ),
    "EXCEPTION_DUPLICATE_BANK": Explanation(
        likely_cause=(
            "The bank file contains duplicate-looking settlement rows for the same payment reference. "
            "It is unsafe to auto-match because this could be a duplicate export row or an actual duplicate credit."
        ),
        recommended_action=(
            "Verify the UTR and bank statement, confirm whether one row is an export duplicate, "
            "and only then mark a single row as the true settlement."
        ),
        confidence="high",
        source="rule_based",
    ),
    "EXCEPTION_ORPHAN_BANK": Explanation(
        likely_cause=(
            "A bank payout row exists, but no corresponding payment was found in the gateway input. "
            "This may be a bad reference, manual adjustment, legacy payout, or missing gateway data."
        ),
        recommended_action=(
            "Search other systems using the bank reference/UTR, confirm whether the gateway extract is incomplete, "
            "and keep this as a bank-side exception until resolved."
        ),
        confidence="medium",
        source="rule_based",
    ),
}


CASE_TO_STATUS = {
    "EXACT_MATCH": "MATCHED_EXACT",
    "DELAYED_MATCH": "MATCHED_DELAYED",
    "SPLIT_SETTLEMENT": "MATCHED_SPLIT",
    "AMOUNT_MISMATCH": "EXCEPTION_AMOUNT_MISMATCH",
    "MISSING_SETTLEMENT": "EXCEPTION_MISSING_SETTLEMENT",
    "DUPLICATE_GATEWAY": "EXCEPTION_DUPLICATE_GATEWAY",
    "DUPLICATE_BANK": "EXCEPTION_DUPLICATE_BANK",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconcile payment gateway records against bank settlement records."
    )
    parser.add_argument("--gateway-csv", type=str, default="", help="Path to gateway payment CSV")
    parser.add_argument("--bank-csv", type=str, default="", help="Path to bank settlement CSV")
    parser.add_argument("--truth-csv", type=str, default="", help="Optional path to synthetic truth CSV")
    parser.add_argument("--out-dir", type=str, default="demo_output", help="Output directory")
    parser.add_argument("--records", type=int, default=DEFAULT_RECORDS, help="Synthetic record count")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument(
        "--generate-demo-data",
        action="store_true",
        help="Generate synthetic demo data before reconciliation",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use an LLM for exception explanations if OPENAI_API_KEY is set",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gpt-4.1-mini",
        help="LLM model name for explanations",
    )
    return parser.parse_args()


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def round2(x: float) -> float:
    return round(float(x), 2)


def choose_case(rng: random.Random) -> str:
    weighted_cases = [
        ("EXACT_MATCH", 0.46),
        ("DELAYED_MATCH", 0.14),
        ("SPLIT_SETTLEMENT", 0.12),
        ("AMOUNT_MISMATCH", 0.10),
        ("MISSING_SETTLEMENT", 0.08),
        ("DUPLICATE_GATEWAY", 0.05),
        ("DUPLICATE_BANK", 0.05),
    ]
    roll = rng.random()
    cumulative = 0.0
    for case, weight in weighted_cases:
        cumulative += weight
        if roll <= cumulative:
            return case
    return "EXACT_MATCH"


def generate_synthetic_data(num_records: int, seed: int, out_dir: Path) -> Tuple[Path, Path, Path]:
    rng = random.Random(seed)
    base_time = datetime(2026, 8, 1, 9, 0, 0)

    gateway_rows: List[Dict] = []
    bank_rows: List[Dict] = []
    truth_rows: List[Dict] = []

    gateway_counter = 1
    bank_counter = 1

    for i in range(1, num_records + 1):
        payment_id = f"pay_{i:05d}"
        order_id = f"order_{100000 + i}"
        customer_id = f"cust_{rng.randint(1000, 9999)}"

        gross_amount = round2(rng.randint(400, 9000) + rng.random())
        fee_amount = round2(gross_amount * rng.uniform(0.015, 0.025))
        tax_amount = round2(fee_amount * 0.18)
        expected_net_amount = round2(gross_amount - fee_amount - tax_amount)

        payment_time = base_time + timedelta(minutes=rng.randint(0, 60 * 24 * 12))
        case = choose_case(rng)

        base_gateway_row = {
            "gateway_row_id": f"gw_{gateway_counter:05d}",
            "payment_id": payment_id,
            "order_id": order_id,
            "customer_id": customer_id,
            "gross_amount": gross_amount,
            "fee_amount": fee_amount,
            "tax_amount": tax_amount,
            "expected_net_amount": expected_net_amount,
            "currency": "INR",
            "payment_time": payment_time.isoformat(),
            "status": "captured",
        }
        gateway_rows.append(base_gateway_row)
        gateway_counter += 1

        truth_rows.append(
            {
                "payment_id": payment_id,
                "truth_case": case,
                "ground_truth_status": CASE_TO_STATUS[case],
            }
        )

        if case == "EXACT_MATCH":
            settlement_time = payment_time + timedelta(hours=rng.randint(6, 24))
            bank_rows.append(
                {
                    "bank_row_id": f"bk_{bank_counter:05d}",
                    "payment_id_hint": payment_id,
                    "order_id_hint": order_id,
                    "utr": f"UTR{rng.randint(10**10, 10**11 - 1)}",
                    "settled_amount": expected_net_amount,
                    "currency": "INR",
                    "settlement_time": settlement_time.isoformat(),
                    "narrative": f"RAZORPAY SETTLEMENT {payment_id}",
                }
            )
            bank_counter += 1

        elif case == "DELAYED_MATCH":
            settlement_time = payment_time + timedelta(days=rng.randint(2, 5), hours=rng.randint(1, 8))
            bank_rows.append(
                {
                    "bank_row_id": f"bk_{bank_counter:05d}",
                    "payment_id_hint": payment_id,
                    "order_id_hint": order_id,
                    "utr": f"UTR{rng.randint(10**10, 10**11 - 1)}",
                    "settled_amount": expected_net_amount,
                    "currency": "INR",
                    "settlement_time": settlement_time.isoformat(),
                    "narrative": f"DELAYED PAYOUT {payment_id}",
                }
            )
            bank_counter += 1

        elif case == "SPLIT_SETTLEMENT":
            first_part = round2(expected_net_amount * rng.uniform(0.30, 0.70))
            second_part = round2(expected_net_amount - first_part)
            settlement_day = payment_time + timedelta(days=rng.randint(1, 3))
            for amount, hours in [(first_part, 2), (second_part, 6)]:
                bank_rows.append(
                    {
                        "bank_row_id": f"bk_{bank_counter:05d}",
                        "payment_id_hint": payment_id,
                        "order_id_hint": order_id,
                        "utr": f"UTR{rng.randint(10**10, 10**11 - 1)}",
                        "settled_amount": amount,
                        "currency": "INR",
                        "settlement_time": (settlement_day + timedelta(hours=hours)).isoformat(),
                        "narrative": f"SPLIT PAYOUT {payment_id}",
                    }
                )
                bank_counter += 1

        elif case == "AMOUNT_MISMATCH":
            delta = round2(rng.uniform(10, 250))
            delta = delta if rng.random() > 0.5 else -delta
            settlement_time = payment_time + timedelta(days=1)
            bank_rows.append(
                {
                    "bank_row_id": f"bk_{bank_counter:05d}",
                    "payment_id_hint": payment_id,
                    "order_id_hint": order_id,
                    "utr": f"UTR{rng.randint(10**10, 10**11 - 1)}",
                    "settled_amount": round2(max(1.0, expected_net_amount + delta)),
                    "currency": "INR",
                    "settlement_time": settlement_time.isoformat(),
                    "narrative": f"ADJUSTED PAYOUT {payment_id}",
                }
            )
            bank_counter += 1

        elif case == "MISSING_SETTLEMENT":
            pass

        elif case == "DUPLICATE_GATEWAY":
            dup_row = dict(base_gateway_row)
            dup_row["gateway_row_id"] = f"gw_{gateway_counter:05d}"
            gateway_rows.append(dup_row)
            gateway_counter += 1

            settlement_time = payment_time + timedelta(hours=12)
            bank_rows.append(
                {
                    "bank_row_id": f"bk_{bank_counter:05d}",
                    "payment_id_hint": payment_id,
                    "order_id_hint": order_id,
                    "utr": f"UTR{rng.randint(10**10, 10**11 - 1)}",
                    "settled_amount": expected_net_amount,
                    "currency": "INR",
                    "settlement_time": settlement_time.isoformat(),
                    "narrative": f"NORMAL PAYOUT {payment_id}",
                }
            )
            bank_counter += 1

        elif case == "DUPLICATE_BANK":
            settlement_time = payment_time + timedelta(hours=15)
            dup_template = {
                "payment_id_hint": payment_id,
                "order_id_hint": order_id,
                "utr": f"UTR{rng.randint(10**10, 10**11 - 1)}",
                "settled_amount": expected_net_amount,
                "currency": "INR",
                "settlement_time": settlement_time.isoformat(),
                "narrative": f"PAYOUT DUPLICATE {payment_id}",
            }
            for _ in range(2):
                row = dict(dup_template)
                row["bank_row_id"] = f"bk_{bank_counter:05d}"
                bank_rows.append(row)
                bank_counter += 1

    # Add a few bank-only orphan rows so the exception report is honest.
    for j in range(1, 4):
        orphan_time = base_time + timedelta(days=10, hours=j)
        bank_rows.append(
            {
                "bank_row_id": f"bk_{bank_counter:05d}",
                "payment_id_hint": f"unknown_pay_{j:03d}",
                "order_id_hint": f"unknown_order_{j:03d}",
                "utr": f"UTR{rng.randint(10**10, 10**11 - 1)}",
                "settled_amount": round2(rng.randint(500, 7000) + rng.random()),
                "currency": "INR",
                "settlement_time": orphan_time.isoformat(),
                "narrative": f"MANUAL CREDIT unknown_pay_{j:03d}",
            }
        )
        bank_counter += 1

    gateway_df = pd.DataFrame(gateway_rows).sort_values(
        ["payment_time", "payment_id", "gateway_row_id"]
    ).reset_index(drop=True)
    bank_df = pd.DataFrame(bank_rows).sort_values(
        ["settlement_time", "payment_id_hint", "bank_row_id"]
    ).reset_index(drop=True)
    truth_df = pd.DataFrame(truth_rows).sort_values("payment_id").reset_index(drop=True)

    gateway_path = out_dir / "synthetic_gateway_payments.csv"
    bank_path = out_dir / "synthetic_bank_settlements.csv"
    truth_path = out_dir / "synthetic_truth.csv"

    gateway_df.to_csv(gateway_path, index=False)
    bank_df.to_csv(bank_path, index=False)
    truth_df.to_csv(truth_path, index=False)

    return gateway_path, bank_path, truth_path


def try_llm_explanation(exception_payload: Dict, model: str) -> Optional[Explanation]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None

    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=api_key,
            base_url=os.getenv("OPENAI_BASE_URL") or None,
        )

        prompt = f"""
You are helping explain a finance reconciliation exception to a junior developer.
Return ONLY valid JSON with keys:
- likely_cause
- recommended_action
- confidence

Keep the response practical and concise.

Exception payload:
{json.dumps(exception_payload, indent=2, default=str)}
""".strip()

        response = client.responses.create(model=model, input=prompt)
        raw = (response.output_text or "").strip()
        data = json.loads(raw)

        return Explanation(
            likely_cause=str(data.get("likely_cause", "")),
            recommended_action=str(data.get("recommended_action", "")),
            confidence=str(data.get("confidence", "medium")),
            source="llm",
        )
    except Exception:
        return None


def explain_exception(exception_type: str, payload: Dict, use_llm: bool, llm_model: str) -> Explanation:
    if use_llm:
        llm_explanation = try_llm_explanation(payload, llm_model)
        if llm_explanation:
            return llm_explanation

    return RULE_BASED_EXPLANATIONS.get(
        exception_type,
        Explanation(
            likely_cause="The record could not be matched cleanly and needs manual review.",
            recommended_action="Inspect source rows and update reconciliation rules if this pattern is expected.",
            confidence="low",
            source="rule_based",
        ),
    )


GATEWAY_REQUIRED_COLUMNS = {
    "gateway_row_id", "payment_id", "order_id", "customer_id",
    "gross_amount", "fee_amount", "tax_amount", "expected_net_amount",
    "currency", "payment_time", "status",
}

BANK_REQUIRED_COLUMNS = {
    "bank_row_id", "payment_id_hint", "order_id_hint", "utr",
    "settled_amount", "currency", "settlement_time", "narrative",
}


def validate_input_data(gateway: pd.DataFrame, bank: pd.DataFrame) -> None:
    """Validate the uploaded CSV schemas with beginner-friendly errors."""
    if gateway.empty:
        raise ValueError("Gateway CSV is empty. Upload a CSV containing payment records.")
    if bank.empty:
        raise ValueError("Bank CSV is empty. Upload a CSV containing settlement records.")

    missing_gateway = sorted(GATEWAY_REQUIRED_COLUMNS - set(gateway.columns))
    missing_bank = sorted(BANK_REQUIRED_COLUMNS - set(bank.columns))
    errors = []

    if missing_gateway:
        errors.append("Gateway CSV is missing: " + ", ".join(missing_gateway))
    if missing_bank:
        errors.append("Bank CSV is missing: " + ", ".join(missing_bank))

    if errors:
        raise ValueError("\n".join(errors))


def reconcile_csv_files(
    gateway_csv,
    bank_csv,
    truth_csv=None,
    use_llm: bool = False,
    llm_model: str = "gpt-4.1-mini",
    amount_tolerance: float = AMOUNT_TOLERANCE,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, Dict]:
    """Run the full reconciliation pipeline on paths or file-like objects."""
    gateway, bank, truth = load_data(gateway_csv, bank_csv, truth_csv)
    results, exceptions, summary = reconcile_data(
        gateway=gateway,
        bank=bank,
        use_llm=use_llm,
        llm_model=llm_model,
        amount_tolerance=amount_tolerance,
    )
    evaluation = evaluate_against_truth(results, truth)
    return results, exceptions, summary, evaluation


def load_data(
    gateway_csv: Path,
    bank_csv: Path,
    truth_csv: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    gateway = pd.read_csv(gateway_csv)
    bank = pd.read_csv(bank_csv)
    truth = pd.read_csv(truth_csv) if truth_csv and Path(truth_csv).exists() else None
    validate_input_data(gateway, bank)

    gateway["payment_time"] = pd.to_datetime(gateway["payment_time"])
    bank["settlement_time"] = pd.to_datetime(bank["settlement_time"])

    for col in ["gross_amount", "fee_amount", "tax_amount", "expected_net_amount"]:
        gateway[col] = gateway[col].astype(float).round(2)

    bank["settled_amount"] = bank["settled_amount"].astype(float).round(2)

    return gateway, bank, truth


def approx_equal(a: float, b: float, tolerance: float = AMOUNT_TOLERANCE) -> bool:
    return abs(float(a) - float(b)) <= tolerance


def reconcile_data(
    gateway: pd.DataFrame,
    bank: pd.DataFrame,
    use_llm: bool,
    llm_model: str,
    amount_tolerance: float = AMOUNT_TOLERANCE,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    started = time.perf_counter()

    gateway_duplicates = set(
        gateway[gateway.duplicated(subset=["payment_id"], keep=False)]["payment_id"].unique()
    )

    bank_duplicate_mask = bank.duplicated(
        subset=["payment_id_hint", "settled_amount", "settlement_time"],
        keep=False,
    )
    bank_duplicate_payment_ids = set(bank.loc[bank_duplicate_mask, "payment_id_hint"].dropna().unique())

    results: List[Dict] = []
    exceptions: List[Dict] = []

    grouped_gateway = gateway.sort_values(["payment_id", "payment_time"]).groupby("payment_id", sort=True)
    known_gateway_ids = set(grouped_gateway.groups.keys())

    for payment_id, group in grouped_gateway:
        canonical = group.iloc[0]
        expected_net = round2(canonical["expected_net_amount"])
        payment_time = canonical["payment_time"]

        candidates = bank.loc[bank["payment_id_hint"] == payment_id].copy()
        candidate_ids = candidates["bank_row_id"].tolist()
        candidate_total = round2(candidates["settled_amount"].sum()) if not candidates.empty else 0.0

        predicted_status = ""
        matched_bank_ids: List[str] = []
        matched_amount = 0.0
        settlement_lag_days: Optional[int] = None
        explanation: Optional[Explanation] = None

        if payment_id in gateway_duplicates:
            predicted_status = "EXCEPTION_DUPLICATE_GATEWAY"

        elif payment_id in bank_duplicate_payment_ids:
            predicted_status = "EXCEPTION_DUPLICATE_BANK"

        elif candidates.empty:
            predicted_status = "EXCEPTION_MISSING_SETTLEMENT"

        else:
            exact_rows = candidates[
                candidates["settled_amount"].apply(lambda x: approx_equal(x, expected_net, amount_tolerance))
            ]

            if len(exact_rows) == 1:
                selected = exact_rows.iloc[0]
                matched_bank_ids = [selected["bank_row_id"]]
                matched_amount = round2(selected["settled_amount"])
                settlement_lag_days = int((selected["settlement_time"] - payment_time).days)
                predicted_status = "MATCHED_DELAYED" if settlement_lag_days > 1 else "MATCHED_EXACT"

            elif len(candidates) > 1 and approx_equal(candidates["settled_amount"].sum(), expected_net, amount_tolerance):
                matched_bank_ids = candidates["bank_row_id"].tolist()
                matched_amount = round2(candidates["settled_amount"].sum())
                settlement_lag_days = int((candidates["settlement_time"].max() - payment_time).days)
                predicted_status = "MATCHED_SPLIT"

            else:
                predicted_status = "EXCEPTION_AMOUNT_MISMATCH"

        if predicted_status.startswith("EXCEPTION_"):
            payload = {
                "payment_id": payment_id,
                "order_id": canonical["order_id"],
                "expected_net_amount": expected_net,
                "gateway_rows_for_payment": group.to_dict(orient="records"),
                "candidate_bank_rows": candidates.to_dict(orient="records"),
            }
            explanation = explain_exception(predicted_status, payload, use_llm, llm_model)

            exceptions.append(
                {
                    "record_side": "gateway",
                    "payment_id": payment_id,
                    "bank_row_ids": ";".join(candidate_ids),
                    "exception_type": predicted_status,
                    "likely_cause": explanation.likely_cause,
                    "recommended_action": explanation.recommended_action,
                    "confidence": explanation.confidence,
                    "explanation_source": explanation.source,
                    "expected_net_amount": expected_net,
                    "candidate_bank_total": candidate_total,
                }
            )

        results.append(
            {
                "payment_id": payment_id,
                "order_id": canonical["order_id"],
                "customer_id": canonical["customer_id"],
                "gross_amount": round2(canonical["gross_amount"]),
                "fee_amount": round2(canonical["fee_amount"]),
                "tax_amount": round2(canonical["tax_amount"]),
                "expected_net_amount": expected_net,
                "payment_time": canonical["payment_time"].isoformat(),
                "candidate_bank_row_ids": ";".join(candidate_ids),
                "matched_bank_row_ids": ";".join(matched_bank_ids),
                "candidate_bank_total": candidate_total,
                "matched_amount": matched_amount,
                "settlement_lag_days": settlement_lag_days,
                "predicted_status": predicted_status,
                "likely_cause": explanation.likely_cause if explanation else "",
                "recommended_action": explanation.recommended_action if explanation else "",
                "explanation_source": explanation.source if explanation else "",
            }
        )

    # Bank-side orphan rows
    orphan_bank_rows = bank.loc[~bank["payment_id_hint"].isin(known_gateway_ids)].copy()
    for _, row in orphan_bank_rows.iterrows():
        payload = row.to_dict()
        explanation = explain_exception("EXCEPTION_ORPHAN_BANK", payload, use_llm, llm_model)
        exceptions.append(
            {
                "record_side": "bank",
                "payment_id": row["payment_id_hint"],
                "bank_row_ids": row["bank_row_id"],
                "exception_type": "EXCEPTION_ORPHAN_BANK",
                "likely_cause": explanation.likely_cause,
                "recommended_action": explanation.recommended_action,
                "confidence": explanation.confidence,
                "explanation_source": explanation.source,
                "expected_net_amount": "",
                "candidate_bank_total": round2(row["settled_amount"]),
            }
        )

    results_df = pd.DataFrame(results)
    exceptions_df = pd.DataFrame(exceptions)

    matched_mask = results_df["predicted_status"].str.startswith("MATCHED_")
    elapsed = max(time.perf_counter() - started, 1e-9)

    exception_breakdown = []
    bank_orphans = 0
    if not exceptions_df.empty:
        exception_breakdown = Counter(exceptions_df["exception_type"]).most_common()
        bank_orphans = int((exceptions_df["exception_type"] == "EXCEPTION_ORPHAN_BANK").sum())

    summary = {
        "total_gateway_rows": int(len(gateway)),
        "unique_gateway_payments": int(gateway["payment_id"].nunique()),
        "bank_rows": int(len(bank)),
        "matched_payments": int(matched_mask.sum()),
        "match_rate": round(float(matched_mask.mean()) if len(results_df) else 0.0, 4),
        "gateway_side_exceptions": int((~matched_mask).sum()),
        "bank_side_orphan_exceptions": bank_orphans,
        "throughput_records_per_second": round(float(len(results_df) / elapsed), 2),
        "exception_breakdown": exception_breakdown,
    }

    return results_df, exceptions_df, summary


def evaluate_against_truth(results_df: pd.DataFrame, truth_df: Optional[pd.DataFrame]) -> Dict:
    if truth_df is None or truth_df.empty:
        return {
            "classification_accuracy": None,
            "evaluated_records": 0,
            "correct_records": 0,
            "confusion_matrix": [],
        }

    scored = results_df.merge(
        truth_df[["payment_id", "ground_truth_status", "truth_case"]],
        on="payment_id",
        how="left",
    )
    scored["is_correct"] = scored["predicted_status"] == scored["ground_truth_status"]

    confusion = (
        scored.groupby(["ground_truth_status", "predicted_status"])
        .size()
        .reset_index(name="count")
        .to_dict(orient="records")
    )

    return {
        "classification_accuracy": round(float(scored["is_correct"].mean()), 4),
        "evaluated_records": int(len(scored)),
        "correct_records": int(scored["is_correct"].sum()),
        "confusion_matrix": confusion,
        "scored_results": scored,
    }


def save_outputs(
    out_dir: Path,
    results_df: pd.DataFrame,
    exceptions_df: pd.DataFrame,
    summary: Dict,
    evaluation: Dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    results_path = out_dir / "reconciliation_results.csv"
    exceptions_path = out_dir / "exceptions_report.csv"
    summary_path = out_dir / "summary.json"

    results_to_save = results_df.copy()

    if isinstance(evaluation.get("scored_results"), pd.DataFrame):
        extra_cols = evaluation["scored_results"][["payment_id", "ground_truth_status", "truth_case", "is_correct"]]
        results_to_save = results_to_save.merge(extra_cols, on="payment_id", how="left")

    results_to_save.to_csv(results_path, index=False)
    exceptions_df.to_csv(exceptions_path, index=False)

    summary_payload = {
        **summary,
        "classification_accuracy": evaluation.get("classification_accuracy"),
        "evaluated_records": evaluation.get("evaluated_records"),
        "correct_records": evaluation.get("correct_records"),
        "confusion_matrix": evaluation.get("confusion_matrix"),
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)


def print_summary(summary: Dict, evaluation: Dict, out_dir: Path) -> None:
    print("\n=== Reconciliation Summary ===")
    print(f"Unique gateway payments : {summary['unique_gateway_payments']}")
    print(f"Gateway rows processed  : {summary['total_gateway_rows']}")
    print(f"Bank rows processed     : {summary['bank_rows']}")
    print(f"Matched payments        : {summary['matched_payments']}")
    print(f"Match rate              : {summary['match_rate']:.2%}")
    print(f"Throughput              : {summary['throughput_records_per_second']} records/sec")

    if evaluation.get("classification_accuracy") is not None:
        print(f"Classification accuracy : {evaluation['classification_accuracy']:.2%}")
        print(f"Correct / evaluated     : {evaluation['correct_records']} / {evaluation['evaluated_records']}")

    if summary.get("exception_breakdown"):
        print("\nException breakdown:")
        for exception_type, count in summary["exception_breakdown"]:
            print(f"  - {exception_type}: {count}")

    print(f"\nOutput folder: {out_dir.resolve()}")


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)

    gateway_csv = Path(args.gateway_csv) if args.gateway_csv else None
    bank_csv = Path(args.bank_csv) if args.bank_csv else None
    truth_csv = Path(args.truth_csv) if args.truth_csv else None

    if args.generate_demo_data or not (gateway_csv and bank_csv):
        gateway_csv, bank_csv, truth_csv = generate_synthetic_data(
            num_records=max(50, args.records),
            seed=args.seed,
            out_dir=out_dir,
        )
        print("Generated synthetic demo files:")
        print(f"  - {gateway_csv}")
        print(f"  - {bank_csv}")
        print(f"  - {truth_csv}")
    else:
        if not gateway_csv.exists():
            raise FileNotFoundError(f"Gateway CSV not found: {gateway_csv}")
        if not bank_csv.exists():
            raise FileNotFoundError(f"Bank CSV not found: {bank_csv}")

    gateway_df, bank_df, truth_df = load_data(gateway_csv, bank_csv, truth_csv)

    results_df, exceptions_df, summary = reconcile_data(
        gateway=gateway_df,
        bank=bank_df,
        use_llm=args.use_llm,
        llm_model=args.llm_model,
        amount_tolerance=AMOUNT_TOLERANCE,
    )

    evaluation = evaluate_against_truth(results_df, truth_df)
    save_outputs(out_dir, results_df, exceptions_df, summary, evaluation)
    print_summary(summary, evaluation, out_dir)

# ---------------------------------------------------------------------------
# Streamlit web UI
# ---------------------------------------------------------------------------

import io
import json
import os

import pandas as pd
import streamlit as st

# from app import AMOUNT_TOLERANCE, reconcile_csv_files  
from app import *


def run_streamlit_app() -> None:
    st.set_page_config(
        page_title="AI Finance Controller | Razorpay Buildathon",
        page_icon="💳",
        layout="wide",
    )

    st.title("💳 AI Finance Controller")
    st.caption("Track 04 • Run the books and the cash position")

    st.markdown(
        "Upload a **gateway payments CSV** and a **bank settlements CSV**, then click "
        "**Reconcile**. The controller matches records, reports the match rate, and "
        "flags exceptions it could not resolve automatically."
    )

    with st.sidebar:
        st.header("Reconciliation settings")

        tolerance = st.number_input(
            "Amount tolerance (₹)",
            min_value=0.0,
            max_value=1000.0,
            value=float(AMOUNT_TOLERANCE),
            step=0.50,
            help="Amounts within this difference are treated as equal.",
        )

        use_llm = st.toggle(
            "Use AI explanations",
            value=False,
            help="Optional: uses OPENAI_API_KEY to explain exceptions. Matching itself remains deterministic.",
        )
        llm_model = st.text_input(
            "AI model",
            value="gpt-4.1-mini",
            disabled=not use_llm,
        )

        if use_llm and not os.getenv("OPENAI_API_KEY"):
            st.warning("OPENAI_API_KEY is not set. Rule-based explanations will be used.")

        st.divider()
        st.subheader("CSV contract")
        st.markdown(
            "**Gateway:** payment_id, order_id, expected_net_amount, payment_time, "
            "plus amount/customer fields.\n\n"
            "**Bank:** payment_id_hint, bank_row_id, utr, settled_amount, settlement_time."
        )

    left, right = st.columns(2)

    with left:
        gateway_file = st.file_uploader(
            "1️⃣ Upload gateway payments CSV",
            type=["csv"],
            key="gateway_file",
        )

    with right:
        bank_file = st.file_uploader(
            "2️⃣ Upload bank settlements CSV",
            type=["csv"],
            key="bank_file",
        )

    # Preview inputs before the user runs the reconciliation.
    if gateway_file or bank_file:
        st.subheader("Input preview")
        p1, p2 = st.columns(2)

        if gateway_file:
            gateway_preview = pd.read_csv(gateway_file)
            with p1:
                st.write(f"**Gateway:** {len(gateway_preview):,} rows")
                st.dataframe(
                    gateway_preview.head(5),
                    use_container_width=True,
                    hide_index=True,
                )
            gateway_file.seek(0)

        if bank_file:
            bank_preview = pd.read_csv(bank_file)
            with p2:
                st.write(f"**Bank:** {len(bank_preview):,} rows")
                st.dataframe(
                    bank_preview.head(5),
                    use_container_width=True,
                    hide_index=True,
                )
            bank_file.seek(0)

    reconcile_clicked = st.button(
        "🔎 Reconcile",
        type="primary",
        use_container_width=True,
        disabled=not (gateway_file and bank_file),
    )

    if reconcile_clicked:
        try:
            with st.spinner("Reconciling records..."):
                results, exceptions, summary, evaluation = reconcile_csv_files(
                    io.BytesIO(gateway_file.getvalue()),
                    io.BytesIO(bank_file.getvalue()),
                    use_llm=use_llm,
                    llm_model=llm_model,
                    amount_tolerance=float(tolerance),
                )

            st.session_state["reconciliation"] = {
                "results": results,
                "exceptions": exceptions,
                "summary": summary,
                "evaluation": evaluation,
            }
        except Exception as exc:
            st.error(f"Reconciliation failed: {exc}")
            st.stop()

    data = st.session_state.get("reconciliation")

    if data:
        results = data["results"]
        exceptions = data["exceptions"]
        summary = data["summary"]
        evaluation = data["evaluation"]

        st.divider()
        st.subheader("Reconciliation result")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Match rate", f"{summary['match_rate']:.1%}")
        c2.metric("Matched payments", f"{summary['matched_payments']:,}")
        c3.metric(
            "Unresolved exceptions",
            f"{summary['gateway_side_exceptions'] + summary['bank_side_orphan_exceptions']:,}",
        )
        c4.metric("Gateway payments", f"{summary['unique_gateway_payments']:,}")

        if summary["unique_gateway_payments"] < 50:
            st.warning(
                "The buildathon brief asks for a 50+ record synthetic batch. "
                "For the final demo, upload at least 50 gateway payments."
            )

        if evaluation.get("classification_accuracy") is not None:
            st.info(
                f"Truth-set classification accuracy: "
                f"**{evaluation['classification_accuracy']:.1%}** "
                f"({evaluation['correct_records']}/{evaluation['evaluated_records']})."
            )

        tab1, tab2, tab3 = st.tabs(
            ["📊 Overview", "⚠️ Exceptions", "🔍 All reconciliation results"]
        )

        with tab1:
            breakdown = summary.get("exception_breakdown", [])

            if breakdown:
                breakdown_df = pd.DataFrame(
                    breakdown,
                    columns=["exception_type", "count"],
                ).set_index("exception_type")
                st.write("### Exception breakdown")
                st.bar_chart(breakdown_df)
            else:
                st.success("No exceptions found.")

            st.write("### Controller actions")
            st.markdown(
                "- Matches gateway payments to bank rows using the payment reference.\n"
                "- Allows a configurable amount tolerance.\n"
                "- Detects delayed settlements and split settlements.\n"
                "- Flags duplicate gateway rows and duplicate bank settlements.\n"
                "- Flags missing settlements and bank-only/orphan rows.\n"
                "- Produces an explanation and recommended action for every exception."
            )

        with tab2:
            if exceptions.empty:
                st.success("No unresolved exceptions.")
            else:
                columns = [
                    "record_side",
                    "payment_id",
                    "exception_type",
                    "expected_net_amount",
                    "candidate_bank_total",
                    "confidence",
                    "likely_cause",
                    "recommended_action",
                ]
                st.dataframe(
                    exceptions[columns],
                    use_container_width=True,
                    hide_index=True,
                )

        with tab3:
            st.dataframe(
                results,
                use_container_width=True,
                hide_index=True,
            )

        st.subheader("Export reports")

        e1, e2, e3 = st.columns(3)

        e1.download_button(
            "⬇️ Reconciliation CSV",
            results.to_csv(index=False).encode("utf-8"),
            file_name="reconciliation_results.csv",
            mime="text/csv",
            use_container_width=True,
        )

        e2.download_button(
            "⬇️ Exceptions CSV",
            exceptions.to_csv(index=False).encode("utf-8"),
            file_name="exceptions_report.csv",
            mime="text/csv",
            use_container_width=True,
        )

        summary_payload = {
            **summary,
            "classification_accuracy": evaluation.get("classification_accuracy"),
            "evaluated_records": evaluation.get("evaluated_records"),
            "correct_records": evaluation.get("correct_records"),
            "confusion_matrix": evaluation.get("confusion_matrix"),
        }

        e3.download_button(
            "⬇️ Summary JSON",
            json.dumps(summary_payload, indent=2, ensure_ascii=False).encode("utf-8"),
            file_name="summary.json",
            mime="application/json",
            use_container_width=True,
        )


    st.divider()
    st.caption(
        "Audit-friendly design: deterministic reconciliation performs the "
        "financial matching; AI is optional and assists with exception explanations."
    )


def _run() -> None:
    try:
        import streamlit as _st
        if _st.runtime.exists():
            run_streamlit_app()
            return
    except Exception:
        pass
    main()


if __name__ == "__main__":
    _run()