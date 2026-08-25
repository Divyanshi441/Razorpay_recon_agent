from __future__ import annotations

import argparse
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


def load_data(
    gateway_csv: Path,
    bank_csv: Path,
    truth_csv: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    gateway = pd.read_csv(gateway_csv)
    bank = pd.read_csv(bank_csv)
    truth = pd.read_csv(truth_csv) if truth_csv and Path(truth_csv).exists() else None

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


if __name__ == "__main__":
    main()
