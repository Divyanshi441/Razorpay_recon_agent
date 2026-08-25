# 💳 AI Finance Controller

An AI-assisted finance operations agent built for the **Razorpay AI Buildathon – Track 04: AI Finance Controller**.

The application automates the reconciliation loop between payment gateway records and bank settlement records. It processes 50+ record batches, matches transactions, calculates the overall match rate, identifies unresolved exceptions, and provides explanations and recommended actions for cases that require attention.

## 🚀 What it does

- 📄 Upload gateway payment and bank settlement CSV files
- 🔎 Automatically reconcile transactions
- 📊 Calculate reconciliation match rate
- ⚠️ Detect unresolved exceptions
- 🔄 Identify delayed and split settlements
- 💰 Detect amount mismatches
- 🧾 Detect duplicate and orphan records
- 🤖 Provide optional AI-assisted exception explanations
- 📥 Export reconciliation results and exception reports
- 🖥️ Interactive Streamlit dashboard

## 🧠 How it works

```text
Gateway Payments CSV
        +
Bank Settlements CSV
        ↓
   Reconciliation Engine
        ↓
 ┌───────────────┐
 │ Transaction   │
 │ Matching      │
 └───────┬───────┘
         ↓
 ┌───────────────────────┐
 │ Matched Transactions  │
 │ + Unresolved          │
 │   Exceptions          │
 └───────────┬───────────┘
             ↓
      Finance Report
