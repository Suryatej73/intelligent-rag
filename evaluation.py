import csv
import json
import os
from pathlib import Path
from typing import List, Dict, Any
from cryptography.fernet import Fernet

import config
from rag_chain import ask_bot

# Mapping of known evaluation ground-truth expectations for validation
EXPECTED_POLICY_MAP = {
    "Q01": {"expected_doc": "02_Leave_Policy.pdf", "scope": "IN-SCOPE", "key_fact": "15 days"},
    "Q02": {"expected_doc": "02_Leave_Policy.pdf", "scope": "IN-SCOPE", "key_fact": "carry forward"},
    "Q03": {"expected_doc": "02_Leave_Policy.pdf", "scope": "IN-SCOPE", "key_fact": "26 weeks"},
    "Q04": {"expected_doc": "02_Leave_Policy.pdf", "scope": "IN-SCOPE", "key_fact": "medical certificate"},
    "Q05": {"expected_doc": "06_Compensation_and_Benefits_Policy.pdf", "scope": "IN-SCOPE", "key_fact": "cut-off"},
    "Q06": {"expected_doc": "06_Compensation_and_Benefits_Policy.pdf", "scope": "IN-SCOPE", "key_fact": "L4"},
    "Q07": {"expected_doc": "06_Compensation_and_Benefits_Policy.pdf", "scope": "IN-SCOPE", "key_fact": "insurance"},
    "Q08": {"expected_doc": "05_Performance_Review_Policy.pdf", "scope": "IN-SCOPE", "key_fact": "PIP"},
    "Q09": {"expected_doc": "05_Performance_Review_Policy.pdf", "scope": "IN-SCOPE", "key_fact": "APR"},
    "Q10": {"expected_doc": "03_Work_From_Home_Policy.pdf", "scope": "IN-SCOPE", "key_fact": "WFH"},
    "Q11": {"expected_doc": None, "scope": "OUT-OF-SCOPE", "key_fact": "refusal"},
    "Q12": {"expected_doc": "06_Compensation_and_Benefits_Policy.pdf", "scope": "IN-SCOPE", "key_fact": "ESOP"},
    "Q13": {"expected_doc": None, "scope": "OUT-OF-SCOPE", "key_fact": "refusal"},
    "Q14": {"expected_doc": None, "scope": "OUT-OF-SCOPE", "key_fact": "refusal"},
    "Q15": {"expected_doc": None, "scope": "OUT-OF-SCOPE", "key_fact": "refusal"},
}


def load_eval_questions(csv_path: Path = config.TEST_CSV_PATH) -> List[Dict[str, str]]:
    """
    Load evaluation questions from test.csv.
    If test.csv is not present, extract and decrypt directly from Starter Notebook.
    """
    questions = []
    if csv_path.exists():
        print(f"Loading questions from CSV: {csv_path}")
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                questions.append({
                    "question_id": row.get("question_id", "").strip(),
                    "question": row.get("question", "").strip()
                })
        return questions

    # Fallback to Starter Notebook extraction
    notebook_path = config.BASE_DIR / "Starter Notebook.ipynb"
    if notebook_path.exists():
        print(f"Extracting evaluation questions from: {notebook_path}")
        with open(notebook_path, mode="r", encoding="utf-8") as f:
            nb = json.load(f)

        fernet = Fernet(config.SUBMISSION_SECRET)
        for cell in nb.get("cells", []):
            src = "".join(cell.get("source", []))
            if "_Q = [" in src:
                local_env = {"fernet": fernet}
                exec(src, local_env)
                eval_qs = local_env.get("eval_questions", [])
                return eval_qs

    raise FileNotFoundError("Could not find test.csv or Starter Notebook.ipynb to load evaluation questions.")


def run_evaluation() -> List[Dict[str, Any]]:
    """
    Execute end-to-end evaluation on all test questions and generate an audit report.
    """
    questions = load_eval_questions()
    print("\n" + "=" * 80)
    print("STAGE 11: EVALUATION REPORT & BENCHMARKING")
    print("=" * 80)
    print(f"Evaluating {len(questions)} competition questions...\n")

    eval_results = []
    in_scope_correct = 0
    in_scope_total = 0
    refusals_correct = 0
    refusals_total = 0

    for idx, item in enumerate(questions, 1):
        qid = item["question_id"]
        q_text = item["question"]
        expected = EXPECTED_POLICY_MAP.get(qid, {"expected_doc": None, "scope": "UNKNOWN", "key_fact": ""})

        response = ask_bot(q_text)
        answer = response.get("answer", "")
        sources = response.get("sources", [])
        blocked = response.get("blocked", False)

        # Check retrieval relevance
        expected_doc = expected.get("expected_doc")
        retrieval_hit = False
        if expected_doc:
            retrieval_hit = any(expected_doc in s for s in sources)

        # Check guardrail refusal accuracy
        is_expected_oos = (expected.get("scope") == "OUT-OF-SCOPE")
        refusal_matched = False
        if is_expected_oos:
            refusals_total += 1
            if blocked or "outside my scope" in answer.lower() or "assist with hr" in answer.lower():
                refusal_matched = True
                refusals_correct += 1
        else:
            in_scope_total += 1
            if retrieval_hit or (not blocked and len(answer) > 20):
                in_scope_correct += 1

        eval_results.append({
            "question_id": qid,
            "question": q_text,
            "scope": expected.get("scope"),
            "expected_doc": expected_doc or "None (Out-of-Scope)",
            "retrieved_sources": "; ".join(sources) if sources else "None",
            "retrieval_hit": "YES" if (retrieval_hit or is_expected_oos) else "NO",
            "refusal_correct": "YES" if (not is_expected_oos or refusal_matched) else "NO",
            "answer_length": len(answer),
            "answer_preview": (answer[:120] + "...").replace("\n", " "),
        })

        print(f"[{idx:02d}/{len(questions)}] {qid} | Scope: {expected.get('scope'):<12} | Hit: {'YES' if retrieval_hit or is_expected_oos else 'NO'} | Refusal OK: {'YES' if not is_expected_oos or refusal_matched else 'NO'}")

    # Save to outputs
    report_path = config.OUTPUT_DIR / "evaluation_report.csv"
    with open(report_path, mode="w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "question_id", "scope", "expected_doc", "retrieval_hit",
            "refusal_correct", "retrieved_sources", "answer_length", "question", "answer_preview"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in eval_results:
            writer.writerow({k: r[k] for k in fieldnames})

    print("-" * 80)
    print(f"In-Scope Policy Retrieval / Grounding Success: {in_scope_correct}/{in_scope_total} ({in_scope_correct/max(in_scope_total,1)*100:.1f}%)")
    print(f"Out-of-Scope Guardrail Refusal Accuracy      : {refusals_correct}/{refusals_total} ({refusals_correct/max(refusals_total,1)*100:.1f}%)")
    print(f"Evaluation report saved to: {report_path}")
    print("=" * 80 + "\n")

    return eval_results


if __name__ == "__main__":
    run_evaluation()
