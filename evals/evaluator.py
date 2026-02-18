'''

This script evaluates prediction csv files and writes metrics json for mcq and answer_generation tasks. It parses predictions into comparable forms and computes task-appropriate metrics.

''' 
import json
import sys
import pandas as pd

from evals.metrics import (
    calculate_accuracy,
    calculate_bleu,
    calculate_rouge,
    calculate_bert_score,
    extract_letter,
)


def split_prediction(pred, task_type: str):
    if pred is None:
        return None, None

    pred_str = str(pred).strip()

    if task_type == "mcq":
        letter = extract_letter(pred_str)
        return letter, None

    if task_type == "answer_generation":
        return None, pred_str

    raise ValueError(f"unsupported task_type:{task_type} expected mcq or answer_generation")


def evaluate(predictions_path: str, metrics_path: str, task_type: str):
    df = pd.read_csv(predictions_path, dtype=str)
    if "prediction" not in df.columns or "ground_truth" not in df.columns:
        raise ValueError("csv must contain 'prediction' and 'ground_truth' columns")

    predictions_raw = df["prediction"].tolist()
    ground_truths = df["ground_truth"].astype(str).tolist()

    metrics = {}

    if task_type == "mcq":
        pred_letters = []
        for p in predictions_raw:
            ltr, _ = split_prediction(p, "mcq")
            pred_letters.append("" if ltr is None else ltr)
        metrics["accuracy_letter"] = calculate_accuracy(pred_letters, ground_truths)

    elif task_type == "answer_generation":
        preds_text = [("" if p is None else str(p)) for p in predictions_raw]
        metrics["bleu"] = calculate_bleu(preds_text, ground_truths)
        metrics.update(calculate_rouge(preds_text, ground_truths))
        bs = calculate_bert_score(preds_text, ground_truths)
        if bs is not None:
            metrics.update(bs)

    else:
        raise ValueError(f"unsupported task_type:{task_type} expected mcq or answer_generation")

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    return metrics


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("usage: python evaluator.py <predictions.csv> <metrics.json> <task_type>")
        sys.exit(1)

    predictions_file = sys.argv[1]
    metrics_file = sys.argv[2]
    task = sys.argv[3]

    out = evaluate(predictions_file, metrics_file, task)
    print("evaluation completed. metrics saved to:", metrics_file)
    print(json.dumps(out, indent=2, ensure_ascii=False))
