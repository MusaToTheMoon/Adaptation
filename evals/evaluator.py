import json
import sys
import logging
import pandas as pd
import torch

from evals.metrics import (
    calculate_accuracy,
    calculate_bleu,
    calculate_rouge,
    calculate_bert_score,
    calculate_bertscore_option_accuracy, # acc via bertscore
    extract_letter,
)

import re
import inspect
assert "lang" in str(inspect.signature(calculate_bleu)), (
    f"Wrong calculate_bleu loaded from {inspect.getsourcefile(calculate_bleu)} "
    f"sig={inspect.signature(calculate_bleu)}"
)

print(f"[DEBUG evaluator] imported evaluator from: {__file__}")

def split_prediction(prediction, task_type):
    """
    For MCQ: extract letter A-F from model output.
    For answer_generation: return raw text.
    """
    if prediction is None:
        return None, None

    text = str(prediction).strip()

    if task_type == "mcq":
        upper = text.upper()

        # First try strict format: ANSWER: X
        m = re.search(r"\bANSWER\s*[:=]\s*([A-F])\b", upper)
        if m:
            return m.group(1), text

        # Fallback: any standalone letter
        m = re.search(r"\b([A-F])\b", upper)
        if m:
            return m.group(1), text

        return None, text

    # For answer generation just return text
    return text, text


def evaluate(predictions_path: str, metrics_path: str, task_type: str, lang: str = "en"):
    df = pd.read_csv(predictions_path)

    if "prediction" not in df.columns or "ground_truth" not in df.columns:
        raise ValueError("csv must contain 'prediction' and 'ground_truth' columns")

    preds = df["prediction"].fillna("").astype(str).tolist()
    gts   = df["ground_truth"].fillna("").astype(str).tolist()

    metrics = {}

    if task_type == "mcq":
        pred_letters = []
        for p in preds:
            ltr = extract_letter(p)
            pred_letters.append("" if ltr is None else ltr)
        metrics["accuracy_letter"] = calculate_accuracy(pred_letters, gts)

    elif task_type == "answer_generation":
        metrics["bleu"] = calculate_bleu(preds, gts, lang=lang)
        metrics.update(calculate_rouge(preds, gts, lang=lang))

        device = "cuda" if torch.cuda.is_available() else "cpu"
        bs = calculate_bert_score(preds, gts, lang=lang, device=device)
        if bs is not None:
            metrics.update(bs)
        else:
            logging.warning(
                "BERTScore was not added to metrics for task_type='answer_generation'. "
                "See earlier log lines from evals.metrics for the root cause. "
                "predictions_path=%s metrics_path=%s lang=%s device=%s",
                predictions_path,
                metrics_path,
                lang,
                device,
            )
        
        ######## accuracy via bertscore ##############
        inputs = df["input"].fillna("").astype(str).tolist() if "input" in df.columns else []
        if inputs:
            sem_acc = calculate_bertscore_option_accuracy(
                preds, gts, inputs, lang=lang, device=device,
            )
            if sem_acc is not None:
                metrics["accuracy_bertscore_option"] = sem_acc
            else:
                logging.warning(
                    "BERTScore option-matching accuracy was not computed. "
                    "predictions_path=%s lang=%s device=%s",
                    predictions_path, lang, device,
                )
        ##############################################

    else:
        raise ValueError(f"unsupported task_type:{task_type} expected mcq or answer_generation")

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    return metrics

if __name__ == "__main__":
    if len(sys.argv) not in {4, 5}:
        print("usage: python evaluator.py <predictions.csv> <metrics.json> <task_type> [lang]")
        sys.exit(1)

    predictions_file = sys.argv[1]
    metrics_file = sys.argv[2]
    task = sys.argv[3]
    lang = sys.argv[4] if len(sys.argv) == 5 else "en"

    out = evaluate(predictions_file, metrics_file, task, lang=lang)
    print("evaluation completed. metrics saved to:", metrics_file)
    print(json.dumps(out, indent=2, ensure_ascii=False))