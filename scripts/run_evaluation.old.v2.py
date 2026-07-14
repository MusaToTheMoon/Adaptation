"""
This script runs an evaluation experiment defined by a yaml config:
- loads data (JSON list of dicts)
- queries a model handler (single or batch)
- saves predictions
- evaluates partial outputs on interruption, final outputs on success

Supported task types:
- mcq: returns a single letter (A-F), metric = accuracy
- answer_generation: returns free-form text, metrics = BLEU/ROUGE/BERTScore (handled in evals/evaluator.py)

Optional debug output:
- If output.debug_path is set in the YAML, this script will save per-example debug rows
  (useful for handlers like AutoCAP that expose internal planning/voting metadata).
"""

import sys
import json
import logging
import atexit
import signal
from pathlib import Path
from typing import Dict, Any

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.utils import load_config, save_predictions  # noqa: E402
from models import load_model_handler  # noqa: E402
from evals.evaluator import evaluate, split_prediction  # noqa: E402


# -------------------------
# Input builders
# -------------------------
def build_mcq_text(item: dict) -> str:
    """Build a readable MCQ block from the dataset schema."""
    stem = (item.get("question") or "").strip()

    option_map = {
        "A": item.get("opa"),
        "B": item.get("opb"),
        "C": item.get("opc"),
        "D": item.get("opd"),
        "E": item.get("ope"),
        "F": item.get("opf"),
    }

    lines = []
    for letter in ["A", "B", "C", "D", "E", "F"]:
        txt = option_map.get(letter)
        if txt is None:
            continue
        txt = str(txt).strip()
        if not txt:
            continue
        lines.append(f"{letter}) {txt}")

    if stem and lines:
        return stem + "\n\n" + "\n".join(lines)
    return stem or ""


def build_ansgen_text(item: dict) -> str:
    """Build the input text for answer_generation."""
    return (item.get("question") or "").strip()


# -------------------------
# Hard GT selection (ONCE AND FOR ALL)
# -------------------------
LETTER_SET = {"A", "B", "C", "D", "E", "F"}


def get_ground_truth_or_die(item: Dict[str, Any], task_type: str, item_id: str) -> str:
    """
    Hard rule:
      - mcq: ground_truth = item['answer'] (letter)
      - answer_generation: ground_truth = item['answer_text'] (text) and MUST NOT be a letter.
    """
    if task_type == "mcq":
        gt = (item.get("answer") or "").strip().upper()
        if not gt:
            raise RuntimeError(f"[FATAL] mcq missing 'answer' for id={item_id}")
        return gt

    # answer_generation: FORCE answer_text
    if "answer_text" not in item:
        raise RuntimeError(
            f"[FATAL] answer_generation requires 'answer_text' but key is missing for id={item_id}. "
            f"Available keys={list(item.keys())}"
        )

    gt = str(item["answer_text"]).strip()
    if not gt:
        raise RuntimeError(f"[FATAL] answer_generation 'answer_text' is empty for id={item_id}")

    # if someone accidentally put the letter in answer_text, crash loudly
    if gt.strip().upper() in LETTER_SET:
        raise RuntimeError(
            f"[FATAL] answer_generation ground_truth is a LETTER for id={item_id}. "
            f"answer_text={gt!r} answer={item.get('answer')!r}"
        )

    return gt


# -------------------------
# Validators
# -------------------------
def is_valid_mcq_item(item: dict) -> bool:
    stem = (item.get("question") or "").strip()
    if not stem:
        return False
    # require at least A-D
    return all((item.get(k) for k in ["opa", "opb", "opc", "opd"]))


def is_valid_ansgen_item(item: dict) -> bool:
    q = (item.get("question") or "").strip()
    if not q:
        return False
    # require answer_text (Task 2)
    return "answer_text" in item and str(item["answer_text"]).strip() != ""


# -------------------------
# Debug helpers
# -------------------------
def save_debug_jsonl(rows, path: str):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# -------------------------
# Main experiment
# -------------------------
def run_experiment(config_path: str):
    logging.info(f"loading config from {config_path}")
    config = load_config(config_path)

    # DEBUG + task parsing
    task_cfg = config.get("task", {})
    task_type = str(task_cfg.get("type", "mcq")).strip().lower()

    if task_type not in {"mcq", "answer_generation"}:
        raise ValueError(f"unsupported task.type={task_type!r}")

    lang = task_cfg.get("lang", "en")

    logging.info(f"[DEBUG] entrypoint file={__file__}")
    logging.info(f"[DEBUG] task_type={task_type!r} lang={lang!r}")

    task_cfg = config.get("task", {})
    task_type = (task_cfg.get("type", "mcq") or "mcq").strip()
    if task_type not in {"mcq", "answer_generation"}:
        raise ValueError(f"unsupported task.type:{task_type} expected mcq or answer_generation")

    # optional (only relevant for answer_generation metrics/tokenization)
    lang = task_cfg.get("lang", "en")

    logging.info(f"[DEBUG] task_type={task_type} lang={lang}")

    logging.info(f"initializing model:{config['model']['name']}")
    model_handler = load_model_handler(config)

    logging.info(f"loading dataset from {config['dataset']['path']}")
    dataset_path = config["dataset"]["path"]
    instruction_path = config["dataset"]["instruction_path"]

    model_cfg = config.get("model", {})
    model_type = str(model_cfg.get("type", "")).strip().lower()
    batch_size = int(model_cfg.get("batch_size", 4))

    # sensible defaults per task
    if task_type == "mcq":
        max_tokens = int(model_cfg.get("max_tokens", 8))
    else:
        max_tokens = int(model_cfg.get("max_tokens", 256))

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    with open(instruction_path, "r", encoding="utf-8") as f:
        instruction = f.read().strip()

    output_cfg = config["output"]
    output_path = output_cfg["predictions_path"]
    metrics_path = output_cfg["metrics_path"]
    debug_path = output_cfg.get("debug_path")
    partial_path = str(Path(output_path).with_suffix(".partial.csv"))
    partial_debug_path = str(Path(debug_path).with_suffix(".partial.jsonl")) if debug_path else None

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    if debug_path:
        Path(debug_path).parent.mkdir(parents=True, exist_ok=True)

    logging.info(f"[DEBUG] predictions_path={output_path}")
    logging.info(f"[DEBUG] metrics_path={metrics_path}")
    if debug_path:
        logging.info(f"[DEBUG] debug_path={debug_path}")

    predictions = []
    debug_rows = []
    finished = False

    def save_now(path):
        save_predictions(predictions, path)

    def save_debug_now(path):
        if path and debug_rows:
            save_debug_jsonl(debug_rows, path)

    def finalize_and_eval(path_for_eval, debug_path_for_eval=None):
        if predictions:
            save_now(path_for_eval)
        if debug_path_for_eval and debug_rows:
            save_debug_now(debug_path_for_eval)
        try:
            m = evaluate(path_for_eval, metrics_path, task_type, lang=lang)
            logging.info(f"[partial] metrics saved to {metrics_path}")
            logging.info(f"[partial] {json.dumps(m, indent=2, ensure_ascii=False)}")
        except Exception as e:
            logging.warning(f"[partial] evaluation failed:{e}")

    def on_exit():
        if not finished:
            finalize_and_eval(partial_path, partial_debug_path)

    atexit.register(on_exit)

    def handle_sigint(sig, frame):
        logging.info("caught ctrl+c saving and evaluating partial results")
        finalize_and_eval(partial_path, partial_debug_path)
        raise SystemExit(130)

    signal.signal(signal.SIGINT, handle_sigint)

    use_batch = hasattr(model_handler, "prompt_batch")
    wants_debug = bool(debug_path)

    # -------------------------
    # Batch path
    # -------------------------
    if use_batch:
        logging.info(f"using prompt_batch batch_size={batch_size}")

        for start in tqdm(range(0, len(dataset), batch_size), desc="processing examples"):
            batch_items = dataset[start : start + batch_size]

            filtered_items = []
            for offset, item in enumerate(batch_items):
                global_idx = start + offset + 1

                if task_type == "mcq":
                    if not is_valid_mcq_item(item):
                        logging.warning(f"[mcq] invalid item #{global_idx}; skipping")
                        continue
                else:
                    if not is_valid_ansgen_item(item):
                        logging.warning(f"[answer_generation] invalid item #{global_idx}; skipping")
                        continue

                filtered_items.append((global_idx, item))

            if not filtered_items:
                continue

            batch_payload = [it for (_, it) in filtered_items]

            # For debug-capable handlers like AutoCAP, collect full debug payloads.
            if wants_debug and model_type == "autocap":
                batch_predictions = [
                    model_handler.prompt(
                        s,
                        instruction=instruction,
                        max_tokens=max_tokens,
                        task_type=task_type,
                        return_debug=True,
                    )
                    for s in batch_payload
                ]
            else:
                batch_predictions = model_handler.prompt_batch(
                    batch_payload,
                    instruction=instruction,
                    max_tokens=max_tokens,
                    task_type=task_type,
                )

            for (global_idx, item), prediction in zip(filtered_items, batch_predictions):
                item_id = item.get("id", str(global_idx))

                # --- FIXED GT SELECTION ---
                gt = get_ground_truth_or_die(item, task_type, item_id)

                if task_type == "mcq":
                    input_text = build_mcq_text(item)

                    if isinstance(prediction, dict) and "final_prediction" in prediction:
                        pred_main, _ = split_prediction(prediction.get("final_prediction"), "mcq")
                        pred_out = "" if pred_main is None else pred_main

                        if wants_debug:
                            debug_rows.append(
                                {
                                    "id": item_id,
                                    "input": input_text,
                                    "ground_truth": gt,
                                    **prediction,
                                }
                            )
                    else:
                        pred_main, _ = split_prediction(prediction, "mcq")
                        pred_out = "" if pred_main is None else pred_main
                else:
                    input_text = build_ansgen_text(item)
                    pred_out = "" if prediction is None else str(prediction).strip()

                # debug first few rows
                if global_idx <= 3:
                    logging.info(f"[DEBUG] id={item_id} task_type={task_type} gt_written={gt!r}")
                    logging.info(f"[DEBUG] id={item_id} pred_written={pred_out!r}")

                predictions.append(
                    {
                        "id": item_id,
                        "input": input_text,
                        "prediction": pred_out,
                        "ground_truth": gt,
                    }
                )

    # -------------------------
    # Single-example path
    # -------------------------
    else:
        logging.info("using prompt per-example")

        for idx, item in enumerate(tqdm(dataset, desc="processing examples"), start=1):
            item_id = item.get("id", str(idx))

            if task_type == "mcq":
                if not is_valid_mcq_item(item):
                    logging.warning(f"[mcq] invalid item #{idx}; skipping")
                    continue
            else:
                if not is_valid_ansgen_item(item):
                    logging.warning(f"[answer_generation] invalid item #{idx}; skipping")
                    continue

            if wants_debug and model_type == "autocap":
                prediction = model_handler.prompt(
                    item,
                    instruction=instruction,
                    max_tokens=max_tokens,
                    task_type=task_type,
                    return_debug=True,
                )
            else:
                prediction = model_handler.prompt(
                    item,
                    instruction=instruction,
                    max_tokens=max_tokens,
                    task_type=task_type,
                )

            # --- FIXED GT SELECTION ---
            gt = get_ground_truth_or_die(item, task_type, item_id)

            if task_type == "mcq":
                input_text = build_mcq_text(item)

                if isinstance(prediction, dict) and "final_prediction" in prediction:
                    pred_main, _ = split_prediction(prediction.get("final_prediction"), "mcq")
                    pred_out = "" if pred_main is None else pred_main

                    if wants_debug:
                        debug_rows.append(
                            {
                                "id": item_id,
                                "input": input_text,
                                "ground_truth": gt,
                                **prediction,
                            }
                        )
                else:
                    pred_main, _ = split_prediction(prediction, "mcq")
                    pred_out = "" if pred_main is None else pred_main
            else:
                input_text = build_ansgen_text(item)
                pred_out = "" if prediction is None else str(prediction).strip()

            if idx <= 3:
                logging.info(f"[DEBUG] id={item_id} task_type={task_type} gt_written={gt!r}")
                logging.info(f"[DEBUG] id={item_id} pred_written={pred_out!r}")

            if task_type == "answer_generation":
                logging.info(
                    f"[DEBUG GT CHECK] id={item_id} answer={item.get('answer')!r} "
                    f"answer_text={item.get('answer_text')!r}"
                )
                logging.info(f"[DEBUG GT CHECK] id={item_id} gt_written={gt!r}")

            predictions.append(
                {
                    "id": item_id,
                    "input": input_text,
                    "prediction": pred_out,
                    "ground_truth": gt,
                }
            )

    logging.info(f"saving predictions to {output_path}")
    if predictions:
        logging.info(f"[CHECK BEFORE SAVE] first row ground_truth={predictions[0]['ground_truth']!r}")
    save_now(output_path)

    if debug_path and debug_rows:
        save_debug_now(debug_path)
        logging.info(f"saved debug rows to {debug_path}")

    logging.info("starting evaluation")
    m = evaluate(output_path, metrics_path, task_type, lang=lang)
    logging.info(f"evaluation completed metrics saved to {metrics_path}")
    logging.info(f"metrics:{json.dumps(m, indent=4, ensure_ascii=False)}")

    finished = True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s|%(levelname)s|%(message)s")
    if len(sys.argv) < 2:
        print("usage: python scripts/run_evaluation.py <config.yaml>")
        raise SystemExit(2)

    config_file = sys.argv[1]
    try:
        run_experiment(config_file)
    except Exception as e:
        logging.error(f"an error occurred:{e}")
        raise