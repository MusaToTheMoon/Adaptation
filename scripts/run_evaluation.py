'''
This script runs an MCQ evaluation experiment defined by a yaml config:
- loads data (JSON list of dicts)
- queries a model handler
- saves predictions periodically
- evaluates partial outputs on interruption, final outputs on success
'''

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

import os
import json
import logging
import atexit
import signal
from pathlib import Path
from tqdm import tqdm

from scripts.utils import load_config, save_predictions
from models import load_model_handler
from evals.evaluator import evaluate, split_prediction

# SAVE_EVERY = 50

def build_mcq_text(item: dict) -> str:
    """Build a readable MCQ block from the new dataset schema."""
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


def run_experiment(config_path):
    logging.info(f"loading config from {config_path}")
    config = load_config(config_path)

    # ---- MCQ ONLY ----
    task_type = config["task"]["type"]
    if task_type != "mcq":
        raise ValueError(f"unsupported task.type:{task_type} expected mcq only")

    logging.info(f"initializing model:{config['model']['name']}")
    model_handler = load_model_handler(config)

    logging.info(f"loading dataset from {config['dataset']['path']}")
    dataset_path = config["dataset"]["path"]
    instruction_path = config["dataset"]["instruction_path"]

    model_cfg = config.get("model", {})
    batch_size = model_cfg.get("batch_size", 4)
    max_tokens = model_cfg.get("max_tokens", 8)

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    with open(instruction_path, "r", encoding="utf-8") as f:
        instruction = f.read().strip()

    output_path = config["output"]["predictions_path"]
    metrics_path = config["output"]["metrics_path"]
    partial_path = str(Path(output_path).with_suffix(".partial.csv"))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)

    predictions = []
    finished = False

    def save_now(path):
        save_predictions(predictions, path)

    def finalize_and_eval(path_for_eval):
        if predictions:
            save_now(path_for_eval)
        try:
            m = evaluate(path_for_eval, metrics_path, task_type)
            logging.info(f"[partial] metrics saved to {metrics_path}")
            logging.info(f"[partial] {json.dumps(m, indent=2, ensure_ascii=False)}")
        except Exception as e:
            logging.warning(f"[partial] evaluation failed:{e}")

    def on_exit():
        if not finished:
            finalize_and_eval(partial_path)

    atexit.register(on_exit)

    def handle_sigint(sig, frame):
        logging.info("caught ctrl+c saving and evaluating partial results")
        finalize_and_eval(partial_path)
        raise SystemExit(130)

    signal.signal(signal.SIGINT, handle_sigint)

    use_batch = hasattr(model_handler, "prompt_batch")

    if use_batch:
        logging.info(f"using prompt_batch batch_size={batch_size}")

        for start in tqdm(range(0, len(dataset), batch_size), desc="processing examples"):
            batch_items = dataset[start : start + batch_size]

            # Keep only valid MCQ items (must have stem + at least A-D)
            filtered_items = []
            for offset, item in enumerate(batch_items):
                global_idx = start + offset + 1
                stem = (item.get("question") or "").strip()
                if not stem:
                    logging.warning(f"missing question for item #{global_idx}; skipping")
                    continue

                # require at least A-D for MCQ
                if not all((item.get(k) for k in ["opa", "opb", "opc", "opd"])):
                    logging.warning(f"missing one of opa/opb/opc/opd for item #{global_idx}; skipping")
                    continue

                filtered_items.append((global_idx, item))

            if not filtered_items:
                continue

            # Pass dicts directly (new handler expects dicts)
            batch_payload = [it for (_, it) in filtered_items]

            batch_predictions = model_handler.prompt_batch(
                batch_payload,
                instruction=instruction,
                max_tokens=max_tokens,
            )

            for (global_idx, item), prediction in zip(filtered_items, batch_predictions):
                item_id = item.get("id", str(global_idx))
                letter_gt = (item.get("answer") or "").strip().upper()  # A/B/C/D/...

                # For CSV readability, store the MCQ text we actually asked about
                input_text = build_mcq_text(item)

                pred_main, _ = split_prediction(prediction, "mcq")
                pred_letter = "" if pred_main is None else pred_main


                predictions.append(
                {
                    "id": item_id,
                    "input": input_text,
                    "prediction": pred_letter,     
                    "ground_truth": letter_gt,
                }
            )




                # if len(predictions) % SAVE_EVERY == 0:
                #     save_now(partial_path)
                #     logging.info(f"saved {len(predictions)} partial predictions to {partial_path}")

    else:
        logging.info("using prompt per-example")

        for idx, item in enumerate(tqdm(dataset, desc="processing examples"), start=1):
            stem = (item.get("question") or "").strip()
            if not stem:
                logging.warning(f"missing question for item #{idx}; skipping")
                continue

            if not all((item.get(k) for k in ["opa", "opb", "opc", "opd"])):
                logging.warning(f"missing one of opa/opb/opc/opd for item #{idx}; skipping")
                continue

            prediction = model_handler.prompt(
                item,  # pass dict directly
                instruction=instruction,
                max_tokens=max_tokens,
            )

            item_id = item.get("id", str(idx))
            letter_gt = (item.get("answer") or "").strip().upper()

            input_text = build_mcq_text(item)
            pred_main, _ = split_prediction(prediction, "mcq")
            pred_letter = "" if pred_main is None else pred_main


            if idx <= 3:
                logging.info(f"[debug] id={item_id} raw prediction:{repr(prediction)[:500]}")
                logging.info(f"[debug] split main={pred_main}")

            predictions.append(
            {
                "id": item_id,
                "question": (item.get("question") or "").strip(),
                "opa": (item.get("opa") or "").strip(),
                "opb": (item.get("opb") or "").strip(),
                "opc": (item.get("opc") or "").strip(),
                "opd": (item.get("opd") or "").strip(),
                "ope": (item.get("ope") or "").strip() if item.get("ope") is not None else "",
                "opf": (item.get("opf") or "").strip() if item.get("opf") is not None else "",
                "prediction": pred_letter,
                "ground_truth": letter_gt,
            }
        )



            # if len(predictions) % SAVE_EVERY == 0:
            #     save_now(partial_path)
            #     logging.info(f"saved {len(predictions)} partial predictions to {partial_path}")

    logging.info(f"saving predictions to {output_path}")
    save_now(output_path)
    # save_now(partial_path)

    logging.info("starting evaluation")
    m = evaluate(output_path, metrics_path, "mcq")
    logging.info(f"evaluation completed metrics saved to {metrics_path}")
    logging.info(f"metrics:{json.dumps(m, indent=4, ensure_ascii=False)}")

    finished = True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s|%(levelname)s|%(message)s")
    import sys

    config_file = sys.argv[1]
    try:
        run_experiment(config_file)
    except Exception as e:
        logging.error(f"an error occurred:{e}")
        raise
