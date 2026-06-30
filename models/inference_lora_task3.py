import os
import sys
import json
import csv
import argparse
from pathlib import Path
from typing import List, Dict, Any

import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Mistral3ForConditionalGeneration,
)
from peft import PeftModel

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.metrics import calculate_bert_score


HF_CACHE = os.environ.get("HF_HOME", "/scratch/ca2627/huggingface")
os.environ.setdefault("HF_HOME", HF_CACHE)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
os.environ.setdefault("TORCH_USE_CUDA_DSA", "1")


def load_json(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list in {path}")
    return data


def parse_dialogue_prediction(raw: str) -> str:
    if not raw:
        return ""
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        upper = line.upper()
        if upper.startswith("ANSWER:"):
            return line[len("ANSWER:"):].strip()
        return line
    return raw.strip()


def build_dialogue_text(item: dict) -> str:
    dialogue = item.get("Dialogue", "")

    if isinstance(dialogue, str):
        return dialogue.strip()

    if isinstance(dialogue, list):
        lines = []
        for turn in dialogue:
            if isinstance(turn, dict):
                role = (
                    turn.get("role")
                    or turn.get("speaker")
                    or turn.get("Role")
                    or turn.get("Speaker")
                    or "Unknown"
                )
                text = (
                    turn.get("content")
                    or turn.get("utterance")
                    or turn.get("text")
                    or turn.get("Content")
                    or turn.get("Utterance")
                    or ""
                )
                lines.append(f"{role}: {str(text).strip()}")
            else:
                lines.append(str(turn).strip())
        return "\n".join(lines)

    return ""


class LoRADialogueInference:
    def __init__(
        self,
        model_name: str,
        base_model: str,
        adapter_path: str,
        use_4bit: bool = False,
        cache_dir: str = HF_CACHE,
        offline: bool = False,
    ):
        self.model_name = (model_name or "mistral").strip().lower()
        self.base_model = base_model
        self.adapter_path = adapter_path
        self.use_4bit = use_4bit
        self.cache_dir = cache_dir
        self.offline = offline

        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

        self.local_files_only = bool(self.offline)
        if self.offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
        else:
            os.environ.pop("HF_HUB_OFFLINE", None)

        if self.model_name == "mistral":
            self._load_mistral()
        elif self.model_name == "medgemma":
            self._load_medgemma()
        else:
            raise ValueError(
                f"Unsupported --model_name '{model_name}'. Expected 'mistral' or 'medgemma'."
            )

    def _load_mistral(self) -> None:
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

        print("[INFO] Loading tokenizer...")
        if os.path.isdir(self.base_model):
            tok_path = os.path.join(self.base_model, "tekken.json")
            self.tokenizer = MistralTokenizer.from_file(tok_path)
        else:
            if self.local_files_only:
                raise ValueError(
                    f"offline=True but base_model is not a local directory: {self.base_model}"
                )
            self.tokenizer = MistralTokenizer.from_hf_hub(self.base_model)

        print("[INFO] Loading base model...")
        if self.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            base = Mistral3ForConditionalGeneration.from_pretrained(
                self.base_model,
                quantization_config=bnb_config,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                cache_dir=self.cache_dir,
                local_files_only=self.local_files_only,
            )
        else:
            base = Mistral3ForConditionalGeneration.from_pretrained(
                self.base_model,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                cache_dir=self.cache_dir,
                local_files_only=self.local_files_only,
            )

        print("[INFO] Loading LoRA adapter...")
        self.model = PeftModel.from_pretrained(base, self.adapter_path)
        self.model.eval()

    def _load_medgemma(self) -> None:
        print("[INFO] Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model,
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
            use_fast=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("[INFO] Loading base model...")
        model_kwargs = dict(
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
        )
        if self.use_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        base = AutoModelForCausalLM.from_pretrained(self.base_model, **model_kwargs)

        print("[INFO] Loading LoRA adapter...")
        self.model = PeftModel.from_pretrained(base, self.adapter_path)
        self.model.eval()

    def generate_raw(
        self,
        item: Dict[str, Any],
        instruction: str,
        max_tokens: int = 256,
        do_sample: bool = False,
    ) -> str:
        if self.model_name == "mistral":
            return self._generate_raw_mistral(item, instruction, max_tokens, do_sample)
        if self.model_name == "medgemma":
            return self._generate_raw_medgemma(item, instruction, max_tokens, do_sample)
        raise ValueError(f"Unsupported model_name '{self.model_name}'")

    def _generate_raw_mistral(
        self,
        item: Dict[str, Any],
        instruction: str,
        max_tokens: int,
        do_sample: bool,
    ) -> str:
        from mistral_common.protocol.instruct.request import ChatCompletionRequest

        user_text = build_dialogue_text(item)
        if not user_text:
            return ""

        messages = [
            {"role": "system", "content": (instruction or "").strip()},
            {"role": "user", "content": [{"type": "text", "text": user_text}]},
        ]

        try:
            torch.cuda.empty_cache()
            req = ChatCompletionRequest(messages=messages)
            tokenized = self.tokenizer.encode_chat_completion(req)
            input_ids = torch.tensor(
                [tokenized.tokens], dtype=torch.long, device=self.model.device
            )
            attention_mask = torch.ones_like(input_ids)

            with torch.inference_mode():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_tokens,
                    do_sample=do_sample,
                )

            if outputs is None or outputs.shape[0] == 0:
                return ""

            gen_ids = outputs[0][len(tokenized.tokens):]
            raw_text = self.tokenizer.decode(gen_ids).strip()
            return raw_text or ""

        except Exception as e:
            print(f"[ERROR] Generation failed for id={item.get('Case ID')}: {e}")
            return ""

        finally:
            torch.cuda.empty_cache()

    def _generate_raw_medgemma(
        self,
        item: Dict[str, Any],
        instruction: str,
        max_tokens: int,
        do_sample: bool,
    ) -> str:
        user_text = build_dialogue_text(item)
        if not user_text:
            return ""

        messages = [
            {"role": "system", "content": (instruction or "").strip()},
            {"role": "user", "content": user_text},
        ]

        try:
            torch.cuda.empty_cache()
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            input_len = inputs["input_ids"].shape[-1]

            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=do_sample,
                )

            if outputs is None or outputs.shape[0] == 0:
                return ""

            gen_ids = outputs[0][input_len:]
            raw_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            return raw_text or ""

        except Exception as e:
            print(f"[ERROR] Generation failed for id={item.get('Case ID')}: {e}")
            return ""

        finally:
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="LoRA inference — Task 3 Dialogue Completion")
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--instruction_file", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="mistral")
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True, help="CSV predictions")
    parser.add_argument("--metrics_file", type=str, required=True, help="JSON metrics")
    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--lang", type=str, default="ar")
    parser.add_argument("--bert_device", type=str, default="cpu")
    args = parser.parse_args()

    with open(args.instruction_file, "r", encoding="utf-8") as f:
        instruction = f.read().strip()

    dataset = load_json(args.test_file)

    runner = LoRADialogueInference(
        model_name=args.model_name,
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        use_4bit=args.use_4bit,
        offline=args.offline,
    )

    outputs = []
    for item in tqdm(dataset, desc="Task-3 Dialogue inference"):
        item_id = str(item.get("Case ID", item.get("id", ""))).strip()
        input_text = build_dialogue_text(item)
        raw_pred = runner.generate_raw(
            item,
            instruction=instruction,
            max_tokens=args.max_tokens,
            do_sample=False,
        )

        pred_out = parse_dialogue_prediction(raw_pred)
        gt = str(item.get("Gold Response", "")).strip()
        primary_reasoning_objective = str(
            item.get("primary_reasoning_objective", item.get("Primary Reasoning Objective", ""))
        ).strip()

        outputs.append(
            {
                "id": item_id,
                "input": input_text,
                "prediction": pred_out,
                "ground_truth": gt,
                "primary_reasoning_objective": primary_reasoning_objective,
                "raw_output": raw_pred,
            }
        )

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "input",
                "prediction",
                "ground_truth",
                "primary_reasoning_objective",
                "raw_output",
            ],
        )
        writer.writeheader()
        writer.writerows(outputs)

    print(f"[INFO] Saved predictions CSV to {args.output_file}")

    predictions = [r["prediction"] for r in outputs]
    ground_truths = [r["ground_truth"] for r in outputs]

    print(f"[INFO] Computing BERTScore on {args.bert_device}...")
    bert_metrics = calculate_bert_score(
        predictions=predictions,
        references=ground_truths,
        lang=args.lang,
        device=args.bert_device,
    )
    if bert_metrics:
        print(f"[INFO] BERTScore: {json.dumps(bert_metrics, ensure_ascii=False, indent=2)}")
    else:
        print("[WARN] BERTScore failed or bert_score not installed.")
        bert_metrics = {}

    metrics_dir = os.path.dirname(args.metrics_file)
    if metrics_dir:
        os.makedirs(metrics_dir, exist_ok=True)

    with open(args.metrics_file, "w", encoding="utf-8") as f:
        json.dump(bert_metrics, f, indent=4, ensure_ascii=False)

    print(f"[INFO] Saved metrics JSON to {args.metrics_file}")


if __name__ == "__main__":
    main()
