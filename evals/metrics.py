'''

This script implements metrics for mcq and answer_generation evaluation. It provides bleu/rouge/bertscore metrics

'''

import re
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer

try:
    import bert_score
except Exception:
    bert_score = None

try:
    from bleurt import score as bleurt_score
except Exception:
    bleurt_score = None

#------------------------------------------Task 1 ----------------------------------------------#

def extract_letter(text):
    if not text:
        return None
    t = str(text).strip()

    m = re.search(r"(?i)\banswer(?:\s*is)?\s*[:：]?\s*([A-F])\b", t)
    if m:
        return m.group(1).upper()

    first = next((ln for ln in t.splitlines() if ln.strip()), "")
    m = re.match(r"^\s*([A-Fa-f])\s*[\.\)]?\s*$", first)
    if m:
        return m.group(1).upper()

    m = re.match(r"^\s*([A-Fa-f])\s*$", t)
    if m:
        return m.group(1).upper()

    return None


def calculate_accuracy(predictions, ground_truths):
    correct = 0
    total = 0

    for p, g in zip(predictions, ground_truths):
        total += 1
        p_letter = extract_letter(p) if p is not None else None
        g_letter = extract_letter(g)

        if p_letter and g_letter and p_letter == g_letter:
            correct += 1

    return correct / total if total > 0 else 0.0

##################### Acc via bertscore #########################
def parse_options_from_input(input_text: str) -> Dict[str, str]:
    """Parse MCQ options from the input column text (e.g. 'A) option text')."""
    if not input_text:
        return {}
    option_re = re.compile(r"^\s*([A-Fa-f])\s*[\)\.:\-]\s*(.+)$", re.MULTILINE)
    options: Dict[str, str] = {}
    for m in option_re.finditer(str(input_text)):
        letter = m.group(1).upper()
        text = m.group(2).strip()
        if text:
            options[letter] = text
    return options


def calculate_bertscore_option_accuracy(
    predictions: List[str],
    ground_truths: List[str],
    inputs: List[str],
    lang: str = "en",
    model_type: Optional[str] = None,
    device: str = "cpu",
    rescale_with_baseline: bool = True,
) -> Optional[float]:
    """
    Accuracy via BERTScore semantic option matching.
    For each sample:
      1) Parse options A..F from the input text.
      2) Compute BERTScore F1 between prediction and each option.
      3) Select the option with highest F1.
      4) Compare selected letter to ground-truth letter -> 1 or 0.
    Returns average accuracy across all samples.
    """
    if bert_score is None:
        return None

    n = len(predictions)
    if n == 0:
        return 0.0

    # Build flat lists for a single batched BERTScore call
    flat_hyps: List[str] = []
    flat_refs: List[str] = []
    pair_meta: List[Tuple[int, str]] = []  # (sample_index, option_letter)

    for i in range(n):
        hyp = "" if predictions[i] is None else str(predictions[i]).strip()
        opts = parse_options_from_input(inputs[i])
        for letter in sorted(opts):
            flat_hyps.append(hyp)
            flat_refs.append(opts[letter])
            pair_meta.append((i, letter))

    if not flat_hyps:
        return 0.0

    try:
        _, _, F1 = bert_score.score(
            flat_hyps,
            flat_refs,
            lang=lang,
            model_type=model_type,
            device=device,
            rescale_with_baseline=rescale_with_baseline,
            verbose=False,
        )
    except Exception:
        return None

    f1_scores = np.clip(F1.detach().cpu().numpy(), 0.0, 1.0)

    # Pick highest-scoring option per sample
    best: Dict[int, Tuple[str, float]] = {}  # idx -> (letter, score)
    for score_val, (idx, letter) in zip(f1_scores.tolist(), pair_meta):
        prev = best.get(idx)
        if prev is None or score_val > prev[1]:
            best[idx] = (letter, score_val)

    correct = 0
    for i in range(n):
        chosen = best.get(i)
        if chosen is None:
            continue
        gt_letter = extract_letter(ground_truths[i])
        if gt_letter and chosen[0] == gt_letter:
            correct += 1

    return float(correct / n)

#####################################################
#------------------------------------------Task 2 ----------------------------------------------#

def _safe_tokenize(text: str, lang: str = "en") -> List[str]:
    text = "" if text is None else str(text).strip()
    if not text:
        return []

    # Arabic: safer than NLTK punkt requirements
    if lang.lower().startswith("ar"):
        return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)

    try:
        return word_tokenize(text)
    except Exception:
        return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def _normalize_arabic(text: str) -> str:
    if text is None:
        return ""
    text = str(text)
    text = re.sub(r"[\u064B-\u0652]", "", text)  # remove diacritics
    text = text.replace("آ", "ا").replace("أ", "ا").replace("إ", "ا")
    text = text.replace("ى", "ي").replace("ؤ", "و").replace("ئ", "ي")
    text = text.replace("ة", "ه")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def calculate_bleu(predictions: List[str], references: List[str], lang: str = "ar") -> float:
    smoothie = SmoothingFunction().method1
    scores = []

    for hyp, ref in zip(predictions, references):
        hyp = "" if hyp is None else str(hyp)
        ref = "" if ref is None else str(ref)

        if lang.lower().startswith("ar"):
            hyp = _normalize_arabic(hyp)
            ref = _normalize_arabic(ref)

        ref_tokens = _safe_tokenize(ref, lang=lang)
        hyp_tokens = _safe_tokenize(hyp, lang=lang)

        if not ref_tokens and not hyp_tokens:
            scores.append(1.0)
            continue
        if not ref_tokens or not hyp_tokens:
            scores.append(0.0)
            continue

        bleu = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothie)
        scores.append(float(bleu))

    return float(sum(scores) / len(scores)) if scores else 0.0


def calculate_rouge(predictions: List[str], references: List[str], lang: str = "en") -> Dict[str, float]:
    # stemming is English-centric; disable for Arabic
    use_stemmer = not lang.lower().startswith("ar")
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=use_stemmer)

    r1 = r2 = rl = 0.0
    n = 0

    for hyp, ref in zip(predictions, references):
        hyp = "" if hyp is None else str(hyp)
        ref = "" if ref is None else str(ref)

        if lang.lower().startswith("ar"):
            hyp = _normalize_arabic(hyp)
            ref = _normalize_arabic(ref)

        if not hyp and not ref:
            r1 += 1.0; r2 += 1.0; rl += 1.0
            n += 1
            continue

        scores = scorer.score(ref, hyp)
        r1 += scores["rouge1"].fmeasure
        r2 += scores["rouge2"].fmeasure
        rl += scores["rougeL"].fmeasure
        n += 1

    n = n or 1
    return {"rouge1": r1 / n, "rouge2": r2 / n, "rougeL": rl / n}


def calculate_bert_score(
    predictions: List[str],
    references: List[str],
    lang: str = "en",
    model_type: Optional[str] = None,
    device: str = "cpu",
    rescale_with_baseline: bool = True,
) -> Optional[Dict[str, float]]:
    if bert_score is None:
        return None

    hyps = ["" if p is None else str(p) for p in predictions]
    refs = ["" if r is None else str(r) for r in references]

    if not hyps or not refs:
        return None

    try:
        P, R, F1 = bert_score.score(
            hyps,
            refs,
            lang=lang,
            model_type=model_type,
            device=device,
            rescale_with_baseline=rescale_with_baseline,
            verbose=False,
        )
        f1 = F1.detach().cpu().numpy()
        p = P.detach().cpu().numpy()
        r = R.detach().cpu().numpy()

        f1 = np.clip(f1, 0.0, 1.0)
        p = np.clip(p, 0.0, 1.0)
        r = np.clip(r, 0.0, 1.0)

        return {
            "bert_precision": float(p.mean()),
            "bert_recall": float(r.mean()),
            "bert_f1": float(f1.mean()),
        }
    except Exception:
        return None