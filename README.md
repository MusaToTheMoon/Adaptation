# Language-Aware Adaptation of General-Purpose LLMs for Arabic Medical Tasks

This repository contains the experimental codebase for running and evaluating
LLM adaptation experiments on multiple tasks (mcq and answer generation),
driven by yaml configuration files.

## 1. Environment Setup

This project uses **Conda**.
**The Conda environment must be activated before running any scripts.**

### Create the environment

```bash
conda create -n adaptation python=3.10
```

### Activate the environment

```bash
conda activate adaptation
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## 2. Repository Structure 

## Repository Structure

```text
Adaptation/
├── configs/                # YAML configs per task & model
│   ├── task1/
│   │   └── gpt5.yaml
│   └── task2/
│       └── gpt5.yaml
│
├── datasets/
│   └── qa/                 # Task 1 & Task 2 datasets
│       ├── arastem.json
│       ├── medarabench.json
│       ├── medarabiq.json
│       ├── mmlu-arabic.json
│
├── evals/                  # Evaluation scripts
│   ├── evaluator.py
│   ├── metrics.py
│   └── __init__.py
│
├── models/                 # Model handlers / wrappers
│   ├── llama70.py
│   ├── meditron70_handler.py
│   ├── openai_handler.py
│   └── __init__.py
│
├── prompts/                # Prompt templates per task
│   ├── task1.txt
│   └── task2.txt
│
├── scripts/                # Entry-point scripts
│   ├── run_evaluation.py
│   ├── utils.py
│   └── __init__.py
│
├── results/
│   ├── metrics/            # Aggregated evaluation results
│   │   ├── task1/
│   │   └── task2/
│   └── predictions/        # Model outputs / predictions
│       ├── task1/
│       └── task2/
│
└── README.md
```

## 3. Running an Experiment

### Set API Keys

Make sure the required API keys are set as environment variables:

```bash
export OPENAI_API_KEY="your_openai_api_key_here"
```
#### HuggingFace

Ensure appropriate huggingface model is downloaded and present in the path pointed to by $HF_HOME env var. 

If not already present, you should run the script with offline=False the first time, allowing for the model to be cached and should run with offline=True in subsequence runs.

Alternatively, you may separately download the model into `$HF_HOME` prior to using it over hpc compute nodes. You can do this by unsetting `HF_HUB_OFFLINE`, then downloading the model, and finally resetting `HF_HUB_OFFLINE=1`.

Exact command for downloading through the terminal: `python -c "from transformers import Mistral3ForConditionalGeneration; Mistral3ForConditionalGeneration.from_pretrained('mistralai/Mistral-Small-3.2-24B-Instruct-2506', device_map='cpu')"`

### Configure the Experiment

Configuration files are stored under the `configs/` directory.
Each config defines:

* model type and model name
* dataset paths
* prompt/task settings

### Run the Experiment

Always run experiments **from the root directory of the repository**. Same command below: 

```bash
python scripts/run_evaluation.py configs/task1/gpt5.yaml
```

