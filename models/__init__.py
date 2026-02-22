from .openai_handler import OpenAIHandler
# do same here to import other model handlers !!!!!!!
from .mistral_small import MistralSmallMCQHandler #first name is name of python file and second name is that of the class from mistral.py 
from .mistral_7b import Mistral7BMCQHandler
from .llama31_8b_inst import Llama31_8BInstMCQHandler
from .llama33_70b_inst import Llama33_70BInstMCQHandler
from .deepseek32 import DeepSeek32MCQHandler
from .qwen3_235b_think import Qwen3_235BThinkMCQHandler
from .medgemma import MedGemma27BMCQHandler

import os


def load_model_handler(config):
    """
    Load the appropriate model handler based on config.
    MCQ-only setup.
    """

    model_cfg = config["model"]
    model_type = model_cfg["type"]

    if model_type == "openai":
        api_key = model_cfg.get("api_key", os.getenv("OPENAI_API_KEY"))
        if not api_key:
            raise ValueError(
                "API key is not provided in config or environment variable (OPENAI_API_KEY)."
            )

        return OpenAIHandler(
            api_key=api_key,
            model=model_cfg["name"]
        )

    elif model_type == "mistral_small":
        print(f"cache_dir={model_cfg.get('cache_dir')}")
        print(f"0 HF_HOME={os.environ.get('HF_HOME')}")

        return MistralSmallMCQHandler(
            model_name=model_cfg.get(
                "name",
                "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
            ),
            cache_dir=model_cfg.get("cache_dir"),
            offline=model_cfg.get("offline", True),
        )

    elif model_type == "mistral_7b":
        print(f"cache_dir={model_cfg.get('cache_dir')}")
        print(f"0 HF_HOME={os.environ.get('HF_HOME')}")

        return Mistral7BMCQHandler(
            model_name=model_cfg.get(
                "name",
                "mistralai/Mistral-7B-Instruct-v0.3"
            ),
            cache_dir=model_cfg.get("cache_dir"),
            offline=model_cfg.get("offline", True),
        )

    elif model_type == "llama31_8b_inst":
        print(f"cache_dir={model_cfg.get('cache_dir')}")
        print(f"0 HF_HOME={os.environ.get('HF_HOME')}")

        return Llama31_8BInstMCQHandler(
            model_name=model_cfg.get(
                "name",
                "meta-llama/Llama-3.1-8B-Instruct"
            ),
            cache_dir=model_cfg.get("cache_dir"),
            offline=model_cfg.get("offline", True),
        )

    elif model_type == "medgemma":
        return MedGemma27BMCQHandler(model_name=model_cfg.get(
            "name",
            "google/medgemma-27b-text-it"
        ),
        cache_dir=model_cfg.get("cache_dir"),
        offline=model_cfg.get("offline", True),
        )

    else:
        raise ValueError(f"Unsupported model type: {model_type}")
