from .openai_handler import OpenAIHandler
# do same here to import other model handlers !!!!!!!
from .mistral_small import MistralSmallMCQHandler #first name is name of python file and second name is that of the class from mistral.py 
from .mistral_small_chaimae import MistralSmallHandler
from .mistral_7b import Mistral7BMCQHandler
from .llama70 import Llama70MCQHandler
from .llama31_8b_inst import Llama31_8BInstMCQHandler
from .llama33_70b_inst import Llama33_70BInstMCQHandler
from .deepseek32 import DeepSeek32MCQHandler
from .qwen3_235b_think import Qwen3_235BThinkMCQHandler
from .gpt5_family import GPT5FamilyMCQHandler
from .medgemma import MedGemma27BMCQHandler
from .allam_7b_inst_prev import ALLaM7BInstPrevMCQHandler
from .aya_expanse_8b import AyaExpanse8BMCQHandler
from .gemma3_27b_it import Gemma3_27BMCQHandler
from .gemini import Gemini3ProHandler
from .falcon import FalconH1MCQHandler
from .fanar import Fanar19BMCQHandler
from .jais import Jais2ChatMCQHandler
from .med42 import Med42MCQHandler
from .meditron import Meditron3MCqHandler
from .silma import Silma9BMCQHandler
from .autocap import AutoCAPMCQHandler
from .autocap_medgemma import AutoCAPMedGemmaHandler

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
        # #print(f"0 HF_HOME={os.environ.get('HF_HOME')}")

        return MistralSmallMCQHandler(
            model_name=model_cfg.get(
                "name",
                "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
            ),
            cache_dir=model_cfg.get("cache_dir"),
            offline=model_cfg.get("offline", True),
        )
    
    elif model_type == "mistral_small_chaimae":
        print(f"cache_dir={model_cfg.get('cache_dir')}")
        # print(f"0 HF_HOME={os.environ.get('HF_HOME')}")

        return MistralSmallHandler(
            model_name=model_cfg.get(
                "name",
                "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
            ),
            cache_dir=model_cfg.get("cache_dir"),
            offline=model_cfg.get("offline", True),
        )

    elif model_type == "mistral_7b":
        print(f"cache_dir={model_cfg.get('cache_dir')}")
        # print(f"0 HF_HOME={os.environ.get('HF_HOME')}")

        return Mistral7BMCQHandler(
            model_name=model_cfg.get(
                "name",
                "mistralai/Mistral-7B-Instruct-v0.3"
            ),
            cache_dir=model_cfg.get("cache_dir"),
            offline=model_cfg.get("offline", True),
        )

    elif model_type == "falcon":
        print(f"cache_dir={model_cfg.get('cache_dir')}")
        # print(f"0 HF_HOME={os.environ.get('HF_HOME')}")

        return FalconH1MCQHandler(
            model_name=model_cfg.get(
                "name",
                "tiiuae/Falcon-H1-7B-Instruct"
            ),
            cache_dir=model_cfg.get("cache_dir"),
            offline=model_cfg.get("offline", True),
        )

    elif model_type == "fanar":
        print(f"cache_dir={model_cfg.get('cache_dir')}")
        # print(f"0 HF_HOME={os.environ.get('HF_HOME')}")

        return Fanar19BMCQHandler(
            model_name=model_cfg.get(
                "name",
                "QCRI/Fanar-1-9B"
            ),
            cache_dir=model_cfg.get("cache_dir"),
            offline=model_cfg.get("offline", True),
        )

    elif model_type == "jais":
        print(f"cache_dir={model_cfg.get('cache_dir')}")
        # print(f"0 HF_HOME={os.environ.get('HF_HOME')}")

        return Jais2ChatMCQHandler(
            model_name=model_cfg.get(
                "name",
                "inceptionai/Jais-2-8B-Chat"
            ),
            cache_dir=model_cfg.get("cache_dir"),
            offline=model_cfg.get("offline", True),
        )

    elif model_type == "med42":
        print(f"cache_dir={model_cfg.get('cache_dir')}")
        # print(f"0 HF_HOME={os.environ.get('HF_HOME')}")

        return Med42MCQHandler(
            model_name=model_cfg.get(
                "name",
                "m42-health/Llama3-Med42-70B"
            ),
            cache_dir=model_cfg.get("cache_dir"),
            offline=model_cfg.get("offline", True),
            device_map=model_cfg.get("device_map", "auto"),
            max_memory_per_gpu_gib=model_cfg.get("max_memory_per_gpu_gib"),
        )

    elif model_type == "meditron":
        print(f"cache_dir={model_cfg.get('cache_dir')}")
        # print(f"0 HF_HOME={os.environ.get('HF_HOME')}")

        return Meditron3MCqHandler(
            model_name=model_cfg.get(
                "name",
                "OpenMeditron/Meditron3-70B"
            ),
            cache_dir=model_cfg.get("cache_dir"),
            offline=model_cfg.get("offline", True),
        )

    elif model_type == "silma":
        print(f"cache_dir={model_cfg.get('cache_dir')}")
        #print(f"0 HF_HOME={os.environ.get('HF_HOME')}")

        return Silma9BMCQHandler(
            model_name=model_cfg.get(
                "name",
                "silma-ai/SILMA-9B-Instruct-v1.0"
            ),
            cache_dir=model_cfg.get("cache_dir"),
            offline=model_cfg.get("offline", True),
        )

    elif model_type == "llama70":
        print(f"cache_dir={model_cfg.get('cache_dir')}")
        #print(f"0 HF_HOME={os.environ.get('HF_HOME')}")

        return Llama70MCQHandler(
            model_name=model_cfg.get(
                "name",
                "meta-llama/Llama-3.3-70B-Instruct"
            ),
            cache_dir=model_cfg.get("cache_dir"),
        )

    elif model_type == "llama31_8b_inst":
        print(f"cache_dir={model_cfg.get('cache_dir')}")
        #print(f"0 HF_HOME={os.environ.get('HF_HOME')}")

        return Llama31_8BInstMCQHandler(
            model_name=model_cfg.get(
                "name",
                "meta-llama/Llama-3.1-8B-Instruct"
            ),
            cache_dir=model_cfg.get("cache_dir"),
            offline=model_cfg.get("offline", True),
        )

    elif model_type == "llama33_70b_inst":
        print(f"cache_dir={model_cfg.get('cache_dir')}")
        #print(f"0 HF_HOME={os.environ.get('HF_HOME')}")

        return Llama33_70BInstMCQHandler(
            model_name=model_cfg.get(
                "name",
                "meta-llama/Llama-3.3-70B-Instruct"
            ),
            cache_dir=model_cfg.get("cache_dir"),
            offline=model_cfg.get("offline", True),
            load_in_4bit=model_cfg.get("load_in_4bit", True),
        )

    elif model_type == "deepseek32":
        print(f"cache_dir={model_cfg.get('cache_dir')}")
        #print(f"0 HF_HOME={os.environ.get('HF_HOME')}")

        return DeepSeek32MCQHandler(
            model_name=model_cfg.get(
                "name",
                "deepseek-ai/DeepSeek-V3.2"
            ),
            cache_dir=model_cfg.get("cache_dir"),
            offline=model_cfg.get("offline", True),
        )

    elif model_type == "qwen3_235b_think":
        print(f"cache_dir={model_cfg.get('cache_dir')}")
        #print(f"0 HF_HOME={os.environ.get('HF_HOME')}")

        return Qwen3_235BThinkMCQHandler(
            model_name=model_cfg.get(
                "name",
                "Qwen/Qwen3-235B-A22B-Instruct-2507"
            ),
            cache_dir=model_cfg.get("cache_dir"),
            offline=model_cfg.get("offline", True),
        )

    elif model_type in ("gpt5", "gpt51", "gpt52"):
        api_key = model_cfg.get("api_key", os.getenv("OPENAI_API_KEY"))
        if not api_key:
            raise ValueError(
                "API key is not provided in config or environment variable (OPENAI_API_KEY)."
            )

        return GPT5FamilyMCQHandler(
            api_key=api_key,
            model=model_cfg.get("name", "gpt-5"),
        )

    elif model_type == "medgemma":
        return MedGemma27BMCQHandler(model_name=model_cfg.get(
            "name",
            "google/medgemma-27b-text-it"
        ),
        cache_dir=model_cfg.get("cache_dir"),
        offline=model_cfg.get("offline", True),
        )

    elif model_type == "allam_7b_inst_prev":
        print(f"cache_dir={model_cfg.get('cache_dir')}")
        #print(f"0 HF_HOME={os.environ.get('HF_HOME')}")

        return ALLaM7BInstPrevMCQHandler(
            model_name=model_cfg.get(
                "name",
                "humain-ai/ALLaM-7B-Instruct-preview"
            ),
            cache_dir=model_cfg.get("cache_dir"),
            offline=model_cfg.get("offline", True),
        )

    elif model_type == "aya_expanse_8b":
        print(f"cache_dir={model_cfg.get('cache_dir')}")
        #print(f"0 HF_HOME={os.environ.get('HF_HOME')}")

        return AyaExpanse8BMCQHandler(
            model_name=model_cfg.get(
                "name",
                "CohereLabs/aya-expanse-8b"
            ),
            cache_dir=model_cfg.get("cache_dir"),
            offline=model_cfg.get("offline", True),
        )

    elif model_type == "gemma3_27b_it":
        print(f"cache_dir={model_cfg.get('cache_dir')}")
        #print(f"0 HF_HOME={os.environ.get('HF_HOME')}")

        return Gemma3_27BMCQHandler(
            model_name=model_cfg.get(
                "name",
                "google/gemma-3-27b-it"
            ),
            cache_dir=model_cfg.get("cache_dir"),
            offline=model_cfg.get("offline", True),
        )

    elif model_type == "gemini":
        return Gemini3ProHandler(
            api_key=os.getenv("GEMINI_API_KEY"),
            model=model_cfg.get(
                "name",
                "gemini-3.1-pro-preview"
            ),
        )

    elif model_type == "autocap":
        return AutoCAPMCQHandler(
            model_name=model_cfg.get(
                "name",
                "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
            ),
            cache_dir=model_cfg.get("cache_dir"),
            offline=model_cfg.get("offline", True),
            candidate_languages=model_cfg.get("candidate_languages"),
            top_k_languages=model_cfg.get("top_k_languages", 3),
            selection_max_tokens=model_cfg.get("selection_max_tokens", 128),
            weight_max_tokens=model_cfg.get("weight_max_tokens", 128),
            cot_max_tokens=model_cfg.get("cot_max_tokens", 512),
            reasoning_max_tokens=model_cfg.get("reasoning_max_tokens"),
            do_sample=model_cfg.get("do_sample", False),
        )

    elif model_type == "autocap_medgemma":
        return AutoCAPMedGemmaHandler(
            model_name=model_cfg.get(
                "name",
                "google/medgemma-27b-text-it",
            ),
            cache_dir=model_cfg.get("cache_dir"),
            offline=model_cfg.get("offline", True),
            candidate_languages=model_cfg.get("candidate_languages"),
            top_k_languages=model_cfg.get("top_k_languages", 3),
            selection_max_tokens=model_cfg.get("selection_max_tokens", 128),
            weight_max_tokens=model_cfg.get("weight_max_tokens", 128),
            cot_max_tokens=model_cfg.get("cot_max_tokens", 512),
            reasoning_max_tokens=model_cfg.get("reasoning_max_tokens"),
            do_sample=model_cfg.get("do_sample", False),
        )

    else:
        raise ValueError(f"Unsupported model type: {model_type}")
