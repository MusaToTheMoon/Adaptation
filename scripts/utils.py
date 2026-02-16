'''

This file contains utility helpers for experiment scripts: handles config loading and prediction serialization

'''
from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Dict, List, Tuple

import yaml
import pandas as pd

# Custom yaml config handler to resolve variable references and detect cycles
class ConfigError(Exception):
    pass

class YAMLConfigHandler:
    # variable for the regex
    VAR_REGEX = re.compile(r"\$\{([^}]+)\}")

    def __init__(self, max_passes: int = 10):
        self.max_passes = max_passes

    # variable lookup by dotpath, e.g. ${model.name} -> looks up root['model']['name']
    def _get_by_dotpath(self, root: Dict[str, Any], path: str) -> Any:
        cur: Any = root
        for part in path.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                raise ConfigError(f"Unknown reference '{path}' at '{part}'")
        return cur

    # variable replacement in strings
    def _replace_str(self, s: str, root: Dict[str, Any], stack: List[str]) -> Tuple[Any, bool]:
        matches = list(self.VAR_REGEX.finditer(s))
        if not matches:
            return s, False

        def resolve(token: str) -> Any:
            token = token.strip()
            if token in stack:
                cycle = " -> ".join(stack + [token])
                raise ConfigError(f"Cyclic reference detected: {cycle}")
            stack.append(token)
            ref = self._get_by_dotpath(root, token)
            val, _ = self._walk(ref, root, stack)
            stack.pop()
            return val

        # If the whole string is exactly one token, return the resolved value as is (type preserved)
        if len(matches) == 1 and matches[0].span() == (0, len(s)):
            return resolve(matches[0].group(1)), True

        # Otherwise, interpolate into a string (resolved values must be scalars)
        out = s
        for m in matches:
            token = m.group(1)
            val = resolve(token)
            if not isinstance(val, (str, int, float, bool)):
                raise ConfigError(
                    f"Cannot interpolate non-scalar for '{token.strip()}' into string: {type(val)}"
                )
            out = out.replace(m.group(0), str(val))

        return out, (out != s)

    # recursive walk to resolve variables in any nested structure
    def _walk(self, node: Any, root: Dict[str, Any], stack: List[str]) -> Tuple[Any, bool]:
        if isinstance(node, str):
            return self._replace_str(node, root, stack)

        if isinstance(node, dict):
            changed = False
            new_dict: Dict[str, Any] = {}
            for k, v in node.items():
                new_v, ch = self._walk(v, root, stack)
                new_dict[k] = new_v
                changed = changed or ch
            return new_dict, changed

        if isinstance(node, list):
            changed = False
            new_list: List[Any] = []
            for v in node:
                new_v, ch = self._walk(v, root, stack)
                new_list.append(new_v)
                changed = changed or ch
            return new_list, changed

        return node, False

    # main entry point to update config with variable resolution and cycle detection
    def update_config(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fixpoint resolver:
        - Runs up to max_passes (default 10)
        - Early stops when a full pass makes no changes
        """
        resolved = deepcopy(cfg)

        for _ in range(self.max_passes):
            resolved, changed = self._walk(resolved, resolved, stack=[])
            if not changed:
                return resolved

        raise ConfigError(
            f"Config did not reach a fixpoint after {self.max_passes} passes. "
            "Likely an indirect cycle or a reference chain longer than max_passes."
        )


# def load_config(config_path):
#     try:
#         with open(config_path, "r") as f:
#             return yaml.safe_load(f)
#     except FileNotFoundError:
#         raise FileNotFoundError(f"configuration file not found at {config_path}")
#     except yaml.YAMLError as e:
#         raise ValueError(f"error parsing yaml config file:{e}")
#     except Exception as e:
#         raise ValueError(f"unexpected error while loading config:{e}")
# updated load_config to support variable references and cycle detection with YAMLConfigHandler
def load_config(config_path: str, *, resolve: bool = True, max_passes: int = 10) -> dict:
    try:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        if cfg is None:
            raise ValueError(f"config file is empty: {config_path}")

        if not isinstance(cfg, dict):
            raise ValueError(
                f"top level YAML must be a mapping (dict), got {type(cfg).__name__}"
            )

        if resolve:
            handler = YAMLConfigHandler(max_passes=max_passes)
            cfg = handler.update_config(cfg)

        return cfg

    except FileNotFoundError:
        raise FileNotFoundError(f"configuration file not found at {config_path}")

    except yaml.YAMLError as e:
        raise ValueError(f"error parsing yaml config file: {e}")

    except ConfigError as e:
        # interpolation / cycle / unknown reference errors
        raise ValueError(f"error resolving yaml config variables: {e}")

    except Exception as e:
        raise ValueError(f"unexpected error while loading config: {e}")

def save_predictions(predictions, output_path):
    try:
        df = pd.DataFrame(predictions)
        df.to_csv(output_path, index=False)
    except Exception as e:
        raise IOError(f"failed to save predictions to {output_path}:{e}")
