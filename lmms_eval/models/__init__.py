import importlib
import os
import hf_transfer
from loguru import logger
import sys
import hf_transfer

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

logger.remove()
logger.add(sys.stdout, level="WARNING")

AVAILABLE_MODELS = {
    "streamforest": "StreamForest",

    "minicpm_v": "MiniCPM_V",
    "internvl2": "InternVL2",
    "llava_ov": "LLaVa_OV",
    
    # "longva": "LongVA",
    # "qwen2_vl": "Qwen2_VL",
    # "qwen2_5_vl": "Qwen2_5_VL",
    

}

for model_name, model_class in AVAILABLE_MODELS.items():
    try:
        exec(f"from .{model_name} import {model_class}")
    except Exception as e:
        logger.debug(f"Failed to import {model_class} from {model_name}: {e}")

if os.environ.get("LMMS_EVAL_PLUGINS", None):
    # Allow specifying other packages to import models from
    for plugin in os.environ["LMMS_EVAL_PLUGINS"].split(","):
        m = importlib.import_module(f"{plugin}.models")
        for model_name, model_class in getattr(m, "AVAILABLE_MODELS").items():
            try:
                exec(f"from {plugin}.models.{model_name} import {model_class}")
            except ImportError as e:
                logger.debug(f"Failed to import {model_class} from {model_name}: {e}")
