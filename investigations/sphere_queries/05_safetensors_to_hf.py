"""Build HF DPRContextEncoder / DPRQuestionEncoder dirs from the converted safetensors.

usage: 05_safetensors_to_hf.py DPR_WEB_SAFETENSORS OUT_DIR
Writes OUT_DIR/ctx and OUT_DIR/question. Only safetensors are read.
"""

import sys
from pathlib import Path

from safetensors.torch import load_file
from transformers import DPRConfig, DPRContextEncoder, DPRQuestionEncoder

source, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
tensors = load_file(str(source))
config = DPRConfig(projection_dim=0)
for prefix, cls, target in (("ctx_model.", DPRContextEncoder, "ctx_encoder"),
                            ("question_model.", DPRQuestionEncoder, "question_encoder")):
    model = cls(config)
    state = {f"{target}.bert_model.{k[len(prefix):]}": v
             for k, v in tensors.items() if k.startswith(prefix)}
    result = model.load_state_dict(state, strict=False)
    print(prefix, "missing:", result.missing_keys[:6], "unexpected:", result.unexpected_keys[:6])
    model.save_pretrained(out_dir / target.split("_")[0])
