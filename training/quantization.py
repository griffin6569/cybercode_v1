"""CyberCodeMini Quantization Helper (QLoRA / 4-bit)

Manages Hugging Face BitsAndBytesConfig creation for QLoRA fine-tuning with graceful CPU fallback.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional


@dataclass
class QuantizationStatus:
    """Status report for model quantization."""

    enabled: bool
    bits: int  # 4 or 8
    quant_type: str  # "nf4" | "fp4"
    double_quant: bool
    compute_dtype: str
    backend: str  # "bitsandbytes" | "cpu_fallback" | "none"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def get_quantization_config(
    enabled: bool = False,
    bits: int = 4,
    quant_type: str = "nf4",
    double_quant: bool = True,
    compute_dtype: str = "bfloat16",
) -> tuple[Optional[Any], QuantizationStatus]:
    """Build BitsAndBytesConfig if available and requested; otherwise return CPU fallback status."""
    if not enabled:
        status = QuantizationStatus(
            enabled=False,
            bits=0,
            quant_type="none",
            double_quant=False,
            compute_dtype=compute_dtype,
            backend="none",
        )
        return None, status

    # Try to import bitsandbytes and transformers BitsAndBytesConfig
    try:
        import torch
        import bitsandbytes as bnb
        from transformers import BitsAndBytesConfig

        if not torch.cuda.is_available():
            print("Warning: CUDA unavailable. QLoRA quantization disabled; falling back to FP32/BF16 CPU execution.")
            status = QuantizationStatus(
                enabled=False,
                bits=bits,
                quant_type=quant_type,
                double_quant=double_quant,
                compute_dtype=compute_dtype,
                backend="cpu_fallback",
            )
            return None, status

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(compute_dtype, torch.bfloat16)

        if bits == 4:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=quant_type,
                bnb_4bit_use_double_quant=double_quant,
                bnb_4bit_compute_dtype=torch_dtype,
            )
        elif bits == 8:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            raise ValueError(f"Unsupported quantization bit width: {bits}")

        status = QuantizationStatus(
            enabled=True,
            bits=bits,
            quant_type=quant_type,
            double_quant=double_quant,
            compute_dtype=compute_dtype,
            backend="bitsandbytes",
        )
        return bnb_config, status

    except (ImportError, Exception) as err:
        print(f"Warning: bitsandbytes quantization unavailable ({err}). Falling back to unquantized training.")
        status = QuantizationStatus(
            enabled=False,
            bits=bits,
            quant_type=quant_type,
            double_quant=double_quant,
            compute_dtype=compute_dtype,
            backend="cpu_fallback",
        )
        return None, status
