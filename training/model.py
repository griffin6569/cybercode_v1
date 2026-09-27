"""CyberCodeMini Model Loader

Loads Qwen2.5-Coder-1.5B-Instruct base model with LoRA/QLoRA adapter initialization.
Performs model architecture inspection and hardware compatibility verification.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

from training.peft_config import build_lora_config, compute_trainable_parameters, validate_target_modules
from training.quantization import get_quantization_config
from training.utils import detect_hardware


@dataclass
class ModelArchitectureSummary:
    """Inspected architecture parameters of the loaded model."""

    model_name: str
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    attention_modules: list[str]
    mlp_modules: list[str]
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def inspect_model_architecture(model: Any, tokenizer: Any = None) -> ModelArchitectureSummary:
    """Inspect and extract actual model layer structure, head counts, and special tokens."""
    cfg = getattr(model, "config", None)

    vocab_size = getattr(cfg, "vocab_size", 151936)
    hidden_size = getattr(cfg, "hidden_size", 1536)
    num_layers = getattr(cfg, "num_hidden_layers", 28)
    num_heads = getattr(cfg, "num_attention_heads", 12)

    attn_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    mlp_modules = ["gate_proj", "up_proj", "down_proj"]

    eos_id = getattr(cfg, "eos_token_id", 151643)
    pad_id = getattr(cfg, "pad_token_id", 151643)

    if tokenizer is not None:
        eos_id = getattr(tokenizer, "eos_token_id", eos_id)
        pad_id = getattr(tokenizer, "pad_token_id", pad_id)

    model_name = getattr(cfg, "_name_or_path", "Qwen/Qwen2.5-Coder-1.5B-Instruct")

    return ModelArchitectureSummary(
        model_name=model_name,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        attention_modules=attn_modules,
        mlp_modules=mlp_modules,
        eos_token_id=eos_id,
        pad_token_id=pad_id,
    )


def load_cybercode_model(
    model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    mode: str = "lora",  # "full" | "lora" | "qlora"
    lora_config: Optional[dict[str, Any]] = None,
    quantization_config: Optional[dict[str, Any]] = None,
    trust_remote_code: bool = True,
    device_map: Optional[str] = "auto",
) -> tuple[Any, dict[str, Any]]:
    """Load model and apply LoRA or QLoRA wrappers if specified."""
    hw = detect_hardware()
    
    # 1. Determine quantization config if qlora requested
    is_qlora = (mode == "qlora") or (quantization_config and quantization_config.get("enabled", False))
    q_enabled = is_qlora and hw.device == "cuda"

    quant_args = quantization_config or {}
    bnb_config, quant_status = get_quantization_config(
        enabled=q_enabled,
        bits=quant_args.get("bits", 4),
        quant_type=quant_args.get("quant_type", "nf4"),
        double_quant=quant_args.get("double_quant", True),
        compute_dtype=quant_args.get("compute_dtype", "bfloat16"),
    )

    # 2. Load Base Model from Transformers
    model = None
    try:
        import torch
        from transformers import AutoModelForCausalLM

        torch_dtype = torch.bfloat16 if hw.device == "cuda" else torch.float32

        load_kwargs: dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
            "torch_dtype": torch_dtype,
        }

        if bnb_config is not None:
            load_kwargs["quantization_config"] = bnb_config
        elif hw.device == "cuda" and device_map:
            load_kwargs["device_map"] = device_map

        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    except Exception as err:
        print(f"Warning: Could not load actual Hugging Face model '{model_name}': {err}")

    # 3. Apply LoRA PEFT adapter if mode in ("lora", "qlora")
    if mode in ("lora", "qlora") and lora_config is not None and lora_config.get("enabled", True):
        target_modules = lora_config.get("target_modules") or lora_config.get("lora_target_modules")
        
        if model is not None and target_modules:
            validate_target_modules(model, target_modules)

        peft_cfg = build_lora_config(
            enabled=True,
            rank=lora_config.get("rank") or lora_config.get("lora_r", 16),
            alpha=lora_config.get("alpha") or lora_config.get("lora_alpha", 32),
            dropout=lora_config.get("dropout") or lora_config.get("lora_dropout", 0.05),
            target_modules=target_modules,
        )

        try:
            from peft import get_peft_model
            if model is not None and peft_cfg is not None:
                model = get_peft_model(model, peft_cfg)
        except Exception as peft_err:
            print(f"Warning: Failed to apply PEFT wrapper: {peft_err}")

    # 4. Summary & parameter statistics
    trainable_p, total_p, pct = compute_trainable_parameters(model)
    arch_summary = inspect_model_architecture(model)

    summary_info = {
        "model_name": model_name,
        "mode": mode,
        "trainable_params": trainable_p,
        "total_params": total_p,
        "trainable_percentage": pct,
        "quantization": quant_status.to_dict(),
        "architecture": arch_summary.to_dict(),
        "hardware": hw.to_dict(),
    }

    return model, summary_info
