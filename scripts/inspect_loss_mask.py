"""CyberCodeMini Loss Mask Visualizer CLI (Phase 8)

Prints a readable visualization of loss masking status ([LOSS] vs [MASKED]) across conversation turns.
Excludes raw secrets/sensitive tokens.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cybercode_datasets.schemas.schema import MessageRole, TrainingExample
from cybercode_datasets.tokenization import CyberCodeTokenizer


def visualize_loss_mask(example: TrainingExample, strategy: str = "assistant_and_tool_outputs"):
    tokenizer_engine = CyberCodeTokenizer(loss_masking_strategy=strategy)
    tokenized = tokenizer_engine.tokenize_example(example)

    print("\n" + "=" * 65)
    print(f"  LOSS MASK VISUALIZATION [ID: {tokenized.example_id}]")
    print(f"  Category: {tokenized.category} | Strategy: {strategy}")
    print(f"  Tokens: {tokenized.final_token_count} (Truncated: {tokenized.was_truncated})")
    print("=" * 65 + "\n")

    for i, msg in enumerate(example.messages):
        role_str = msg.role.value.upper() if hasattr(msg.role, "value") else str(msg.role).upper()
        
        # Determine status under strategy
        is_trained = False
        if strategy == "all_messages":
            is_trained = True
        elif strategy == "assistant_only":
            if msg.role == MessageRole.ASSISTANT:
                is_trained = True
        elif strategy == "assistant_and_tool_outputs":
            if msg.role in (MessageRole.ASSISTANT, MessageRole.TOOL):
                is_trained = True

        status_tag = "[LOSS]" if is_trained else "[MASKED]"
        
        # Truncate content for display safety
        content_preview = msg.content.strip().replace("\n", " ")
        if len(content_preview) > 80:
            content_preview = content_preview[:80] + "..."

        print(f"Turn #{i+1:02d} [{role_str:11s}] -> {status_tag:8s} | {content_preview}")
        if msg.tool_calls:
            tc_names = [tc.name for tc in msg.tool_calls]
            print(f"         Tool Calls: {', '.join(tc_names)}")

    print("\n" + "=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Inspect loss mask for a dataset example.")
    parser.add_argument("--input", "-i", default="data/raw/dev_dataset.jsonl", help="Dataset path")
    parser.add_argument("--example-id", "-id", default=None, help="Example source ID to inspect")
    parser.add_argument("--index", "-idx", type=int, default=0, help="Index of example if ID not specified")
    parser.add_argument("--strategy", "-s", default="assistant_and_tool_outputs", help="assistant_only | assistant_and_tool_outputs | all_messages")

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' not found.")
        return 1

    target_example = None
    with open(input_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            ex = TrainingExample.model_validate(data)
            
            if args.example_id:
                if ex.metadata.source_id == args.example_id or ex.metadata.original_id == args.example_id:
                    target_example = ex
                    break
            elif idx == args.index:
                target_example = ex
                break

    if target_example is None:
        print(f"Error: Target example not found in '{input_path}'.")
        return 1

    visualize_loss_mask(target_example, strategy=args.strategy)
    return 0


if __name__ == "__main__":
    sys.exit(main())
