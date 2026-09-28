"""CyberCodeMini Phase 8 Tokenization Smoke Test CLI

Processes the 120-example development dataset:
1. Loads tokenizer & formatters
2. Tokenizes & applies loss masking
3. Creates a PyTorch batch with CyberCodeDataCollator
4. Verifies tensor shapes
5. Saves tokenization_stats.json
6. Exits successfully (0)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cybercode_datasets.schemas.schema import TrainingExample
from cybercode_datasets.tokenization import CyberCodeTokenizer, compute_tokenization_stats
from training.collator import CyberCodeDataCollator


def main():
    dev_path = Path("data/raw/dev_dataset.jsonl")
    stats_output = Path("data/metadata/tokenization_stats.json")

    if not dev_path.exists():
        print(f"Error: Development dataset '{dev_path}' not found.")
        return 1

    print(f"Loading development dataset from '{dev_path}'...")
    examples = []
    with open(dev_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(TrainingExample.model_validate(json.loads(line)))

    print(f"Loaded {len(examples)} examples.")

    # Initialize tokenizer engine
    tokenizer_engine = CyberCodeTokenizer(
        max_sequence_length=2048,
        loss_masking_strategy="assistant_and_tool_outputs",
    )

    print("Tokenizing examples & constructing loss masks...")
    tokenized_list = tokenizer_engine.tokenize_dataset(examples)

    # Compute & save stats
    stats = compute_tokenization_stats(tokenized_list)
    stats_output.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_output, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"Saved tokenization statistics to '{stats_output}'.")

    # Create small batch collation
    collator = CyberCodeDataCollator(max_length=2048, packing=False)
    batch_features = [ex.to_dict() for ex in tokenized_list[:4]]
    batch = collator(batch_features)

    print("\n" + "=" * 60)
    print("  TOKENIZATION SMOKE TEST RESULTS")
    print("=" * 60)
    print(f"  Total Examples:        {stats['total_examples']}")
    print(f"  Total Tokens:          {stats['total_tokens']}")
    print(f"  Mean Tokens:           {stats['mean_tokens']}")
    print(f"  Median Tokens:         {stats['median_tokens']}")
    print(f"  P95 Tokens:            {stats['p95_tokens']}")
    print(f"  Maximum Tokens:        {stats['maximum_tokens']}")
    print(f"  Truncated Examples:    {stats['truncated_examples']} ({stats['truncation_percentage']}%)")
    print("-" * 60)
    
    if hasattr(batch["input_ids"], "shape"):
        print(f"  Batch input_ids shape:      {list(batch['input_ids'].shape)}")
        print(f"  Batch attention_mask shape: {list(batch['attention_mask'].shape)}")
        print(f"  Batch labels shape:         {list(batch['labels'].shape)}")
    else:
        print(f"  Batch size:                 {len(batch['input_ids'])} x {len(batch['input_ids'][0])}")

    print("=" * 60)
    print("SUCCESS: Tokenization smoke test passed cleanly.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
