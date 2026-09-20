"""Small deterministic arithmetic dataset for the SpeCo online loop."""

from pathlib import Path

import pandas as pd

root = Path(
    "/home/kxqandccx/0z5a/speco-l20-20260919/c5-native-latest/evidence/full-e2e"
)
rows = []
for index in range(16):
    rows.append(
        {
            "data_source": "openai/gsm8k",
            "prompt": [
                {
                    "role": "user",
                    "content": f"What is {index} + 2? Answer with #### followed by the number.",
                }
            ],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": str(index + 2)},
            "extra_info": {"split": "train", "index": index},
        }
    )
pd.DataFrame(rows).to_parquet(root / "train.parquet")
pd.DataFrame(rows[:4]).to_parquet(root / "val.parquet")
