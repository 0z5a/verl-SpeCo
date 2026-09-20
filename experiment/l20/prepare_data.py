"""Small deterministic arithmetic dataset for the SpeCo online loop."""

from pathlib import Path

import pandas as pd

root = Path(
    "/home/kxqandccx/0z5a/speco-l20-20260919/c5-native-latest/evidence/full-e2e"
)
rows = []
for index in range(16):
    left, right = 173 + index, 29 + index
    rows.append(
        {
            "data_source": "openai/gsm8k",
            "prompt": [
                {
                    "role": "user",
                    "content": (
                        f"Calculate {left} times {right}. Explain your calculation in two sentences, "
                        "then write #### followed by the final number."
                    ),
                }
            ],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": str(left * right)},
            "extra_info": {"split": "train", "index": index},
        }
    )
pd.DataFrame(rows).to_parquet(root / "train.parquet")
pd.DataFrame(rows[:4]).to_parquet(root / "val.parquet")
