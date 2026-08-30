import json
import sys
from pathlib import Path

import pandas as pd


def cm_dict_to_dataframe(cm_dict: dict) -> pd.DataFrame:
    """Convert a confusion matrix dict to a pandas DataFrame.

    Args:
        cm_dict: Dict with "actual|predicted" keys and count values.

    Returns:
        DataFrame with actual labels as rows and predicted labels as columns.
    """
    if not cm_dict:
        return pd.DataFrame()

    pairs = []
    for key, count in cm_dict.items():
        actual, predicted = key.split("|", 1)
        pairs.append((actual, predicted, count))

    all_labels = sorted({p[0] for p in pairs} | {p[1] for p in pairs})

    df = pd.DataFrame(0, index=all_labels, columns=all_labels)
    for actual, predicted, count in pairs:
        df.loc[actual, predicted] = count

    df.index.name = "Actual"
    df.columns.name = "Predicted"
    return df


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file, warning and skipping malformed lines."""
    results = []
    skipped = 0
    total_lines = 0
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            total_lines += 1
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError as e:
                skipped += 1
                print(
                    f"WARNING: Skipping malformed JSON at {path}:{line_num}: {e}",
                    file=sys.stderr,
                )
    if skipped > 0:
        skip_rate = skipped / total_lines
        print(
            f"WARNING: Skipped {skipped}/{total_lines} malformed lines "
            f"({skip_rate:.1%}) in {path}",
            file=sys.stderr,
        )
        if skip_rate > 0.05:
            raise ValueError(
                f"Too many malformed lines ({skipped}/{total_lines}, "
                f"{skip_rate:.1%}). File may be corrupted: {path}"
            )
    return results
