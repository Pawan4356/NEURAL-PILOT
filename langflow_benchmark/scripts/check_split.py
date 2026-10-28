"""Print the holdout every system must use, so you can compare it with what
the Langflow agent / Claude print in step 2.

    python scripts/check_split.py telco_churn

With --neuralpilot it also runs NeuralPilot's own loader and splitter
(read-only, nothing is written) and checks that the test rows are identical.
That mode needs NeuralPilot's environment:

    ..\\.venv\\Scripts\\python.exe scripts/check_split.py telco_churn --neuralpilot   (Windows)
    ../.venv/bin/python scripts/check_split.py telco_churn --neuralpilot            (Ubuntu)
"""

from __future__ import annotations

import argparse
import sys

from common import PROJECT_ROOT, get_dataset, holdout_split, load_like_neuralpilot


def neuralpilot_test_index(entry: dict):
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from neuralpilot.engine import splits
    from neuralpilot.engine.loader import load_dataset
    from neuralpilot.schemas import RequirementSpec

    df = load_dataset(entry["abs_path"]).df
    data = df.dropna(subset=[entry["target"]])
    requirement = RequirementSpec.model_construct(
        target_column=entry["target"], problem_type=entry["problem_type"], time_column=None, group_column=None
    )
    spec = splits.build_split_spec(data, requirement)
    return splits.make_holdout_split(data, data[entry["target"]], spec).test_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--neuralpilot", action="store_true", help="cross-check with NeuralPilot's own code")
    args = parser.parse_args()

    entry = get_dataset(args.dataset)
    df, notes = load_like_neuralpilot(entry["abs_path"], entry["target"])
    train_idx, test_idx = holdout_split(df, entry["target"], entry["problem_type"])

    print(f"dataset     : {entry['name']}  ({entry['abs_path']})")
    print(f"rows used   : {len(df)}  columns: {df.shape[1]}")
    for note in notes:
        print(f"note        : {note}")
    print(f"train rows  : {len(train_idx)}")
    print(f"test rows   : {len(test_idx)}")
    print(f"first 5 IDs : {list(map(int, test_idx[:5]))}")

    if args.neuralpilot:
        np_test = neuralpilot_test_index(entry)
        same = list(np_test) == list(test_idx)
        print(f"NeuralPilot : {len(np_test)} test rows, first 5 {list(map(int, np_test[:5]))}")
        print("MATCH: identical holdout rows" if same else "MISMATCH: holdout rows differ")
        if not same:
            sys.exit(1)


if __name__ == "__main__":
    main()
