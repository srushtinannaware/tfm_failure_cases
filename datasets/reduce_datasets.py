import pandas as pd
from pathlib import Path

TRAIN_N, TEST_N = 1000, 200

# run this from inside the "datasets" folder
base = Path(__file__).parent

for folder_name in ["original_dataset", "tweaked_dataset"]:
    in_dir = base / folder_name
    out_dir = base / f"{folder_name}_small"
    out_dir.mkdir(exist_ok=True)

    for csv in in_dir.glob("*.csv"):
        df = pd.read_csv(csv).sample(frac=1, random_state=42).reset_index(drop=True)
        train, test = df.iloc[:TRAIN_N], df.iloc[TRAIN_N:TRAIN_N + TEST_N]
        train.to_csv(out_dir / f"{csv.stem}_train.csv", index=False)
        test.to_csv(out_dir / f"{csv.stem}_test.csv", index=False)
        print(f"{folder_name}/{csv.name}: {len(df)} -> {len(train)} train / {len(test)} test")