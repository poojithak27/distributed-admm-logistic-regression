"""Generate synthetic CSV partitions matching the SeaWulf data format."""
import numpy as np
import pandas as pd
import os

os.makedirs("data", exist_ok=True)
np.random.seed(42)

for i in range(8):
    n = 5000
    X = np.random.randn(n, 25)
    logits = 0.3*X[:,0] + 0.1*X[:,3] + 0.25*X[:,24] + np.random.randn(n)*0.5
    y = (logits > 0).astype(int)
    df = pd.DataFrame(X, columns=[f"x{j}" for j in range(1, 26)])
    df["y"] = y
    df.to_csv(f"data/partition_{i:02d}.csv", index=False)
    print(f"Written data/partition_{i:02d}.csv  ({n} rows)")

print("Done — 8 partitions, 40,000 total rows.")
