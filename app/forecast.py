import pandas as pd
from prophet import Prophet

# Load ML Dataset
df = pd.read_csv("ml_dataset.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# store forecasts from every hashtag
all_forecasts = []

# Train one Prophet model per #
for hashtag in df["hastag"].unique():
    print(f"Training model for {hashtag}: ")
    
    hashtag_df = (
        df[df["hastag"] == hashtag]
        [["timestamp", "trend_score"]].rename(
            columns={
                "timestamp": "ds",
                "trend_score": "y"
            }
        )
    )