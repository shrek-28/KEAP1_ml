import pandas as pd 

df = pd.read_csv("data/combined_scores/docking_score_data_no_outliers.csv")
cutoffs = pd.read_csv("data/new_data_pred/top_scorers_summary.csv")

for _, row in cutoffs.iterrows():

    pct = row["top_percent"]
    cutoff_score = row["cutoff_score"]

    top_df = df[df["Score"] < cutoff_score]

    print(top_df)
    top_df.to_csv(
        f"data/new_data_pred/cutoff_representatives/top_{pct}_percent.csv",
        index=False
    )