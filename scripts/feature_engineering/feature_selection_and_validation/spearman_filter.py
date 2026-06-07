import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

import pandas as pd
import numpy as np
import os

def reduce_features_by_score_correlation(
    input_file,
    output_file,
    summary_log_file="data/spearman_feature_selection_log.csv",
    identifier_col="identifier",
    score_col="Score",
    threshold=0.2
):
    # Load data
    df = pd.read_csv(input_file)

    # Numeric columns only
    numeric_df = df.select_dtypes(include=[np.number])

    # Ensure score exists
    assert score_col in numeric_df.columns

    score_series = numeric_df[score_col]

    selected_features = []

    for col in numeric_df.columns:
        if col == score_col:
            continue

        corr = numeric_df[col].corr(score_series, method="spearman")

        if not np.isnan(corr) and abs(corr) > threshold:
            selected_features.append(col)

    # Build reduced dataframe
    cols_to_keep = []

    if identifier_col in df.columns:
        cols_to_keep.append(identifier_col)

    cols_to_keep.extend(selected_features)
    cols_to_keep.append(score_col)

    reduced_df = df[cols_to_keep]
    reduced_df.to_csv(output_file, index=False)

    # ---------------------------
    # LOGGING SECTION (NEW)
    # ---------------------------
    os.makedirs(os.path.dirname(summary_log_file), exist_ok=True)

    log_entry = pd.DataFrame([{
        "input_file": input_file,
        "num_selected_features": len(selected_features),
        "selected_features": ";".join(selected_features)
    }])

    if os.path.exists(summary_log_file):
        log_entry.to_csv(summary_log_file, mode="a", header=False, index=False)
    else:
        log_entry.to_csv(summary_log_file, index=False)

    print(f"Selected {len(selected_features)} features from {input_file}")
    print(f"Output shape: {reduced_df.shape}")

    return reduced_df, selected_features
# only one use data  
reduce_features_by_score_correlation(
    input_file="data/engineered_features/with_descriptors.csv", 
    output_file="data/spearman_reduced_features/descriptors_only.csv"
)
reduce_features_by_score_correlation(
    input_file="/Users/shreyasree/Documents/GitHub/KEAP1drugdiscovery/data/engineered_features/descriptor_ratios_both_directions.csv",
    output_file="data/spearman_reduced_features/ratios_only.csv"
)
reduce_features_by_score_correlation(
    input_file="data/engineered_features/descriptor_transformations.csv", 
    output_file="data/spearman_reduced_features/transformations_only.csv"
)
reduce_features_by_score_correlation(
    input_file="data/engineered_features/descriptor_interactions.csv", 
    output_file="data/spearman_reduced_features/interactions_only.csv"
)

# merged datasets 
reduce_features_by_score_correlation(
    input_file="data/engineered_features/merged/raw_descriptors_and_transformations.csv", 
    output_file="data/spearman_reduced_features/raw_descs_and_transforms.csv"
)
reduce_features_by_score_correlation(
    input_file="data/engineered_features/merged/raw_descriptors_and_interactions.csv", 
    output_file="data/spearman_reduced_features/raw_descs_and_interactions.csv"
)
reduce_features_by_score_correlation(
    input_file="data/engineered_features/merged/raw_descriptors_and_ratios.csv", 
    output_file="data/spearman_reduced_features/raw_descs_and_ratios.csv"
)
reduce_features_by_score_correlation(
    input_file="data/engineered_features/merged/transformations_and_ratios.csv",
    output_file="data/spearman_reduced_features/transforms_and_ratios.csv"
)
reduce_features_by_score_correlation(
    input_file="data/engineered_features/merged/transformations_and_interactions.csv",
    output_file="data/spearman_reduced_features/transforms_and_interactions.csv"
)
reduce_features_by_score_correlation(
    input_file="data/engineered_features/merged/interactions_and_ratios.csv",
    output_file="data/spearman_reduced_features/interactions_and_ratios.csv"
)
reduce_features_by_score_correlation(
    input_file="data/engineered_features/merged/all_4_combined.csv",
    output_file="data/spearman_reduced_features/all_4_combined.csv"
)