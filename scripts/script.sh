# step 1: analysis of representative data

# data cleaning - removal of all NaN scores
# produces CSV file with all clean scores + CSV with all NaN and failed values 
cd /Users/shreyasree/Documents/GitHub/KEAP1_ml/data
mkdir combined_scores
cd /Users/shreyasree/Documents/GitHub/KEAP1_ml
python3 scripts/representative_analysis/data_cleaning.py --input data/docking_scores_representatives --output data/combined_scores/combined_valid_scores.csv --failed data/combined_scores/combined_failed_scores.csv

# data cleaning - representative data 
# generates a file of combined cluster representative details and its docking score 
python3 scripts/representative_analysis/representative_analysis.py --reps data/cluster_representatives_7_per_cluster.csv --scores data/combined_scores/combined_valid_scores.csv --output data/representative_docking_scores

# generates a histogram of the distribution of docking scores
Rscript scripts/representative_analysis/visualization/docking_score_dist.R data/combined_scores/combined_valid_scores.csv Score plots/representative_analysis/docking_score_histogram.pdf 0.5

# generating a boxplot of distribution of representative analysis scores 
Rscript scripts/representative_analysis/visualization/representative_analysis_dist.R data/representative_docking_scores plots/representative_analysis/representative_score_dist_analysis.pdf "Docking Affinity Distribution of Representative Molecules against KEAP1"

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# STEP 2: FEATURE ENGINEERING

cd /Users/shreyasree/Documents/GitHub/KEAP1_ml/data/engineered_features
mkdir merged

cd /Users/shreyasree/Documents/GitHub/KEAP1_ml
# data cleaning (outlier removal)
python3 scripts/feature_engineering/data_cleaning/outlier_removal.py --input data/combined_scores/combined_valid_scores.csv --col_name Score --output data/combined_scores/docking_score_data_no_outliers.csv
# combining descriptors and docking score valeus 
python3 scripts/feature_engineering/data_cleaning/scores_and_descs_combined.py --descriptors data/final_complete_descriptor_matrix.csv --scores data/combined_scores/docking_score_data_no_outliers.csv --output_1 data/combined_scores/with_descriptors.csv --output_2 data/engineered_features/with_descriptors.csv
# feature engineering (interaction, ratios, transformations, merged datasets)
# interactions
python3 scripts/feature_engineering/engineered_features/interaction_feats.py --input data/engineered_features/with_descriptors.csv --output data/engineered_features/descriptor_interactions.csv
# ratios
python3 scripts/feature_engineering/engineered_features/ratios.py --input data/engineered_features/with_descriptors.csv --output data/engineered_features/descriptor_ratios_both_directions_1.csv
# transformations
python3 scripts/feature_engineering/engineered_features/transformations.py --input data/engineered_features/with_descriptors.csv --output data/engineered_features/descriptor_transformations_1.csv

# merging features 
# descriptors + interactions
python3 scripts/feature_engineering/engineered_features/mergers.py --input data/engineered_features/with_descriptors.csv data/engineered_features/descriptor_interactions.csv --output data/engineered_features/merged/raw_descriptors_and_interactions.csv
# descriptors + ratios
python3 scripts/feature_engineering/engineered_features/mergers.py --input data/engineered_features/with_descriptors.csv data/engineered_features/descriptor_ratios_both_directions.csv --output data/engineered_features/merged/raw_descriptors_and_ratios.csv
# descriptors + transformations
python3 scripts/feature_engineering/engineered_features/mergers.py --input data/engineered_features/with_descriptors.csv data/engineered_features/descriptor_transformations.csv --output data/engineered_features/merged/raw_descriptors_and_transformations.csv
# transformations + interactions
python3 scripts/feature_engineering/engineered_features/mergers.py --input data/engineered_features/descriptor_transformations.csv data/engineered_features/descriptor_interactions.csv --output data/engineered_features/merged/transformations_and_interactions.csv
# transformations + ratios
python3 scripts/feature_engineering/engineered_features/mergers.py --input data/engineered_features/descriptor_transformations.csv data/engineered_features/descriptor_ratios_both_directions.csv --output data/engineered_features/merged/transformations_and_ratios.csv
# interactions + ratios
python3 scripts/feature_engineering/engineered_features/mergers.py --input data/engineered_features/descriptor_interactions.csv data/engineered_features/descriptor_ratios_both_directions.csv --output data/engineered_features/merged/interactions_and_ratios.csv
# all 4 combined 
python3 scripts/feature_engineering/engineered_features/mergers.py --input data/engineered_features/descriptor_interactions.csv data/engineered_features/descriptor_ratios_both_directions.csv data/engineered_features/descriptor_transformations.csv data/engineered_features/with_descriptors.csv --output data/engineered_features/merged/all_4_combined.csv

# reducing features (removal with spearman correlation < 0.2)
cd /Users/shreyasree/Documents/GitHub/KEAP1_ml
python3 scripts/feature_engineering/feature_selection_and_validation/spearman_filter.py --input data/engineered_features/with_descriptors.csv --output data/spearman_reduced_features/descriptors_only.csv
python3 scripts/feature_engineering/feature_selection_and_validation/spearman_filter.py --input data/engineered_features/descriptor_ratios_both_directions.csv --output data/spearman_reduced_features/ratios_only.csv
python3 scripts/feature_engineering/feature_selection_and_validation/spearman_filter.py --input data/engineered_features/descriptor_interactions.csv --output data/spearman_reduced_features/interactions_only.csv
python3 scripts/feature_engineering/feature_selection_and_validation/spearman_filter.py --input data/engineered_features/descriptor_transformations.csv --output data/spearman_reduced_features/transformations_only.csv
python3 scripts/feature_engineering/feature_selection_and_validation/spearman_filter.py --input data/engineered_features/merged/all_4_combined.csv --output data/spearman_reduced_features/all_4_combined.csv
python3 scripts/feature_engineering/feature_selection_and_validation/spearman_filter.py --input data/engineered_features/merged/interactions_and_ratios.csv --output data/spearman_reduced_features/interactions_and_ratios.csv
python3 scripts/feature_engineering/feature_selection_and_validation/spearman_filter.py --input data/engineered_features/merged/raw_descriptors_and_interactions.csv --output data/spearman_reduced_features/raw_descs_and_interactions.csv
python3 scripts/feature_engineering/feature_selection_and_validation/spearman_filter.py --input data/engineered_features/merged/raw_descriptors_and_ratios.csv --output data/spearman_reduced_features/raw_descs_and_ratios.csv
python3 scripts/feature_engineering/feature_selection_and_validation/spearman_filter.py --input data/engineered_features/merged/raw_descriptors_and_transformations.csv --output data/spearman_reduced_features/raw_descs_and_transforms.csv
python3 scripts/feature_engineering/feature_selection_and_validation/spearman_filter.py --input data/engineered_features/merged/transformations_and_interactions.csv --output data/spearman_reduced_features/transforms_and_interactions.csv
python3 scripts/feature_engineering/feature_selection_and_validation/spearman_filter.py --input data/engineered_features/merged/transformations_and_ratios.csv --output data/spearman_reduced_features/transforms_and_ratios.csv

# RMSE - identification of which is the best dataset 
cd /Users/shreyasree/Documents/GitHub/KEAP1_ml
python3 scripts/feature_engineering/feature_selection_and_validation/final_feature_selection.py --input data/spearman_reduced_features --output data/all_datasets_rmse.csv --logs data/rf_progress_log.csv

# getting the features from RMSE table  - splitting RMSE results by dataset
python3 scripts/feature_engineering/feature_selection_and_validation/feature_val_split.py --input data/all_datasets_rmse.csv --output data/rmse_split_results

# selecting best number of features (k) using Kneedle technique
python3 scripts/feature_engineering/feature_selection_and_validation/best_rmse_curve.py --input data/rmse_split_results --output data/knee_detection_results.csv

# getting best features 
python3 scripts/feature_engineering/feature_selection_and_validation/feature_list_set_generation.py --features data/all_datasets_rmse.csv --best-k data/knee_detection_results.csv --output data/final_feature_list.csv

# getting datasets with best features 
python3 scripts/feature_engineering/feature_selection_and_validation/final_dataset_generation.py --mapping data/final_feature_list.csv --input data/spearman_reduced_features --output data/final_features_data

# generating lineplots of best RMSE and best K value 
Rscript scripts/feature_engineering/feature_selection_and_validation/visualization/best_rmse_k_lineplot.R data/knee_detection_results.csv plots/rmse_analysis/best_rmse.pdf plots/rmse_analysis/best_k.pdf

# best features lineplot 
Rscript scripts/feature_engineering/feature_selection_and_validation/visualization/rmse_n_features.R data/rmse_split_results/all_4_combined.csv plots/rmse_n_features/all_4_combined.pdf "All 4 Feature Sets"
Rscript scripts/feature_engineering/feature_selection_and_validation/visualization/rmse_n_features.R data/rmse_split_results/descriptors_only.csv plots/rmse_n_features/descriptors_only.pdf "Descriptors Only"
Rscript scripts/feature_engineering/feature_selection_and_validation/visualization/rmse_n_features.R data/rmse_split_results/interactions_and_ratios.csv plots/rmse_n_features/interactions_and_ratios.pdf "Interactions and Ratios"
Rscript scripts/feature_engineering/feature_selection_and_validation/visualization/rmse_n_features.R data/rmse_split_results/interactions_only.csv plots/rmse_n_features/interactions_only.pdf "Interactions Only" 
Rscript scripts/feature_engineering/feature_selection_and_validation/visualization/rmse_n_features.R data/rmse_split_results/ratios_only.csv plots/rmse_n_features/ratios_only.pdf "Ratios Only"
Rscript scripts/feature_engineering/feature_selection_and_validation/visualization/rmse_n_features.R data/rmse_split_results/raw_descs_and_interactions.csv plots/rmse_n_features/raw_descs_and_interactions.pdf "Descriptors and Interactions" 
Rscript scripts/feature_engineering/feature_selection_and_validation/visualization/rmse_n_features.R data/rmse_split_results/raw_descs_and_ratios.csv plots/rmse_n_features/raw_descs_and_ratios.pdf "Descriptors and Ratios"
Rscript scripts/feature_engineering/feature_selection_and_validation/visualization/rmse_n_features.R data/rmse_split_results/raw_descs_and_transforms.csv plots/rmse_n_features/raw_descs_and_transforms.pdf "Descriptors and Transformations"
Rscript scripts/feature_engineering/feature_selection_and_validation/visualization/rmse_n_features.R data/rmse_split_results/transformations_only.csv plots/rmse_n_features/transforms_only.pdf "Transformations Only" 
Rscript scripts/feature_engineering/feature_selection_and_validation/visualization/rmse_n_features.R data/rmse_split_results/transforms_and_interactions.csv plots/rmse_n_features/transforms_and_interactions.pdf "Transformations and Interactions"
Rscript scripts/feature_engineering/feature_selection_and_validation/visualization/rmse_n_features.R data/rmse_split_results/transforms_and_ratios.csv plots/rmse_n_features/transforms_and_ratios.pdf "Transformations and Ratios" 

