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

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# STEP 3: REGRESSION MODELLING 

# linear regression
python3 scripts/regression_modelling_primary/lin_reg.py --input data/final_features_data --output data/regression_results/linear_regression_results.csv
# lasso regression
python3 scripts/regression_modelling_primary/lasso_regression.py --input data/final_features_data --output data/regression_results/lasso_regression_results.csv
# ridge regression
python3 scripts/regression_modelling_primary/ridge_regression.py --input data/final_features_data --output data/regression_results/ridge_regression_results.csv
# elastic net regression
python3 scripts/regression_modelling_primary/elastic_net.py --input data/final_features_data --output data/regression_results/elastic_net_regression_results.csv
# polynomial regression 
python3 scripts/regression_modelling_primary/polynomial_regression.py --input data/final_features_data --output data/regression_results/poly_results.csv
# KNN regression 
python3 scripts/regression_modelling_primary/knn_regressor.py --input data/final_features_data --output data/regression_results/knn_regressor_results.csv
# support vector regression 
python3 scripts/regression_modelling_primary/support_vector_regressor.py --input data/final_features_data --output data/regression_results/svr_results.csv
# decision tree regression 
python3 scripts/regression_modelling_primary/decision_tree_regressor.py --input data/final_features_data --output data/regression_results/decision_tree_results.csv
# random forest regression 
python3 scripts/regression_modelling_primary/random_forest.py --input data/final_features_data --output data/regression_results/random_forest_results.csv
# xgboost regression 
python3 scripts/regression_modelling_primary/xgboost_regressor.py --input data/final_features_data --output data/regression_results/xgboost_results.csv
# gradient boosting regression 
python3 scripts/regression_modelling_primary/gradient_boosting.py --input data/final_features_data --output data/regression_results/gbr_results.csv
# light GBM regression 
python3 scripts/regression_modelling_primary/lightgbm_regressor.py --input data/final_features_data --output data/regression_results/lightgbm_results.csv
# catboost regression
python3 scripts/regression_modelling_primary/catboost_regressor.py --input data/final_features_data --output data/regression_results/catboost_results.csv
# adaboost regression 
python3 scripts/regression_modelling_primary/adaboost_regressor.py --input data/final_features_data --output data/regression_results/adaboost_results.csv
# stacking regression 
python3 scripts/regression_modelling_primary/stacked_model.py --input data/final_features_data --output data/regression_results/stack_results.csv
# voting regressor 
python3 scripts/regression_modelling_primary/voting_regressor.py --input data/final_features_data --output data/regression_results/voting_results.csv

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# STEP 4: REGRESSION RESULT ANALYSIS  

# combining results 
python3 scripts/regression_result_analysis/result_combiner.py --input data/regression_results --output data/regression_result_analysis/combined_results.csv

# friedman statistical test
python3 scripts/regression_result_analysis/friedman_test.py --input data/regression_result_analysis/combined_results.csv --output data/stat_tests
 
# nemenyi-posthoc test 
python3 scripts/regression_result_analysis/nemenyi_posthoc.py --input data/regression_result_analysis/combined_results.csv --output data/stat_tests

# visualizations 
# model wise analysis 
Rscript scripts/regression_result_analysis/visualization/model_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/model_wise_plots/linear_reg/rmse.pdf --model linear_regression --metric RMSE --title "Linear Regression RMSE Across Feature Sets"
Rscript scripts/regression_result_analysis/visualization/model_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/model_wise_plots/linear_reg/mae.pdf --model linear_regression --metric MAE --title "Linear Regression MAE Across Feature Sets"
Rscript scripts/regression_result_analysis/visualization/model_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/model_wise_plots/linear_reg/mape.pdf --model linear_regression --metric MAPE --title "Linear Regression MAPE Across Feature Sets"
Rscript scripts/regression_result_analysis/visualization/model_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/model_wise_plots/linear_reg/r2.pdf --model linear_regression --metric R2 --title "Linear Regression R2 Across Feature Sets"

Rscript scripts/regression_result_analysis/visualization/model_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/model_wise_plots/ridge_reg/rmse.pdf --model ridge_regression --metric RMSE --title "Ridge Regression RMSE Across Feature Sets"
Rscript scripts/regression_result_analysis/visualization/model_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/model_wise_plots/ridge_reg/mae.pdf --model ridge_regression --metric MAE --title "Ridge Regression MAE Across Feature Sets"
Rscript scripts/regression_result_analysis/visualization/model_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/model_wise_plots/ridge_reg/mape.pdf --model ridge_regression --metric MAPE --title "Ridge Regression MAPE Across Feature Sets"
Rscript scripts/regression_result_analysis/visualization/model_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/model_wise_plots/ridge_reg/r2.pdf --model ridge_regression --metric R2 --title "Ridge Regression R2 Across Feature Sets"

Rscript scripts/regression_result_analysis/visualization/model_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/model_wise_plots/lasso_reg/rmse.pdf --model lasso_regression --metric RMSE --title "Lasso Regression RMSE Across Feature Sets"
Rscript scripts/regression_result_analysis/visualization/model_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/model_wise_plots/lasso_reg/mae.pdf --model lasso_regression --metric MAE --title "Lasso Regression MAE Across Feature Sets"
Rscript scripts/regression_result_analysis/visualization/model_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/model_wise_plots/lasso_reg/mape.pdf --model lasso_regression --metric MAPE --title "Lasso Regression MAPE Across Feature Sets"
Rscript scripts/regression_result_analysis/visualization/model_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/model_wise_plots/lasso_reg/r2.pdf --model lasso_regression --metric R2 --title "Lasso Regression R2 Across Feature Sets"

Rscript scripts/regression_result_analysis/visualization/model_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/model_wise_plots/elasticnet_reg/rmse.pdf --model elastic_net_regression --metric RMSE --title "Elastic Net Regression RMSE Across Feature Sets"
Rscript scripts/regression_result_analysis/visualization/model_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/model_wise_plots/elasticnet_reg/mae.pdf --model elastic_net_regression --metric MAE --title "Elastic Net Regression MAE Across Feature Sets"
Rscript scripts/regression_result_analysis/visualization/model_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/model_wise_plots/elasticnet_reg/mape.pdf --model elastic_net_regression --metric MAPE --title "Elastic Net Regression MAPE Across Feature Sets"
Rscript scripts/regression_result_analysis/visualization/model_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/model_wise_plots/elasticnet_reg/r2.pdf --model elastic_net_regression --metric R2 --title "Elastic Net Regression R2 Across Feature Sets"

Rscript scripts/regression_result_analysis/visualization/model_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/model_wise_plots/poly_reg/rmse.pdf --model poly --metric RMSE --title "Polynomial Regression RMSE Across Feature Sets"
Rscript scripts/regression_result_analysis/visualization/model_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/model_wise_plots/poly_reg/mae.pdf --model poly --metric MAE --title "Polynomial Regression MAE Across Feature Sets"
Rscript scripts/regression_result_analysis/visualization/model_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/model_wise_plots/poly_reg/mape.pdf --model poly --metric MAPE --title "Polynomial Regression MAPE Across Feature Sets"
Rscript scripts/regression_result_analysis/visualization/model_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/model_wise_plots/poly_reg/r2.pdf --model poly --metric R2 --title "Polynomial Regression R2 Across Feature Sets"

Rscript scripts/regression_result_analysis/visualization/model_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/model_wise_plots/knn_reg/rmse.pdf --model knn_regressor --metric RMSE --title "K-Nearest Neighbours Regression RMSE Across Feature Sets"
Rscript scripts/regression_result_analysis/visualization/model_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/model_wise_plots/knn_reg/mae.pdf --model knn_regressor --metric MAE --title "K-Nearest Neighbours Regression MAE Across Feature Sets"
Rscript scripts/regression_result_analysis/visualization/model_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/model_wise_plots/knn_reg/mape.pdf --model knn_regressor --metric MAPE --title "K-Nearest Neighbours Regression MAPE Across Feature Sets"
Rscript scripts/regression_result_analysis/visualization/model_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/model_wise_plots/knn_reg/r2.pdf --model knn_regressor --metric R2 --title "K-Nearest Neighbours Regression R2 Across Feature Sets"

Rscript scripts/regression_result_analysis/visualization/model_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/model_wise_plots/svm_reg/rmse.pdf --model svr --metric RMSE --title "Support Vector Regression RMSE Across Feature Sets"
Rscript scripts/regression_result_analysis/visualization/model_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/model_wise_plots/svm_reg/mae.pdf --model svr --metric MAE --title "Support Vector Regression MAE Across Feature Sets"
Rscript scripts/regression_result_analysis/visualization/model_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/model_wise_plots/svm_reg/mape.pdf --model svr --metric MAPE --title "Support Vector Regression MAPE Across Feature Sets"
Rscript scripts/regression_result_analysis/visualization/model_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/model_wise_plots/svm_reg/r2.pdf --model svr --metric R2 --title "Support Vector Regression R2 Across Feature Sets"

Rscript scripts/regression_result_analysis/visualization/model_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/model_wise_plots/decision_tree_reg/rmse.pdf --model decision_tree --metric RMSE --title "Decision Tree Regression RMSE Across Feature Sets"
Rscript scripts/regression_result_analysis/visualization/model_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/model_wise_plots/decision_tree_reg/mae.pdf --model decision_tree --metric MAE --title "Decision Tree Regression MAE Across Feature Sets"
Rscript scripts/regression_result_analysis/visualization/model_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/model_wise_plots/decision_tree_reg/mape.pdf --model decision_tree --metric MAPE --title "Decision Tree Regression MAPE Across Feature Sets"
Rscript scripts/regression_result_analysis/visualization/model_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/model_wise_plots/decision_tree_reg/r2.pdf --model decision_tree --metric R2 --title "Decision Tree Regression R2 Across Feature Sets"

Rscript scripts/regression_result_analysis/visualization/model_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/model_wise_plots/random_forest_reg/rmse.pdf --model random_forest --metric RMSE --title "Random Forest Regression RMSE Across Feature Sets"
Rscript scripts/regression_result_analysis/visualization/model_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/model_wise_plots/random_forest_reg/mae.pdf --model random_forest --metric MAE --title "Random Forest Regression MAE Across Feature Sets"
Rscript scripts/regression_result_analysis/visualization/model_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/model_wise_plots/random_forest_reg/mape.pdf --model random_forest --metric MAPE --title "Random Forest Regression MAPE Across Feature Sets"
Rscript scripts/regression_result_analysis/visualization/model_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/model_wise_plots/random_forest_reg/r2.pdf --model random_forest --metric R2 --title "Random Forest Regression R2 Across Feature Sets"

Rscript scripts/regression_result_analysis/visualization/model_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/model_wise_plots/xgboost_reg/rmse.pdf --model xgboost --metric RMSE --title "XGBoost Regression RMSE Across Feature Sets"
Rscript scripts/regression_result_analysis/visualization/model_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/model_wise_plots/xgboost_reg/mae.pdf --model xgboost --metric MAE --title "XGBoost Regression MAE Across Feature Sets"
Rscript scripts/regression_result_analysis/visualization/model_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/model_wise_plots/xgboost_reg/mape.pdf --model xgboost --metric MAPE --title "XGBoost Regression MAPE Across Feature Sets"
Rscript scripts/regression_result_analysis/visualization/model_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/model_wise_plots/xgboost_reg/r2.pdf --model xgboost --metric R2 --title "XGBoost Regression R2 Across Feature Sets"

Rscript scripts/regression_result_analysis/visualization/model_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/model_wise_plots/catboost_reg/rmse.pdf --model catboost --metric RMSE --title "CatBoost Regression RMSE Across Feature Sets"
Rscript scripts/regression_result_analysis/visualization/model_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/model_wise_plots/catboost_reg/mae.pdf --model catboost --metric MAE --title "CatBoost Regression MAE Across Feature Sets"
Rscript scripts/regression_result_analysis/visualization/model_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/model_wise_plots/catboost_reg/mape.pdf --model catboost --metric MAPE --title "CatBoost Regression MAPE Across Feature Sets"
Rscript scripts/regression_result_analysis/visualization/model_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/model_wise_plots/catboost_reg/r2.pdf --model catboost --metric R2 --title "CatBoost Regression R2 Across Feature Sets"

Rscript scripts/regression_result_analysis/visualization/model_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/model_wise_plots/adaboost_reg/rmse.pdf --model adaboost --metric RMSE --title "AdaBoost Regression RMSE Across Feature Sets"
Rscript scripts/regression_result_analysis/visualization/model_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/model_wise_plots/adaboost_reg/mae.pdf --model adaboost --metric MAE --title "AdaBoost Regression MAE Across Feature Sets"
Rscript scripts/regression_result_analysis/visualization/model_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/model_wise_plots/adaboost_reg/mape.pdf --model adaboost --metric MAPE --title "AdaBoost Regression MAPE Across Feature Sets"
Rscript scripts/regression_result_analysis/visualization/model_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/model_wise_plots/adaboost_reg/r2.pdf --model adaboost --metric R2 --title "AdaBoost Regression R2 Across Feature Sets"

Rscript scripts/regression_result_analysis/visualization/model_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/model_wise_plots/lightgbm_reg/rmse.pdf --model lightgbm --metric RMSE --title "LightGBM Regression RMSE Across Feature Sets"
Rscript scripts/regression_result_analysis/visualization/model_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/model_wise_plots/lightgbm_reg/mae.pdf --model lightgbm --metric MAE --title "LightGBM Regression MAE Across Feature Sets"
Rscript scripts/regression_result_analysis/visualization/model_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/model_wise_plots/lightgbm_reg/mape.pdf --model lightgbm --metric MAPE --title "LightGBM Regression MAPE Across Feature Sets"
Rscript scripts/regression_result_analysis/visualization/model_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/model_wise_plots/lightgbm_reg/r2.pdf --model lightgbm --metric R2 --title "LightGBM Regression R2 Across Feature Sets"

Rscript scripts/regression_result_analysis/visualization/model_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/model_wise_plots/grad_boost_reg/rmse.pdf --model gbr --metric RMSE --title "Gradient Boosting Regression RMSE Across Feature Sets"
Rscript scripts/regression_result_analysis/visualization/model_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/model_wise_plots/grad_boost_reg/mae.pdf --model gbr --metric MAE --title "Gradient Boosting Regression MAE Across Feature Sets"
Rscript scripts/regression_result_analysis/visualization/model_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/model_wise_plots/grad_boost_reg/mape.pdf --model gbr --metric MAPE --title "Gradient Boosting Regression MAPE Across Feature Sets"
Rscript scripts/regression_result_analysis/visualization/model_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/model_wise_plots/grad_boost_reg/r2.pdf --model gbr --metric R2 --title "Gradient Boosting Regression R2 Across Feature Sets"

Rscript scripts/regression_result_analysis/visualization/model_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/model_wise_plots/stacking_reg/rmse.pdf --model stack --metric RMSE --title "Stacking Regression RMSE Across Feature Sets"
Rscript scripts/regression_result_analysis/visualization/model_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/model_wise_plots/stacking_reg/mae.pdf --model stack --metric MAE --title "Stacking Regression MAE Across Feature Sets"
Rscript scripts/regression_result_analysis/visualization/model_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/model_wise_plots/stacking_reg/mape.pdf --model stack --metric MAPE --title "Stacking Regression MAPE Across Feature Sets"
Rscript scripts/regression_result_analysis/visualization/model_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/model_wise_plots/stacking_reg/r2.pdf --model stack --metric R2 --title "Stacking Regression R2 Across Feature Sets"

Rscript scripts/regression_result_analysis/visualization/model_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/model_wise_plots/voting_reg/rmse.pdf --model voting --metric RMSE --title "Voting Regression RMSE Across Feature Sets"
Rscript scripts/regression_result_analysis/visualization/model_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/model_wise_plots/voting_reg/mae.pdf --model voting --metric MAE --title "Voting Regression MAE Across Feature Sets"
Rscript scripts/regression_result_analysis/visualization/model_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/model_wise_plots/voting_reg/mape.pdf --model voting --metric MAPE --title "Voting Regression MAPE Across Feature Sets"
Rscript scripts/regression_result_analysis/visualization/model_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/model_wise_plots/voting_reg/r2.pdf --model voting --metric R2 --title "Voting Regression R2 Across Feature Sets"

# dataset wise analysis 
cd /Users/shreyasree/Documents/GitHub/KEAP1_ml

Rscript scripts/regression_result_analysis/visualization/dataset_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/dataset_wise_plots/descriptors_only/rmse.pdf --dataset descriptors_only.csv --metric RMSE --title "Descriptors Only RMSE Across Models"
Rscript scripts/regression_result_analysis/visualization/dataset_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/dataset_wise_plots/descriptors_only/mae.pdf --dataset descriptors_only.csv --metric MAE --title "Descriptors Only MAE Across Models"
Rscript scripts/regression_result_analysis/visualization/dataset_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/dataset_wise_plots/descriptors_only/mape.pdf --dataset descriptors_only.csv --metric MAPE --title "Descriptors Only MAPE Across Models"
Rscript scripts/regression_result_analysis/visualization/dataset_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/dataset_wise_plots/descriptors_only/r2.pdf --dataset descriptors_only.csv --metric R2 --title "Descriptors Only R2 Across Models"

Rscript scripts/regression_result_analysis/visualization/dataset_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/dataset_wise_plots/ratios_only/rmse.pdf --dataset ratios_only.csv --metric RMSE --title "Ratios Only RMSE Across Models"
Rscript scripts/regression_result_analysis/visualization/dataset_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/dataset_wise_plots/ratios_only/mae.pdf --dataset ratios_only.csv --metric MAE --title "Ratios Only MAE Across Models"
Rscript scripts/regression_result_analysis/visualization/dataset_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/dataset_wise_plots/ratios_only/mape.pdf --dataset ratios_only.csv --metric MAPE --title "Ratios Only MAPE Across Models"
Rscript scripts/regression_result_analysis/visualization/dataset_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/dataset_wise_plots/ratios_only/r2.pdf --dataset ratios_only.csv --metric R2 --title "Ratios Only R2 Across Models"

Rscript scripts/regression_result_analysis/visualization/dataset_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/dataset_wise_plots/transformations_only/rmse.pdf --dataset transformations_only.csv --metric RMSE --title "Transformations Only RMSE Across Models"
Rscript scripts/regression_result_analysis/visualization/dataset_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/dataset_wise_plots/transformations_only/mae.pdf --dataset transformations_only.csv --metric MAE --title "Transformations Only MAE Across Models"
Rscript scripts/regression_result_analysis/visualization/dataset_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/dataset_wise_plots/transformations_only/mape.pdf --dataset transformations_only.csv --metric MAPE --title "Transformations Only MAPE Across Models"
Rscript scripts/regression_result_analysis/visualization/dataset_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/dataset_wise_plots/transformations_only/r2.pdf --dataset transformations_only.csv --metric R2 --title "Transformations Only R2 Across Models"

Rscript scripts/regression_result_analysis/visualization/dataset_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/dataset_wise_plots/interactions_only/rmse.pdf --dataset interactions_only.csv --metric RMSE --title "Interactions Only RMSE Across Models"
Rscript scripts/regression_result_analysis/visualization/dataset_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/dataset_wise_plots/interactions_only/mae.pdf --dataset interactions_only.csv --metric MAE --title "Interactions Only MAE Across Models"
Rscript scripts/regression_result_analysis/visualization/dataset_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/dataset_wise_plots/interactions_only/mape.pdf --dataset interactions_only.csv --metric MAPE --title "Interactions Only MAPE Across Models"
Rscript scripts/regression_result_analysis/visualization/dataset_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/dataset_wise_plots/interactions_only/r2.pdf --dataset interactions_only.csv --metric R2 --title "Interactions Only R2 Across Models"

Rscript scripts/regression_result_analysis/visualization/dataset_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/dataset_wise_plots/raw_descs_and_ratios/rmse.pdf --dataset raw_descs_and_ratios.csv --metric RMSE --title "Raw Descriptors and Ratios RMSE Across Models"
Rscript scripts/regression_result_analysis/visualization/dataset_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/dataset_wise_plots/raw_descs_and_ratios/mae.pdf --dataset raw_descs_and_ratios.csv --metric MAE --title "Raw Descriptors and Ratios MAE Across Models"
Rscript scripts/regression_result_analysis/visualization/dataset_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/dataset_wise_plots/raw_descs_and_ratios/mape.pdf --dataset raw_descs_and_ratios.csv --metric MAPE --title "Raw Descriptors and Ratios MAPE Across Models"
Rscript scripts/regression_result_analysis/visualization/dataset_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/dataset_wise_plots/raw_descs_and_ratios/r2.pdf --dataset raw_descs_and_ratios.csv --metric R2 --title "Raw Descriptors and Ratios R2 Across Models"

Rscript scripts/regression_result_analysis/visualization/dataset_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/dataset_wise_plots/raw_descs_and_interactions/rmse.pdf --dataset raw_descs_and_interactions.csv --metric RMSE --title "Raw Descriptors and Interactions RMSE Across Models"
Rscript scripts/regression_result_analysis/visualization/dataset_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/dataset_wise_plots/raw_descs_and_interactions/mae.pdf --dataset raw_descs_and_interactions.csv --metric MAE --title "Raw Descriptors and Interactions MAE Across Models"
Rscript scripts/regression_result_analysis/visualization/dataset_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/dataset_wise_plots/raw_descs_and_interactions/mape.pdf --dataset raw_descs_and_interactions.csv --metric MAPE --title "Raw Descriptors and Interactions MAPE Across Models"
Rscript scripts/regression_result_analysis/visualization/dataset_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/dataset_wise_plots/raw_descs_and_interactions/r2.pdf --dataset raw_descs_and_interactions.csv --metric R2 --title "Raw Descriptors and Interactions R2 Across Models"

Rscript scripts/regression_result_analysis/visualization/dataset_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/dataset_wise_plots/raw_descs_and_transforms/rmse.pdf --dataset raw_descs_and_transforms.csv --metric RMSE --title "Raw Descriptors and Transformations RMSE Across Models"
Rscript scripts/regression_result_analysis/visualization/dataset_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/dataset_wise_plots/raw_descs_and_transforms/mae.pdf --dataset raw_descs_and_transforms.csv --metric MAE --title "Raw Descriptors and Transformations MAE Across Models"
Rscript scripts/regression_result_analysis/visualization/dataset_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/dataset_wise_plots/raw_descs_and_transforms/mape.pdf --dataset raw_descs_and_transforms.csv --metric MAPE --title "Raw Descriptors and Transformations MAPE Across Models"
Rscript scripts/regression_result_analysis/visualization/dataset_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/dataset_wise_plots/raw_descs_and_transforms/r2.pdf --dataset raw_descs_and_transforms.csv --metric R2 --title "Raw Descriptors and Transformations R2 Across Models"

Rscript scripts/regression_result_analysis/visualization/dataset_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/dataset_wise_plots/transforms_and_interactions/rmse.pdf --dataset transforms_and_interactions.csv --metric RMSE --title "Interactions and Transformations RMSE Across Models"
Rscript scripts/regression_result_analysis/visualization/dataset_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/dataset_wise_plots/transforms_and_interactions/mae.pdf --dataset transforms_and_interactions.csv --metric MAE --title "Interactions and Transformations MAE Across Models"
Rscript scripts/regression_result_analysis/visualization/dataset_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/dataset_wise_plots/transforms_and_interactions/mape.pdf --dataset transforms_and_interactions.csv --metric MAPE --title "Interactions and Transformations MAPE Across Models"
Rscript scripts/regression_result_analysis/visualization/dataset_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/dataset_wise_plots/transforms_and_interactions/r2.pdf --dataset transforms_and_interactions.csv --metric R2 --title "Interactions and Transformations R2 Across Models"

Rscript scripts/regression_result_analysis/visualization/dataset_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/dataset_wise_plots/transforms_and_ratios/rmse.pdf --dataset transforms_and_ratios.csv --metric RMSE --title "Ratios and Transformations RMSE Across Models"
Rscript scripts/regression_result_analysis/visualization/dataset_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/dataset_wise_plots/transforms_and_ratios/mae.pdf --dataset transforms_and_ratios.csv --metric MAE --title "Ratios and Transformations MAE Across Models"
Rscript scripts/regression_result_analysis/visualization/dataset_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/dataset_wise_plots/transforms_and_ratios/mape.pdf --dataset transforms_and_ratios.csv --metric MAPE --title "Ratios and Transformations MAPE Across Models"
Rscript scripts/regression_result_analysis/visualization/dataset_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/dataset_wise_plots/transforms_and_ratios/r2.pdf --dataset transforms_and_ratios.csv --metric R2 --title "Ratios and Transformations R2 Across Models"

Rscript scripts/regression_result_analysis/visualization/dataset_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/dataset_wise_plots/interactions_and_ratios/rmse.pdf --dataset interactions_and_ratios.csv --metric RMSE --title "Ratios and Interactions RMSE Across Models"
Rscript scripts/regression_result_analysis/visualization/dataset_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/dataset_wise_plots/interactions_and_ratios/mae.pdf --dataset interactions_and_ratios.csv --metric MAE --title "Ratios and Interactions MAE Across Models"
Rscript scripts/regression_result_analysis/visualization/dataset_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/dataset_wise_plots/interactions_and_ratios/mape.pdf --dataset interactions_and_ratios.csv --metric MAPE --title "Ratios and Interactions MAPE Across Models"
Rscript scripts/regression_result_analysis/visualization/dataset_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/dataset_wise_plots/interactions_and_ratios/r2.pdf --dataset interactions_and_ratios.csv --metric R2 --title "Ratios and Interactions R2 Across Models"

Rscript scripts/regression_result_analysis/visualization/dataset_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/dataset_wise_plots/all_4_combined/rmse.pdf --dataset all_4_combined.csv --metric RMSE --title "Combined RMSE Across Models"
Rscript scripts/regression_result_analysis/visualization/dataset_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/dataset_wise_plots/all_4_combined/mae.pdf --dataset all_4_combined.csv --metric MAE --title "Combined MAE Across Models"
Rscript scripts/regression_result_analysis/visualization/dataset_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/dataset_wise_plots/all_4_combined/mape.pdf --dataset all_4_combined.csv --metric MAPE --title "Combined MAPE Across Models"
Rscript scripts/regression_result_analysis/visualization/dataset_wise_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/dataset_wise_plots/all_4_combined/r2.pdf --dataset all_4_combined.csv --metric R2 --title "Combined R2 Across Models"

# complete model analysis 
cd /Users/shreyasree/Documents/GitHub/KEAP1_ml
Rscript scripts/regression_result_analysis/visualization/complete_regression_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/complete_regression_heatmaps/mae_complete.pdf --metric MAE_mean --title "Mean MAE across Models and Feature Sets"
Rscript scripts/regression_result_analysis/visualization/complete_regression_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/complete_regression_heatmaps/rmse_complete.pdf --metric RMSE_mean --title "Mean RMSE across Models and Feature Sets"
Rscript scripts/regression_result_analysis/visualization/complete_regression_analysis.R --input data/regression_result_analysis/combined_results.csv --output plots/complete_regression_heatmaps/r2_complete.pdf --metric R2_mean --title "Mean R2 across Models and Feature Sets"

# nemenyi p-value heatmap
Rscript scripts/regression_result_analysis/visualization/nemenyi_p_value_heatmap.R --input data/stat_tests/nemenyi_pvalues.csv --output plots/statistical_tests/p_value_heatmap.pdf

# average rank comparison
Rscript scripts/regression_result_analysis/visualization/average_rank_comparison.R --input data/stat_tests/average_ranks.csv --output plots/statistical_tests/average_rank_comparison.pdf

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# STEP 5: SHAP AND RETRAINING MODELS FOR INTERPRETABILITY ON BEST DATASET 

cd /Users/shreyasree/Documents/GitHub/KEAP1_ml/data
mkdir SHAP_data 

cd /Users/shreyasree/Documents/GitHub/KEAP1_ml
python3 scripts/re_train_SHAP/adaboost_regressor.py --input data/final_features_data/ratios_only.csv --output data/SHAP_data/adaboost_regressor 
python3 scripts/re_train_SHAP/catboost_regressor.py --input data/final_features_data/ratios_only.csv --output data/SHAP_data/catboost_regressor  
python3 scripts/re_train_SHAP/decision_tree_regressor.py --input data/final_features_data/ratios_only.csv --output data/SHAP_data/decision_tree_regressor  
python3 scripts/re_train_SHAP/elastic_net_regression.py --input data/final_features_data/ratios_only.csv --output data/SHAP_data/elastic_net_regressor
python3 scripts/re_train_SHAP/gradient_boosting.py  --input data/final_features_data/ratios_only.csv --output data/SHAP_data/gradient_boosting_regression
python3 scripts/re_train_SHAP/knn_regressor.py --input data/final_features_data/ratios_only.csv --output data/SHAP_data/knn_regression
python3 scripts/re_train_SHAP/lasso_regressor.py --input data/final_features_data/ratios_only.csv --output data/SHAP_data/lasso_regression
python3 scripts/re_train_SHAP/lightgbm_regressor.py --input data/final_features_data/ratios_only.csv --output data/SHAP_data/lightgbm_regression
python3 scripts/re_train_SHAP/lin_reg.py --input data/final_features_data/ratios_only.csv --output data/SHAP_data/linear_regression
python3 scripts/re_train_SHAP/polynomial_reg.py --input data/final_features_data/ratios_only.csv --output data/SHAP_data/polynomial_regression
python3 scripts/re_train_SHAP/random_forest_regressor.py --input data/final_features_data/ratios_only.csv --output data/SHAP_data/random_forest_regression
python3 scripts/re_train_SHAP/ridge_regressor.py --input data/final_features_data/ratios_only.csv --output data/SHAP_data/ridge_regression
python3 scripts/re_train_SHAP/svm_regressor.py --input data/final_features_data/ratios_only.csv --output data/SHAP_data/svr_regression
python3 scripts/re_train_SHAP/xgboost_regressor.py --input data/final_features_data/ratios_only.csv --output data/SHAP_data/xgboost_regression

# shap data visualization - beeswarm plots 
cd /Users/shreyasree/Documents/GitHub/KEAP1_ml
Rscript scripts/re_train_SHAP/visualization/beeswarm_plot.R --input data/SHAP_data/adaboost_regressor/shap_plot_data.csv --output plots/SHAP_plots/adaboost_regressor/beeswarm.pdf --title "SHAP Importance: AdaBoost Regression"
Rscript scripts/re_train_SHAP/visualization/beeswarm_plot.R --input data/SHAP_data/catboost_regressor/shap_plot_data.csv --output plots/SHAP_plots/catboost_regressor/beeswarm.pdf --title "SHAP Importance: CatBoost Regression"
Rscript scripts/re_train_SHAP/visualization/beeswarm_plot.R --input data/SHAP_data/decision_tree_regressor/shap_plot_data.csv --output plots/SHAP_plots/decision_tree_regressor/beeswarm.pdf --title "SHAP Importance: Decision Tree Regression"
Rscript scripts/re_train_SHAP/visualization/beeswarm_plot.R --input data/SHAP_data/elastic_net_regressor/shap_plot_data.csv --output plots/SHAP_plots/elastic_net_regressor/beeswarm.pdf --title "SHAP Importance: Elastic Net Regression"
Rscript scripts/re_train_SHAP/visualization/beeswarm_plot.R --input data/SHAP_data/gradient_boosting_regression/shap_plot_data.csv --output plots/SHAP_plots/gradient_boosting_regression/beeswarm.pdf --title "SHAP Importance: Gradient Boosting Regression"
Rscript scripts/re_train_SHAP/visualization/beeswarm_plot.R --input data/SHAP_data/knn_regression/shap_plot_data.csv --output plots/SHAP_plots/knn_regression/beeswarm.pdf --title "SHAP Importance: KNN Regression"
Rscript scripts/re_train_SHAP/visualization/beeswarm_plot.R --input data/SHAP_data/lasso_regression/shap_plot_data.csv --output plots/SHAP_plots/lasso_regression/beeswarm.pdf --title "SHAP Importance: LASSO Regression"
Rscript scripts/re_train_SHAP/visualization/beeswarm_plot.R --input data/SHAP_data/lightgbm_regression/shap_plot_data.csv --output plots/SHAP_plots/lightgbm_regression/beeswarm.pdf --title "SHAP Importance: LightGBM Regression"
Rscript scripts/re_train_SHAP/visualization/beeswarm_plot.R --input data/SHAP_data/linear_regression/shap_plot_data.csv --output plots/SHAP_plots/linear_regression/beeswarm.pdf --title "SHAP Importance: Linear Regression"
Rscript scripts/re_train_SHAP/visualization/beeswarm_plot.R --input data/SHAP_data/random_forest_regression/shap_plot_data.csv --output plots/SHAP_plots/random_forest_regression/beeswarm.pdf --title "SHAP Importance: Random Forest Regression"
Rscript scripts/re_train_SHAP/visualization/beeswarm_plot.R --input data/SHAP_data/polynomial_regression/shap_plot_data.csv --output plots/SHAP_plots/polynomial_regression/beeswarm.pdf --title "SHAP Importance: Polynomial Regression"
Rscript scripts/re_train_SHAP/visualization/beeswarm_plot.R --input data/SHAP_data/ridge_regression/shap_plot_data.csv --output plots/SHAP_plots/ridge_regression/beeswarm.pdf --title "SHAP Importance: Ridge Regression"
Rscript scripts/re_train_SHAP/visualization/beeswarm_plot.R --input data/SHAP_data/svr_regression/shap_plot_data.csv --output plots/SHAP_plots/svr_regression/beeswarm.pdf --title "SHAP Importance: SVR Regression"
Rscript scripts/re_train_SHAP/visualization/beeswarm_plot.R --input data/SHAP_data/xgboost_regression/shap_plot_data.csv --output plots/SHAP_plots/xgboost_regression/beeswarm.pdf --title "SHAP Importance: XGBoost Regression"

# shap data visualization - feature importance bar plot 
Rscript scripts/re_train_SHAP/visualization/feature_importance_bar_plot.R --input data/SHAP_data/adaboost_regressor/shap_importance.csv --output plots/SHAP_plots/adaboost_regressor/top20_feat_imp.pdf --title "SHAP Importance: AdaBoost Regression"
Rscript scripts/re_train_SHAP/visualization/feature_importance_bar_plot.R --input data/SHAP_data/catboost_regressor/shap_importance.csv --output plots/SHAP_plots/catboost_regressor/top20_feat_imp.pdf --title "SHAP Top 20 Features: CatBoost Regression"
Rscript scripts/re_train_SHAP/visualization/feature_importance_bar_plot.R --input data/SHAP_data/decision_tree_regressor/shap_importance.csv --output plots/SHAP_plots/decision_tree_regressor/top20_feat_imp.pdf --title "SHAP Top 20 Features: Decision Tree Regression"
Rscript scripts/re_train_SHAP/visualization/feature_importance_bar_plot.R --input data/SHAP_data/elastic_net_regressor/shap_importance.csv --output plots/SHAP_plots/elastic_net_regressor/top20_feat_imp.pdf --title "SHAP Top 20 Features: Elastic Net Regression"
Rscript scripts/re_train_SHAP/visualization/feature_importance_bar_plot.R --input data/SHAP_data/gradient_boosting_regression/shap_importance.csv --output plots/SHAP_plots/gradient_boosting_regression/top20_feat_imp.pdf --title "SHAP Top 20 Features: Gradient Boosting Regression"
Rscript scripts/re_train_SHAP/visualization/feature_importance_bar_plot.R --input data/SHAP_data/knn_regression/shap_importance.csv --output plots/SHAP_plots/knn_regression/top20_feat_imp.pdf --title "SHAP Top 20 Features: KNN Regression"
Rscript scripts/re_train_SHAP/visualization/feature_importance_bar_plot.R --input data/SHAP_data/lasso_regression/shap_importance.csv --output plots/SHAP_plots/lasso_regression/top20_feat_imp.pdf --title "SHAP Top 20 Features: LASSO Regression"
Rscript scripts/re_train_SHAP/visualization/feature_importance_bar_plot.R --input data/SHAP_data/lightgbm_regression/shap_importance.csv --output plots/SHAP_plots/lightgbm_regression/top20_feat_imp.pdf --title "SHAP Top 20 Features: LightGBM Regression"
Rscript scripts/re_train_SHAP/visualization/feature_importance_bar_plot.R --input data/SHAP_data/linear_regression/shap_importance.csv --output plots/SHAP_plots/linear_regression/top20_feat_imp.pdf --title "SHAP Top 20 Features: Linear Regression"
Rscript scripts/re_train_SHAP/visualization/feature_importance_bar_plot.R --input data/SHAP_data/random_forest_regression/shap_importance.csv --output plots/SHAP_plots/random_forest_regression/top20_feat_imp.pdf --title "SHAP Top 20 Features: Random Forest Regression"
Rscript scripts/re_train_SHAP/visualization/feature_importance_bar_plot.R --input data/SHAP_data/polynomial_regression/shap_importance.csv --output plots/SHAP_plots/polynomial_regression/top20_feat_imp.pdf --title "SHAP Top 20 Features: Polynomial Regression"
Rscript scripts/re_train_SHAP/visualization/feature_importance_bar_plot.R --input data/SHAP_data/ridge_regression/shap_importance.csv --output plots/SHAP_plots/ridge_regression/top20_feat_imp.pdf --title "SHAP Top 20 Features: Ridge Regression"
Rscript scripts/re_train_SHAP/visualization/feature_importance_bar_plot.R --input data/SHAP_data/svr_regression/shap_importance.csv --output plots/SHAP_plots/svr_regression/top20_feat_imp.pdf --title "SHAP Top 20 Features: SVR Regression"
Rscript scripts/re_train_SHAP/visualization/feature_importance_bar_plot.R --input data/SHAP_data/xgboost_regression/shap_importance.csv --output plots/SHAP_plots/xgboost_regression/top20_feat_imp.pdf --title "SHAP Top 20 Features: XGBoost Regression"

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## STEP 6: RE-TRAINING USING XGBOOST 

# pairwise descriptor ratios for retraining data generation
python3 scripts/retraining/data_cleaning.py --ratios_ip data/final_features_data/ratios_only.csv --docking_ip data/combined_scores/docking_score_data_no_outliers.csv --desc data/final_complete_descriptor_matrix.csv --output data/retraining_all_data.csv

# XGBoost retraining 
python3 scripts/retraining/xgboost_training.py --input data/final_features_data/ratios_only.csv --output data/final_xgboost

# Generating predictions using trained model 
python3 scripts/retraining/prediction.py --model data/final_xgboost/best_xgboost_model.pkl --data data/retraining_all_data.csv --output data/new_data_pred/predictions.csv

# extraction of top-scoring molecules from prediction results 
python3 scripts/retraining/prediction_filters.py --input data/new_data_pred/predictions.csv --output_dir data/new_data_pred/top_scorers --summary data/new_data_pred/top_scorers_summary.csv

# filter-docking results using predicted score cutoffs
python3 scripts/retraining/representative_filters.py scripts/retraining/representative_filters.py

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
