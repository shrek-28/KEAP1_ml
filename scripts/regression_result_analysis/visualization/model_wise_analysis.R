library(ggplot2)

plot_metric_with_ci <- function(
    df,
    model_name,
    metric,
    plot_title,
    output_file,
    line_color = "#FF46A2",
    width = 10,
    height = 6
) {
  
  mean_col <- paste0(metric, "_mean")
  ci_col   <- paste0(metric, "_ci")
  
  model_df <- subset(df, model == model_name)
  
  if (nrow(model_df) == 0) {
    stop("Model not found in dataset")
  }
  
  # Clean dataset labels
  model_df$dataset <- sub("\\.csv$", "", model_df$dataset)

  fixed_order <- c(
    "descriptors_only",
    "ratios_only",
    "transformations_only",
    "interactions_only",
    "raw_descs_and_ratios",
    "raw_descs_and_transforms",
    "transforms_and_ratios",
    "interactions_and_ratios",
    "raw_descs_and_interactions",
    "transforms_and_interactions",
    "all_4_combined"
  )
  
  model_df$dataset <- factor(model_df$dataset, levels = fixed_order)
  
  # Ensure numeric safety
  model_df[[mean_col]] <- as.numeric(model_df[[mean_col]])
  model_df[[ci_col]]   <- as.numeric(model_df[[ci_col]])
  
  # CI bounds based on your definition
  model_df$lower <- model_df[[mean_col]] - model_df[[ci_col]]
  model_df$upper <- model_df[[mean_col]] + model_df[[ci_col]]
  
  p <- ggplot(model_df, aes(x = dataset, y = .data[[mean_col]], group = 1)) +
    geom_line(color = line_color, linewidth = 1) +
    geom_point(color = line_color, size = 2) +
    geom_errorbar(
      aes(ymin = lower, ymax = upper),
      width = 0.2,
      linewidth = 0.8,
      color = line_color
    ) +
    labs(
      title = plot_title,
      x = "Feature Set",
      y = metric
    ) +
    theme_bw() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1)
    )
  
  y_min <- min(model_df$lower)
  y_max <- max(model_df$upper)
  pad <- 0.05 * (y_max + y_min)
  
  p <- p + coord_cartesian(ylim = c(y_min - pad, y_max + pad))
  
  ggsave(
    filename = output_file,
    plot = p,
    width = width,
    height = height
  )
  
  return(p)
}

df <- read.csv("/Users/shreyasree/Documents/GitHub/KEAP1_ml/data/regression_result_analysis/combined_results.csv")

## 1
plot_metric_with_ci(df = df,model_name = "linear_regression", metric = "RMSE", plot_title = "Linear Regression RMSE Across Feature Sets", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/model_wise_plots/linear_reg/rmse.pdf")
plot_metric_with_ci(df = df,model_name = "linear_regression", metric = "MAE", plot_title = "Linear Regression MAE Across Feature Sets", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/model_wise_plots/linear_reg/mae.pdf")
plot_metric_with_ci(df = df,model_name = "linear_regression", metric = "MAPE", plot_title = "Linear Regression MAPE Across Feature Sets", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/model_wise_plots/linear_reg/mape.pdf")
plot_metric_with_ci(df = df,model_name = "linear_regression", metric = "R2", plot_title = "Linear Regression R2 Across Feature Sets", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/model_wise_plots/linear_reg/r2.pdf")

plot_metric_with_ci(df = df,model_name = "ridge_regression", metric = "RMSE", plot_title = "Ridge Regression RMSE Across Feature Sets", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/model_wise_plots/ridge_reg/rmse.pdf")
plot_metric_with_ci(df = df,model_name = "ridge_regression", metric = "MAE", plot_title = "Ridge Regression MAE Across Feature Sets", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/model_wise_plots/ridge_reg/mae.pdf")
plot_metric_with_ci(df = df,model_name = "ridge_regression", metric = "MAPE", plot_title = "Ridge Regression MAPE Across Feature Sets", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/model_wise_plots/ridge_reg/mape.pdf")
plot_metric_with_ci(df = df,model_name = "ridge_regression", metric = "R2", plot_title = "Ridge Regression R2 Across Feature Sets", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/model_wise_plots/ridge_reg/r2.pdf")

plot_metric_with_ci(df = df,model_name = "lasso_regression", metric = "RMSE", plot_title = "Lasso Regression RMSE Across Feature Sets", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/model_wise_plots/lasso_reg/rmse.pdf")
plot_metric_with_ci(df = df,model_name = "lasso_regression", metric = "MAE", plot_title = "Lasso Regression MAE Across Feature Sets", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/model_wise_plots/lasso_reg/mae.pdf")
plot_metric_with_ci(df = df,model_name = "lasso_regression", metric = "MAPE", plot_title = "Lasso Regression MAPE Across Feature Sets", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/model_wise_plots/lasso_reg/mape.pdf")
plot_metric_with_ci(df = df,model_name = "lasso_regression", metric = "R2", plot_title = "Lasso Regression R2 Across Feature Sets", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/model_wise_plots/lasso_reg/r2.pdf")

plot_metric_with_ci(df = df,model_name = "elastic_net_regression", metric = "RMSE", plot_title = "Elastic Net Regression RMSE Across Feature Sets", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/model_wise_plots/elasticnet_reg/rmse.pdf")
plot_metric_with_ci(df = df,model_name = "elastic_net_regression", metric = "MAE", plot_title = "Elastic Net Regression MAE Across Feature Sets", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/model_wise_plots/elasticnet_reg/mae.pdf")
plot_metric_with_ci(df = df,model_name = "elastic_net_regression", metric = "MAPE", plot_title = "Elastic Net Regression MAPE Across Feature Sets", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/model_wise_plots/elasticnet_reg/mape.pdf")
plot_metric_with_ci(df = df,model_name = "elastic_net_regression", metric = "R2", plot_title = "Elastic Net Regression R2 Across Feature Sets", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/model_wise_plots/elasticnet_reg/r2.pdf")

## 5
plot_metric_with_ci(df = df,model_name = "poly", metric = "RMSE", plot_title = "Polynomial Regression RMSE Across Feature Sets", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/model_wise_plots/poly_reg/rmse.pdf")
plot_metric_with_ci(df = df,model_name = "poly", metric = "MAE", plot_title = "Polynomial Regression MAE Across Feature Sets", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/model_wise_plots/poly_reg/mae.pdf")
plot_metric_with_ci(df = df,model_name = "poly", metric = "MAPE", plot_title = "Polynomial Regression MAPE Across Feature Sets", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/model_wise_plots/poly_reg/mape.pdf")
plot_metric_with_ci(df = df,model_name = "poly", metric = "R2", plot_title = "Polynomial Regression R2 Across Feature Sets", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/model_wise_plots/poly_reg/r2.pdf")

plot_metric_with_ci(df = df,model_name = "knn_regressor", metric = "RMSE", plot_title = "K-Nearest Neighbours Regression RMSE Across Feature Sets", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/model_wise_plots/knn_reg/rmse.pdf")
plot_metric_with_ci(df = df,model_name = "knn_regressor", metric = "MAE", plot_title = "K-Nearest Neighbours Regression MAE Across Feature Sets", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/model_wise_plots/knn_reg/mae.pdf")
plot_metric_with_ci(df = df,model_name = "knn_regressor", metric = "MAPE", plot_title = "K-Nearest Neighbours Regression MAPE Across Feature Sets", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/model_wise_plots/knn_reg/mape.pdf")
plot_metric_with_ci(df = df,model_name = "knn_regressor", metric = "R2", plot_title = "K-Nearest Neighbours Regression R2 Across Feature Sets", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/model_wise_plots/knn_reg/r2.pdf")

plot_metric_with_ci(df = df,model_name = "svr", metric = "RMSE", plot_title = "Support Vector Regression RMSE Across Feature Sets", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/model_wise_plots/svm_reg/rmse.pdf")
plot_metric_with_ci(df = df,model_name = "svr", metric = "MAE", plot_title = "Support Vector Regression MAE Across Feature Sets", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/model_wise_plots/svm_reg/mae.pdf")
plot_metric_with_ci(df = df,model_name = "svr", metric = "MAPE", plot_title = "Support Vector Regression MAPE Across Feature Sets", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/model_wise_plots/svm_reg/mape.pdf")
plot_metric_with_ci(df = df,model_name = "svr", metric = "R2", plot_title = "Support Vector Regression R2 Across Feature Sets", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/model_wise_plots/svm_reg/r2.pdf")

plot_metric_with_ci(df = df,model_name = "decision_tree", metric = "RMSE", plot_title = "Decision Tree Regression RMSE Across Feature Sets", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/model_wise_plots/decision_tree_reg/rmse.pdf")
plot_metric_with_ci(df = df,model_name = "decision_tree", metric = "MAE", plot_title = "Decision Tree Regression MAE Across Feature Sets", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/model_wise_plots/decision_tree_reg/mae.pdf")
plot_metric_with_ci(df = df,model_name = "decision_tree", metric = "MAPE", plot_title = "Decision Tree Regression MAPE Across Feature Sets", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/model_wise_plots/decision_tree_reg/mape.pdf")
plot_metric_with_ci(df = df,model_name = "decision_tree", metric = "R2", plot_title = "Decision Tree Regression R2 Across Feature Sets", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/model_wise_plots/decision_tree_reg/r2.pdf")

## 9
plot_metric_with_ci(df = df,model_name = "random_forest", metric = "RMSE", plot_title = "Random Forest Regression RMSE Across Feature Sets", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/model_wise_plots/random_forest_reg/rmse.pdf")
plot_metric_with_ci(df = df,model_name = "random_forest", metric = "MAE", plot_title = "Random Forest Regression MAE Across Feature Sets", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/model_wise_plots/random_forest_reg/mae.pdf")
plot_metric_with_ci(df = df,model_name = "random_forest", metric = "MAPE", plot_title = "Random Forest Regression MAPE Across Feature Sets", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/model_wise_plots/random_forest_reg/mape.pdf")
plot_metric_with_ci(df = df,model_name = "random_forest", metric = "R2", plot_title = "Random Forest Regression R2 Across Feature Sets", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/model_wise_plots/random_forest_reg/r2.pdf")

plot_metric_with_ci(df = df,model_name = "xgboost", metric = "RMSE", plot_title = "XGBoost Regression RMSE Across Feature Sets", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/model_wise_plots/xgboost_reg/rmse.pdf")
plot_metric_with_ci(df = df,model_name = "xgboost", metric = "MAE", plot_title = "XGBoost Regression MAE Across Feature Sets", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/model_wise_plots/xgboost_reg/mae.pdf")
plot_metric_with_ci(df = df,model_name = "xgboost", metric = "MAPE", plot_title = "XGBoost Regression MAPE Across Feature Sets", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/model_wise_plots/xgboost_reg/mape.pdf")
plot_metric_with_ci(df = df,model_name = "xgboost", metric = "R2", plot_title = "XGBoost Regression R2 Across Feature Sets", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/model_wise_plots/xgboost_reg/r2.pdf")

plot_metric_with_ci(df = df,model_name = "catboost", metric = "RMSE", plot_title = "CatBoost Regression RMSE Across Feature Sets", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/model_wise_plots/catboost_reg/rmse.pdf")
plot_metric_with_ci(df = df,model_name = "catboost", metric = "MAE", plot_title = "CatBoost Regression MAE Across Feature Sets", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/model_wise_plots/catboost_reg/mae.pdf")
plot_metric_with_ci(df = df,model_name = "catboost", metric = "MAPE", plot_title = "CatBoost Regression MAPE Across Feature Sets", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/model_wise_plots/catboost_reg/mape.pdf")
plot_metric_with_ci(df = df,model_name = "catboost", metric = "R2", plot_title = "CatBoost Regression R2 Across Feature Sets", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/model_wise_plots/catboost_reg/r2.pdf")

plot_metric_with_ci(df = df,model_name = "adaboost", metric = "RMSE", plot_title = "AdaBoost Regression RMSE Across Feature Sets", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/model_wise_plots/adaboost_reg/rmse.pdf")
plot_metric_with_ci(df = df,model_name = "adaboost", metric = "MAE", plot_title = "AdaBoost Regression MAE Across Feature Sets", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/model_wise_plots/adaboost_reg/mae.pdf")
plot_metric_with_ci(df = df,model_name = "adaboost", metric = "MAPE", plot_title = "AdaBoost Regression MAPE Across Feature Sets", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/model_wise_plots/adaboost_reg/mape.pdf")
plot_metric_with_ci(df = df,model_name = "adaboost", metric = "R2", plot_title = "AdaBoost Regression R2 Across Feature Sets", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/model_wise_plots/adaboost_reg/r2.pdf")

## 13
plot_metric_with_ci(df = df,model_name = "lightgbm", metric = "RMSE", plot_title = "LightGBM Regression RMSE Across Feature Sets", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/model_wise_plots/lightgbm_reg/rmse.pdf")
plot_metric_with_ci(df = df,model_name = "lightgbm", metric = "MAE", plot_title = "LightGBM Regression MAE Across Feature Sets", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/model_wise_plots/lightgbm_reg/mae.pdf")
plot_metric_with_ci(df = df,model_name = "lightgbm", metric = "MAPE", plot_title = "LightGBM Regression MAPE Across Feature Sets", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/model_wise_plots/lightgbm_reg/mape.pdf")
plot_metric_with_ci(df = df,model_name = "lightgbm", metric = "R2", plot_title = "LightGBM Regression R2 Across Feature Sets", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/model_wise_plots/lightgbm_reg/r2.pdf")

plot_metric_with_ci(df = df,model_name = "gbr", metric = "RMSE", plot_title = "Gradient Boosting Regression RMSE Across Feature Sets", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/model_wise_plots/grad_boost_reg/rmse.pdf")
plot_metric_with_ci(df = df,model_name = "gbr", metric = "MAE", plot_title = "Gradient Boosting Regression MAE Across Feature Sets", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/model_wise_plots/grad_boost_reg/mae.pdf")
plot_metric_with_ci(df = df,model_name = "gbr", metric = "MAPE", plot_title = "Gradient Boosting Regression MAPE Across Feature Sets", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/model_wise_plots/grad_boost_reg/mape.pdf")
plot_metric_with_ci(df = df,model_name = "gbr", metric = "R2", plot_title = "Gradient Boosting Regression R2 Across Feature Sets", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/model_wise_plots/grad_boost_reg/r2.pdf")

plot_metric_with_ci(df = df,model_name = "stack", metric = "RMSE", plot_title = "Stacking Regression RMSE Across Feature Sets", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/model_wise_plots/stacking_reg/rmse.pdf")
plot_metric_with_ci(df = df,model_name = "stack", metric = "MAE", plot_title = "Stacking Regression MAE Across Feature Sets", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/model_wise_plots/stacking_reg/mae.pdf")
plot_metric_with_ci(df = df,model_name = "stack", metric = "MAPE", plot_title = "Stacking Regression MAPE Across Feature Sets", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/model_wise_plots/stacking_reg/mape.pdf")
plot_metric_with_ci(df = df,model_name = "stack", metric = "R2", plot_title = "Stacking Regression R2 Across Feature Sets", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/model_wise_plots/stacking_reg/r2.pdf")

plot_metric_with_ci(df = df,model_name = "voting", metric = "RMSE", plot_title = "Voting Regression RMSE Across Feature Sets", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/model_wise_plots/voting_reg/rmse.pdf")
plot_metric_with_ci(df = df,model_name = "voting", metric = "MAE", plot_title = "Voting Regression MAE Across Feature Sets", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/model_wise_plots/voting_reg/mae.pdf")
plot_metric_with_ci(df = df,model_name = "voting", metric = "MAPE", plot_title = "Voting Regression MAPE Across Feature Sets", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/model_wise_plots/voting_reg/mape.pdf")
plot_metric_with_ci(df = df,model_name = "voting", metric = "R2", plot_title = "Voting Regression R2 Across Feature Sets", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/model_wise_plots/voting_reg/r2.pdf")
