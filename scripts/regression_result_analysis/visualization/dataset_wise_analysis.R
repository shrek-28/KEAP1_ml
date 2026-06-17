library(ggplot2)

plot_metric_with_ci_flipped <- function(
    df,
    dataset_name,
    metric,
    plot_title,
    output_file,
    line_color = "#FF46A2",
    width = 10,
    height = 6
) {
  
  mean_col <- paste0(metric, "_mean")
  ci_col   <- paste0(metric, "_ci")
  
  # filter by dataset instead of model
  data_df <- subset(df, dataset == dataset_name)
  
  if (nrow(data_df) == 0) {
    stop("Dataset not found in dataframe")
  }
  
  # clean dataset (even though fixed, keeps consistency)
  data_df$dataset <- sub("\\.csv$", "", data_df$dataset)
  
  # ensure ordering on X-axis (models now)
  fixed_model_order <- c(
    "linear_regression",
    "ridge_regression",
    "lasso_regression",
    "elastic_net_regression",
    "polynomial_regression",
    "knn_regressor",
    "support_vector_regression",
    "decision_tree",
    "random_forest",
    "xgboost",
    "catboost",
    "adaboost",
    "gradient_boosting",
    "lightgbm",
    "stacked_model",
    "voting_regressor"
  )
  
  data_df$model <- factor(data_df$model, levels = fixed_model_order)
  
  # numeric safety
  data_df[[mean_col]] <- as.numeric(data_df[[mean_col]])
  data_df[[ci_col]]   <- as.numeric(data_df[[ci_col]])
  
  # CI bounds
  data_df$lower <- data_df[[mean_col]] - data_df[[ci_col]]
  data_df$upper <- data_df[[mean_col]] + data_df[[ci_col]]
  
  p <- ggplot(data_df, aes(x = model, y = .data[[mean_col]], group = 1)) +
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
      x = "Model",
      y = metric
    ) +
    theme_bw() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1)
    )
  
  y_min <- min(data_df$lower)
  y_max <- max(data_df$upper)
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

plot_metric_with_ci_flipped(df = df, dataset_name = "descriptors_only.csv", metric = "RMSE", plot_title = "Descriptors Only RMSE Across Models", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/dataset_wise_plots/descriptors_only/rmse.pdf")
plot_metric_with_ci_flipped(df = df, dataset_name = "descriptors_only.csv", metric = "MAE", plot_title = "Descriptors Only MAE Across Models", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/dataset_wise_plots/descriptors_only/mae.pdf")
plot_metric_with_ci_flipped(df = df, dataset_name = "descriptors_only.csv", metric = "MAPE", plot_title = "Descriptors Only MAPE Across Models", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/dataset_wise_plots/descriptors_only/mape.pdf")
plot_metric_with_ci_flipped(df = df, dataset_name = "descriptors_only.csv", metric = "R2", plot_title = "Descriptors Only R2 Across Models", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/dataset_wise_plots/descriptors_only/r2.pdf")

plot_metric_with_ci_flipped(df = df, dataset_name = "ratios_only.csv", metric = "RMSE", plot_title = "Ratios Only RMSE Across Models", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/dataset_wise_plots/ratios_only/rmse.pdf")
plot_metric_with_ci_flipped(df = df, dataset_name = "ratios_only.csv", metric = "MAE", plot_title = "Ratios Only MAE Across Models", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/dataset_wise_plots/ratios_only/mae.pdf")
plot_metric_with_ci_flipped(df = df, dataset_name = "ratios_only.csv", metric = "MAPE", plot_title = "Ratios Only MAPE Across Models", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/dataset_wise_plots/ratios_only/mape.pdf")
plot_metric_with_ci_flipped(df = df, dataset_name = "ratios_only.csv", metric = "R2", plot_title = "Ratios Only R2 Across Models", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/dataset_wise_plots/ratios_only/r2.pdf")

plot_metric_with_ci_flipped(df = df, dataset_name = "transformations_only.csv", metric = "RMSE", plot_title = "Transformations Only RMSE Across Models", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/dataset_wise_plots/transformations_only/rmse.pdf")
plot_metric_with_ci_flipped(df = df, dataset_name = "transformations_only.csv", metric = "MAE", plot_title = "Transformations Only MAE Across Models", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/dataset_wise_plots/transformations_only/mae.pdf")
plot_metric_with_ci_flipped(df = df, dataset_name = "transformations_only.csv", metric = "MAPE", plot_title = "Transformations Only MAPE Across Models", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/dataset_wise_plots/transformations_only/mape.pdf")
plot_metric_with_ci_flipped(df = df, dataset_name = "transformations_only.csv", metric = "R2", plot_title = "Transformations Only R2 Across Models", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/dataset_wise_plots/transformations_only/r2.pdf")

plot_metric_with_ci_flipped(df = df, dataset_name = "interactions_only.csv", metric = "RMSE", plot_title = "Interactions Only RMSE Across Models", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/dataset_wise_plots/interactions_only/rmse.pdf")
plot_metric_with_ci_flipped(df = df, dataset_name = "interactions_only.csv", metric = "MAE", plot_title = "Interactions Only MAE Across Models", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/dataset_wise_plots/interactions_only/mae.pdf")
plot_metric_with_ci_flipped(df = df, dataset_name = "interactions_only.csv", metric = "MAPE", plot_title = "Interactions Only MAPE Across Models", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/dataset_wise_plots/interactions_only/mape.pdf")
plot_metric_with_ci_flipped(df = df, dataset_name = "interactions_only.csv", metric = "R2", plot_title = "Interactions Only R2 Across Models", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/dataset_wise_plots/interactions_only/r2.pdf")

plot_metric_with_ci_flipped(df = df, dataset_name = "raw_descs_and_ratios.csv", metric = "RMSE", plot_title = "Raw Descriptors and Ratios RMSE Across Models", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/dataset_wise_plots/raw_descs_and_ratios/rmse.pdf")
plot_metric_with_ci_flipped(df = df, dataset_name = "raw_descs_and_ratios.csv", metric = "MAE", plot_title = "Raw Descriptors and Ratios MAE Across Models", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/dataset_wise_plots/raw_descs_and_ratios/mae.pdf")
plot_metric_with_ci_flipped(df = df, dataset_name = "raw_descs_and_ratios.csv", metric = "MAPE", plot_title = "Raw Descriptors and Ratios MAPE Across Models", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/dataset_wise_plots/raw_descs_and_ratios/mape.pdf")
plot_metric_with_ci_flipped(df = df, dataset_name = "raw_descs_and_ratios.csv", metric = "R2", plot_title = "Raw Descriptors and Ratios R2 Across Models", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/dataset_wise_plots/raw_descs_and_ratios/r2.pdf")

plot_metric_with_ci_flipped(df = df, dataset_name = "raw_descs_and_interactions.csv", metric = "RMSE", plot_title = "Raw Descriptors and Interactions RMSE Across Models", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/dataset_wise_plots/raw_descs_and_interactions/rmse.pdf")
plot_metric_with_ci_flipped(df = df, dataset_name = "raw_descs_and_interactions.csv", metric = "MAE", plot_title = "Raw Descriptors and Interactions MAE Across Models", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/dataset_wise_plots/raw_descs_and_interactions/mae.pdf")
plot_metric_with_ci_flipped(df = df, dataset_name = "raw_descs_and_interactions.csv", metric = "MAPE", plot_title = "Raw Descriptors and Interactions MAPE Across Models", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/dataset_wise_plots/raw_descs_and_interactions/mape.pdf")
plot_metric_with_ci_flipped(df = df, dataset_name = "raw_descs_and_interactions.csv", metric = "R2", plot_title = "Raw Descriptors and Interactions R2 Across Models", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/dataset_wise_plots/raw_descs_and_interactions/r2.pdf")

plot_metric_with_ci_flipped(df = df, dataset_name = "raw_descs_and_transforms.csv", metric = "RMSE", plot_title = "Raw Descriptors and Transformations RMSE Across Models", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/dataset_wise_plots/raw_descs_and_transforms/rmse.pdf")
plot_metric_with_ci_flipped(df = df, dataset_name = "raw_descs_and_transforms.csv", metric = "MAE", plot_title = "Raw Descriptors and Transformations MAE Across Models", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/dataset_wise_plots/raw_descs_and_transforms/mae.pdf")
plot_metric_with_ci_flipped(df = df, dataset_name = "raw_descs_and_transforms.csv", metric = "MAPE", plot_title = "Raw Descriptors and Transformations MAPE Across Models", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/dataset_wise_plots/raw_descs_and_transforms/mape.pdf")
plot_metric_with_ci_flipped(df = df, dataset_name = "raw_descs_and_transforms.csv", metric = "R2", plot_title = "Raw Descriptors and Transformations R2 Across Models", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/dataset_wise_plots/raw_descs_and_transforms/r2.pdf")

plot_metric_with_ci_flipped(df = df, dataset_name = "transforms_and_interactions.csv", metric = "RMSE", plot_title = "Interactions and Transformations RMSE Across Models", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/dataset_wise_plots/transforms_and_interactions/rmse.pdf")
plot_metric_with_ci_flipped(df = df, dataset_name = "transforms_and_interactions.csv", metric = "MAE", plot_title = "Interactions and Transformations MAE Across Models", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/dataset_wise_plots/transforms_and_interactions/mae.pdf")
plot_metric_with_ci_flipped(df = df, dataset_name = "transforms_and_interactions.csv", metric = "MAPE", plot_title = "Interactions and Transformations MAPE Across Models", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/dataset_wise_plots/transforms_and_interactions/mape.pdf")
plot_metric_with_ci_flipped(df = df, dataset_name = "transforms_and_interactions.csv", metric = "R2", plot_title = "Interactions and Transformations R2 Across Models", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/dataset_wise_plots/transforms_and_interactions/r2.pdf")

plot_metric_with_ci_flipped(df = df, dataset_name = "transforms_and_ratios.csv", metric = "RMSE", plot_title = "Ratios and Transformations RMSE Across Models", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/dataset_wise_plots/transforms_and_ratios/rmse.pdf")
plot_metric_with_ci_flipped(df = df, dataset_name = "transforms_and_ratios.csv", metric = "MAE", plot_title = "Ratios and Transformations MAE Across Models", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/dataset_wise_plots/transforms_and_ratios/mae.pdf")
plot_metric_with_ci_flipped(df = df, dataset_name = "transforms_and_ratios.csv", metric = "MAPE", plot_title = "Ratios and Transformations MAPE Across Models", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/dataset_wise_plots/transforms_and_ratios/mape.pdf")
plot_metric_with_ci_flipped(df = df, dataset_name = "transforms_and_ratios.csv", metric = "R2", plot_title = "Ratios and Transformations R2 Across Models", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/dataset_wise_plots/transforms_and_ratios/r2.pdf")

plot_metric_with_ci_flipped(df = df, dataset_name = "interactions_and_ratios.csv", metric = "RMSE", plot_title = "Ratios and Interactions RMSE Across Models", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/dataset_wise_plots/interactions_and_ratios/rmse.pdf")
plot_metric_with_ci_flipped(df = df, dataset_name = "interactions_and_ratios.csv", metric = "MAE", plot_title = "Ratios and Interactions MAE Across Models", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/dataset_wise_plots/interactions_and_ratios/mae.pdf")
plot_metric_with_ci_flipped(df = df, dataset_name = "interactions_and_ratios.csv", metric = "MAPE", plot_title = "Ratios and Interactions MAPE Across Models", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/dataset_wise_plots/interactions_and_ratios/mape.pdf")
plot_metric_with_ci_flipped(df = df, dataset_name = "interactions_and_ratios.csv", metric = "R2", plot_title = "Ratios and Interactions R2 Across Models", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/dataset_wise_plots/interactions_and_ratios/r2.pdf")

plot_metric_with_ci_flipped(df = df, dataset_name = "all_4_combined.csv", metric = "RMSE", plot_title = "Combined RMSE Across Models", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/dataset_wise_plots/all_4_combined/rmse.pdf")
plot_metric_with_ci_flipped(df = df, dataset_name = "all_4_combined.csv", metric = "MAE", plot_title = "Combined MAE Across Models", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/dataset_wise_plots/all_4_combined/mae.pdf")
plot_metric_with_ci_flipped(df = df, dataset_name = "all_4_combined.csv", metric = "MAPE", plot_title = "Combined MAPE Across Models", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/dataset_wise_plots/all_4_combined/mape.pdf")
plot_metric_with_ci_flipped(df = df, dataset_name = "all_4_combined.csv", metric = "R2", plot_title = "Combined R2 Across Models", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/dataset_wise_plots/all_4_combined/r2.pdf")

