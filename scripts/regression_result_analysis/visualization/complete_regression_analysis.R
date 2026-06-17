library(tidyverse)

fixed_feature_order <- c(
  "descriptors_only",
  "ratios_only",
  "transformations_only",
  "interactions_only",
  "raw_descs_and_ratios",
  "raw_descs_and_transforms",
  "raw_descs_and_interactions",
  "transforms_and_ratios",
  "interactions_and_ratios",
  "transforms_and_interactions",
  "all_4_combined"
)

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

library(tidyverse)

plot_model_heatmap <- function(data,
                               metric = "MAE_mean",
                               title = NULL,
                               x_label = "Feature Set",
                               y_label = "Model",
                               model_order = NULL,
                               feature_order = NULL,
                               reverse_fill = TRUE,
                               show_values = TRUE,
                               value_digits = 2,
                               output_path = NULL,
                               width = 7,
                               height = 10,
                               dpi = 300) {
  
  # -----------------------------
  # Validate metric
  # -----------------------------
  if (!metric %in% colnames(data)) {
    stop(paste("Metric not found:", metric))
  }
  
  # -----------------------------
  # Clean dataset names
  # -----------------------------
  data <- data %>%
    mutate(dataset = str_replace(dataset, "\\.csv$", ""))
  
  # -----------------------------
  # Build heatmap data
  # -----------------------------
  heatmap_df <- data %>%
    select(model, dataset, all_of(metric)) %>%
    group_by(model, dataset) %>%
    summarise(value = mean(.data[[metric]], na.rm = TRUE),
              .groups = "drop")
  
  # -----------------------------
  # Apply custom model order
  # -----------------------------
  if (!is.null(model_order)) {
    heatmap_df$model <- factor(heatmap_df$model, levels = model_order)
  }
  
  # -----------------------------
  # Apply custom feature order
  # -----------------------------
  if (!is.null(feature_order)) {
    feature_order <- str_replace(feature_order, "\\.csv$", "")
    heatmap_df$dataset <- factor(heatmap_df$dataset, levels = feature_order)
  }
  
  # -----------------------------
  # Title
  # -----------------------------
  if (is.null(title)) {
    title <- paste0(metric, " (Model vs Feature Set)")
  }
  
  # -----------------------------
  # Pink gradient
  # -----------------------------
  fill_scale <- if (reverse_fill) {
    scale_fill_gradient(low = "#c51b8a", high = "#fde0dd", name = metric)
  } else {
    scale_fill_gradient(low = "#fde0dd", high = "#c51b8a", name = metric)
  }
  
  # -----------------------------
  # Plot
  # -----------------------------
  p <- ggplot(heatmap_df, aes(x = dataset, y = model, fill = value)) +
    geom_tile(color = "white", linewidth = 0.3) +
    
    {if (show_values)
      geom_text(aes(label = round(value, value_digits)), size = 3)
    } +
    
    fill_scale +
    
    labs(
      title = title,
      x = x_label,
      y = y_label
    ) +
    
    theme_minimal() +
    theme(
      plot.title = element_text(face = "bold", hjust = 0.5),
      axis.text.x = element_text(angle = 45, hjust = 1),
      panel.grid = element_blank()
    )
  
  # -----------------------------
  # Save if path provided
  # -----------------------------
  if (!is.null(output_path)) {
    ggsave(
      filename = output_path,
      plot = p,
      width = width,
      height = height,
      dpi = dpi
    )
  }
  
  return(p)
}

df <- read.csv("/Users/shreyasree/Documents/GitHub/KEAP1_ml/data/regression_result_analysis/combined_results.csv")

plot_model_heatmap(df, metric = "MAE_mean",
                   title="Mean MAE across Models and Feature Sets",
                   model_order=fixed_model_order,
                   feature_order=fixed_feature_order,
                   output_path="/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/complete_regression_heatmaps/mae_complete.pdf"
                   )

plot_model_heatmap(df, metric = "RMSE_mean",
                   title="Mean RMSE across Models and Feature Sets",
                   model_order=fixed_model_order,
                   feature_order=fixed_feature_order,
                   output_path="/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/complete_regression_heatmaps/rmse_complete.pdf"
)

plot_model_heatmap(df, metric = "R2_mean",
                   title="Mean R2 across Models and Feature Sets",
                   model_order=fixed_model_order,
                   feature_order=fixed_feature_order,
                   output_path="/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/complete_regression_heatmaps/r2_complete.pdf"
)
