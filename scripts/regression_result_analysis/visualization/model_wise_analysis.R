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

plot_metric_with_ci(df = df,model_name = "poly", metric = "RMSE", plot_title = "Polynomial Regression RMSE Across Feature Sets", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/model_wise_plots/rmse_poly.pdf")
plot_metric_with_ci(df = df,model_name = "poly", metric = "MAE", plot_title = "Polynomial Regression MAE Across Feature Sets", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/model_wise_plots/mae_poly.pdf")
plot_metric_with_ci(df = df,model_name = "poly", metric = "MAPE", plot_title = "Polynomial Regression MAPE Across Feature Sets", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/model_wise_plots/mape_poly.pdf")
plot_metric_with_ci(df = df,model_name = "poly", metric = "R2", plot_title = "Polynomial Regression R2 Across Feature Sets", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/model_wise_plots/r2_poly.pdf")
plot_metric_with_ci(df = df,model_name = "poly", metric = "MSE", plot_title = "Polynomial Regression MSE Across Feature Sets", output_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/model_wise_plots/mse_poly.pdf")
