# ============================================================
# Horizontal Bar Plots: RMSE + Optimal k
# User-defined output paths
# ============================================================
library(ggplot2)
library(readr)
library(dplyr)

plot_feature_lines <- function(input_file,
                               rmse_plot_path,
                               k_plot_path) {
  
  df <- read_csv(input_file, show_col_types = FALSE)
  
  # ----------------------------------------------------------
  # Rename dataset labels
  # ----------------------------------------------------------
  
  df$dataset <- recode(
    df$dataset,
    "descriptors_only" = "Descriptors",
    "raw_descs_and_ratios" = "Raw + Ratios",
    "raw_descs_and_interactions" = "Raw + Interactions",
    "transforms_and_interactions" = "Transforms + Interactions",
    "transforms_and_ratios" = "Transforms + Ratios",
    "raw_descs_and_transforms" = "Raw + Transforms",
    "ratios_only" = "Ratios",
    "interactions_only" = "Interactions",
    "transformations_only" = "Transforms",
    "all_4_combined" = "All Features",
    "interactions_and_ratios" = "Interactions + Ratios"
  )
  
  # ==========================================================
  # RMSE LINE PLOT
  # ==========================================================
  
  df_rmse <- df %>%
    arrange(rmse) %>%
    mutate(dataset = factor(dataset, levels = dataset))
  
  p1 <- ggplot(df_rmse, aes(x = dataset, y = rmse, group = 1)) +
    geom_line(color = "pink", linewidth = 1) +
    geom_point(color = "pink", size = 3) +
    
    geom_text(
      aes(label = round(rmse, 3)),
      vjust = -0.7,
      size = 3.5
    ) +
    
    labs(
      title = "RMSE across Feature Sets",
      x = "Feature Set",
      y = "RMSE"
    ) +
    
    theme_bw(base_size = 12) +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold"),
      axis.text.x = element_text(angle = 30, hjust = 1)
    )
  
  ggsave(rmse_plot_path, p1, width = 10, height = 6, dpi = 300)
  
  # ==========================================================
  # k LINE PLOT
  # ==========================================================
  
  df_k <- df %>%
    arrange(k) %>%
    mutate(dataset = factor(dataset, levels = dataset))
  
  p2 <- ggplot(df_k, aes(x = dataset, y = k, group = 1)) +
    geom_line(color = "#FF46A2", linewidth = 1) +
    geom_point(color = "#FF46A2", size = 3) +
    
    geom_text(
      aes(label = k),
      vjust = -0.7,
      size = 3.5
    ) +
    
    labs(
      title = "Optimal Feature Count across Feature Sets",
      x = "Feature Set",
      y = "k"
    ) +
    
    theme_bw(base_size = 12) +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold"),
      axis.text.x = element_text(angle = 30, hjust = 1)
    )
  
  ggsave(k_plot_path, p2, width = 10, height = 6, dpi = 300)
  
  return(list(rmse_plot = p1, k_plot = p2))
}

plot_feature_lines(
  input_file = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/data/knee_detection_results.csv",
  rmse_plot_path = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/rmse_analysis/best_rmse.pdf",
  k_plot_path = "/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/rmse_analysis/best_k.pdf"
)