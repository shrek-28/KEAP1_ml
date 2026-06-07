# ============================================================
# RMSE vs Number of Features Plot
# ============================================================


library(ggplot2)
library(ggrepel)

plot_rmse_vs_features <- function(input_file,
                                  output_file,
                                  plot_title = "RMSE vs Number of Features") {
  
  # ---------------- Read Data ----------------
  df <- read.csv(input_file, stringsAsFactors = FALSE)
  
  # Ensure proper ordering on x-axis
  df <- df[order(df$n_features), ]
  
  # RMSE labels rounded to 2 decimals
  df$rmse_label <- sprintf("%.2f", df$rmse)
  
  # ---------------- Plot ----------------
  p <- ggplot(df, aes(x = n_features, y = rmse)) +
    
    geom_line(linewidth = 0.8) +
    
    geom_point(size = 3) +
    
    geom_text_repel(
      aes(label = rmse_label),
      size = 3.5,
      max.overlaps = Inf,
      box.padding = 0.3,
      point.padding = 0.2,
      segment.alpha = 0.5
    ) +
    
    scale_x_continuous(
      breaks = df$n_features
    ) +
    
    labs(
      title = plot_title,
      x = "Number of Features",
      y = "RMSE"
    ) +
    
    theme_bw(base_size = 12) +
    
    theme(
      plot.title = element_text(
        hjust = 0.5,
        face = "bold"
      ),
      axis.title = element_text(face = "bold"),
      panel.grid.minor = element_blank()
    )
  
  # ---------------- Save ----------------
  ggsave(
    filename = output_file,
    plot = p,
    width = 25,
    height = 8,
    dpi = 300
  )
  
  return(p)
}


plot_rmse_vs_features(
  input_file = "/Users/shreyasree/Documents/GitHub/KEAP1drugdiscovery/data/rmse_split_results/all_4_combined.csv",
  output_file = "/Users/shreyasree/Documents/GitHub/KEAP1drugdiscovery/plots/rmse_n_features/all_4_combined.pdf",
  plot_title = "All 4 Feature Sets"
)
plot_rmse_vs_features(
  input_file = "/Users/shreyasree/Documents/GitHub/KEAP1drugdiscovery/data/rmse_split_results/descriptors_only.csv",
  output_file = "/Users/shreyasree/Documents/GitHub/KEAP1drugdiscovery/plots/rmse_n_features/descriptors_only.pdf",
  plot_title = "Descriptors Only"
)
plot_rmse_vs_features(
  input_file = "/Users/shreyasree/Documents/GitHub/KEAP1drugdiscovery/data/rmse_split_results/interactions_and_ratios.csv",
  output_file = "/Users/shreyasree/Documents/GitHub/KEAP1drugdiscovery/plots/rmse_n_features/interactions_and_ratios.pdf",
  plot_title = "Interactions and Ratios"
)
plot_rmse_vs_features(
  input_file = "/Users/shreyasree/Documents/GitHub/KEAP1drugdiscovery/data/rmse_split_results/interactions_only.csv",
  output_file = "/Users/shreyasree/Documents/GitHub/KEAP1drugdiscovery/plots/rmse_n_features/interactions_only.pdf",
  plot_title = "Interactions Only"
)
plot_rmse_vs_features(
  input_file = "/Users/shreyasree/Documents/GitHub/KEAP1drugdiscovery/data/rmse_split_results/ratios_only.csv",
  output_file = "/Users/shreyasree/Documents/GitHub/KEAP1drugdiscovery/plots/rmse_n_features/ratios_only.pdf",
  plot_title = "Ratios Only"
)
plot_rmse_vs_features(
  input_file = "/Users/shreyasree/Documents/GitHub/KEAP1drugdiscovery/data/rmse_split_results/transformations_only.csv",
  output_file = "/Users/shreyasree/Documents/GitHub/KEAP1drugdiscovery/plots/rmse_n_features/transforms_only.pdf",
  plot_title = "Transformations Only"
)
plot_rmse_vs_features(
  input_file = "/Users/shreyasree/Documents/GitHub/KEAP1drugdiscovery/data/rmse_split_results/raw_descs_and_interactions.csv",
  output_file = "/Users/shreyasree/Documents/GitHub/KEAP1drugdiscovery/plots/rmse_n_features/raw_descs_and_interactions.pdf",
  plot_title = "Descriptors and Interactions"
)
plot_rmse_vs_features(
  input_file = "/Users/shreyasree/Documents/GitHub/KEAP1drugdiscovery/data/rmse_split_results/raw_descs_and_ratios.csv",
  output_file = "/Users/shreyasree/Documents/GitHub/KEAP1drugdiscovery/plots/rmse_n_features/raw_descs_and_ratios.pdf",
  plot_title = "Descriptors and Ratios"
)
plot_rmse_vs_features(
  input_file = "/Users/shreyasree/Documents/GitHub/KEAP1drugdiscovery/data/rmse_split_results/raw_descs_and_transforms.csv",
  output_file = "/Users/shreyasree/Documents/GitHub/KEAP1drugdiscovery/plots/rmse_n_features/raw_descs_and_transforms.pdf",
  plot_title = "Descriptors and Transformations"
)
plot_rmse_vs_features(
  input_file="/Users/shreyasree/Documents/GitHub/KEAP1drugdiscovery/data/rmse_split_results/transforms_and_interactions.csv",
  output_file = "/Users/shreyasree/Documents/GitHub/KEAP1drugdiscovery/plots/rmse_n_features/transforms_and_interactions.pdf",
  plot_title = "Interactions and Transformations"
)
plot_rmse_vs_features(
  input_file="/Users/shreyasree/Documents/GitHub/KEAP1drugdiscovery/data/rmse_split_results/transforms_and_ratios.csv",
  output_file = "/Users/shreyasree/Documents/GitHub/KEAP1drugdiscovery/plots/rmse_n_features/transforms_and_ratios.pdf",
  plot_title = "Ratios and Transformations"
)
