#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
input_file <- args[which(args == "--input") + 1]
output_file <- args[which(args == "--output") + 1]
plot_title <- args[which(args == "--title") + 1]

library(ggplot2)

df <- read.csv(input_file)
df$Feature <- gsub("_div_", "/", df$Feature, fixed = TRUE)

# Rank features by mean absolute SHAP value
feature_order <- aggregate(abs(SHAPValue) ~ Feature, data = df, FUN = mean)
feature_order <- feature_order[order(feature_order$`abs(SHAPValue)`, decreasing = TRUE), ]

# Select top 20 features
top_features <- head(feature_order$Feature, 20)
df <- df[df$Feature %in% top_features, ]

# Order features by importance
df$Feature <- factor(df$Feature, levels = rev(top_features))

# Scale feature values within each feature for meaningful color mapping
df$FeatureValueScaled <- ave(
  df$FeatureValue,
  df$Feature,
  FUN = function(x) rank(x, ties.method = "average") / sum(!is.na(x))
)

p <- ggplot(df, aes(x = SHAPValue, y = Feature, color = FeatureValueScaled)) +
  geom_jitter(height = 0.25, width = 0, alpha = 0.7, size = 1.2) +
  scale_color_gradient(
    low = "blue",
    high = "red",
    name = "Feature value"
  ) +
  labs(
    x = "SHAP value",
    y = "Feature",
    title = plot_title
  ) +
  theme_classic() +
  theme(
    axis.text.y = element_text(size = 8),
    legend.position = "top"
  )

# Create output directory if it does not exist
output_dir <- dirname(output_file)
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

ggsave(output_file, p, width = 10, height = 8)