#!/usr/bin/env Rscript

library(tidyverse)

args <- commandArgs(trailingOnly = TRUE)

input_file <- args[which(args == "--input") + 1]
output_file <- args[which(args == "--output") + 1]
metric <- args[which(args == "--metric") + 1]
plot_title <- args[which(args == "--title") + 1]

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

df <- read.csv(input_file)

df <- df %>%
  mutate(dataset = str_replace(dataset, "\\.csv$", ""))

heatmap_df <- df %>%
  select(model, dataset, all_of(metric)) %>%
  group_by(model, dataset) %>%
  summarise(value = mean(.data[[metric]], na.rm = TRUE), .groups = "drop")

heatmap_df$model <- factor(heatmap_df$model, levels = fixed_model_order)
heatmap_df$dataset <- factor(heatmap_df$dataset, levels = fixed_feature_order)

p <- ggplot(heatmap_df, aes(x = dataset, y = model, fill = value)) +
  geom_tile(color = "white", linewidth = 0.3) +
  geom_text(aes(label = round(value, 2)), size = 3) +
  scale_fill_gradient(low = "#c51b8a", high = "#fde0dd", name = metric) +
  labs(
    title = plot_title,
    x = "Feature Set",
    y = "Model"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5),
    axis.text.x = element_text(angle = 45, hjust = 1),
    panel.grid = element_blank()
  )

dir.create(dirname(output_file), recursive = TRUE, showWarnings = FALSE)

ggsave(
  output_file,
  p,
  width = 7,
  height = 10,
  dpi = 300
)