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

# ---------------------------------------------------------
# Read input
# ---------------------------------------------------------

df <- read.csv(input_file)

df <- df %>%
  mutate(
    dataset = str_replace(dataset, "\\.csv$", "")
  )

# ---------------------------------------------------------
# Calculate model × feature-set means
# ---------------------------------------------------------

heatmap_df <- df %>%
  select(model, dataset, all_of(metric)) %>%
  group_by(model, dataset) %>%
  summarise(
    value = mean(.data[[metric]], na.rm = TRUE),
    .groups = "drop"
  )

# ---------------------------------------------------------
# Main heatmap
# ---------------------------------------------------------

main_df <- heatmap_df %>%
  filter(
    model %in% fixed_model_order,
    dataset %in% fixed_feature_order
  )

# ---------------------------------------------------------
# Row means
# ---------------------------------------------------------

row_means <- main_df %>%
  group_by(model) %>%
  summarise(
    value = mean(value, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    dataset = "RowMean"
  )

# ---------------------------------------------------------
# Column means
# ---------------------------------------------------------

column_means <- main_df %>%
  group_by(dataset) %>%
  summarise(
    value = mean(value, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    model = "ColumnMean"
  )

# ---------------------------------------------------------
# Overall mean
# ---------------------------------------------------------

overall_mean <- main_df %>%
  summarise(
    value = mean(value, na.rm = TRUE)
  ) %>%
  mutate(
    model = "ColumnMean",
    dataset = "RowMean"
  )

# ---------------------------------------------------------
# Factor ordering
# ---------------------------------------------------------

main_df$model <- factor(
  main_df$model,
  levels = fixed_model_order
)

main_df$dataset <- factor(
  main_df$dataset,
  levels = c(
    fixed_feature_order,
    "RowMean"
  )
)

row_means$model <- factor(
  row_means$model,
  levels = fixed_model_order
)

row_means$dataset <- factor(
  row_means$dataset,
  levels = c(
    fixed_feature_order,
    "RowMean"
  )
)

column_means$model <- factor(
  column_means$model,
  levels = c(
    fixed_model_order,
    "ColumnMean"
  )
)

column_means$dataset <- factor(
  column_means$dataset,
  levels = c(
    fixed_feature_order,
    "RowMean"
  )
)

overall_mean$model <- factor(
  overall_mean$model,
  levels = c(
    fixed_model_order,
    "ColumnMean"
  )
)

overall_mean$dataset <- factor(
  overall_mean$dataset,
  levels = c(
    fixed_feature_order,
    "RowMean"
  )
)

# ---------------------------------------------------------
# Plot
# ---------------------------------------------------------

p <- ggplot() +

  # -------------------------------------------------------
  # Main heatmap
  # -------------------------------------------------------

  geom_tile(
    data = main_df,
    aes(
      x = dataset,
      y = model,
      fill = value
    ),
    color = "white",
    linewidth = 0.3
  ) +

  # Main values
  geom_text(
    data = main_df,
    aes(
      x = dataset,
      y = model,
      label = round(value, 2)
    ),
    color = "black",
    size = 3
  ) +

  # -------------------------------------------------------
  # Row mean cells
  # -------------------------------------------------------

  geom_tile(
    data = row_means,
    aes(
      x = dataset,
      y = model
    ),
    fill = "white",
    color = "white",
    linewidth = 0.3
  ) +

  geom_text(
    data = row_means,
    aes(
      x = dataset,
      y = model,
      label = round(value, 2)
    ),
    color = "black",
    size = 3
  ) +

  # -------------------------------------------------------
  # Column mean cells
  # -------------------------------------------------------

  geom_tile(
    data = column_means,
    aes(
      x = dataset,
      y = model
    ),
    fill = "white",
    color = "white",
    linewidth = 0.3
  ) +

  geom_text(
    data = column_means,
    aes(
      x = dataset,
      y = model,
      label = round(value, 2)
    ),
    color = "black",
    size = 3
  ) +

  # -------------------------------------------------------
  # Overall mean cell
  # -------------------------------------------------------

  geom_tile(
    data = overall_mean,
    aes(
      x = dataset,
      y = model
    ),
    fill = "white",
    color = "white",
    linewidth = 0.3
  ) +

  geom_text(
    data = overall_mean,
    aes(
      x = dataset,
      y = model,
      label = round(value, 2)
    ),
    color = "black",
    size = 3
  ) +

  # -------------------------------------------------------
  # Fill scale
  # -------------------------------------------------------

  scale_fill_gradient(
    low = "#c51b8a",
    high = "#fde0dd",
    name = metric,
    guide = guide_colorbar(
      position = "top"
    )
  ) +

  # -------------------------------------------------------
  # Suppress labels for mean row/column
  # -------------------------------------------------------

  scale_x_discrete(
    labels = c(
      fixed_feature_order,
      ""
    )
  ) +

  scale_y_discrete(
    labels = c(
      fixed_model_order,
      ""
    )
  ) +

  # -------------------------------------------------------
  # Labels
  # -------------------------------------------------------

  labs(
    title = plot_title,
    x = "Feature Set",
    y = "Model"
  ) +

  # -------------------------------------------------------
  # Theme
  # -------------------------------------------------------

  theme_minimal() +

  theme(
    plot.title = element_text(
      face = "bold",
      hjust = 0.5,
      color = "black"
    ),

    axis.text.x = element_text(
      angle = 45,
      hjust = 1,
      color = "black"
    ),

    axis.text.y = element_text(
      color = "black"
    ),

    axis.title.x = element_text(
      color = "black"
    ),

    axis.title.y = element_text(
      color = "black"
    ),

    panel.grid = element_blank(),

    legend.position = "top",

    legend.justification = "center"
  )

# ---------------------------------------------------------
# Create output directory
# ---------------------------------------------------------

dir.create(
  dirname(output_file),
  recursive = TRUE,
  showWarnings = FALSE
)

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

ggsave(
  output_file,
  p,
  width = 8,
  height = 10.5,
  dpi = 300
)