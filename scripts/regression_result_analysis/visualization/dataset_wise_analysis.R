#!/usr/bin/env Rscript

library(ggplot2)

args <- commandArgs(trailingOnly = TRUE)

input_file <- args[which(args == "--input") + 1]
output_file <- args[which(args == "--output") + 1]
dataset_name <- args[which(args == "--dataset") + 1]
metric <- args[which(args == "--metric") + 1]
plot_title <- args[which(args == "--title") + 1]

df <- read.csv(input_file, stringsAsFactors = FALSE)

mean_col <- paste0(metric, "_mean")
ci_col <- paste0(metric, "_ci")

data_df <- subset(df, dataset == dataset_name)

if (nrow(data_df) == 0) {
  stop(paste("Dataset not found:", dataset_name))
}

data_df$dataset <- sub("\\.csv$", "", data_df$dataset)

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

data_df[[mean_col]] <- as.numeric(data_df[[mean_col]])
data_df[[ci_col]] <- as.numeric(data_df[[ci_col]])

data_df$lower <- data_df[[mean_col]] - data_df[[ci_col]]
data_df$upper <- data_df[[mean_col]] + data_df[[ci_col]]

p <- ggplot(data_df, aes(x = model, y = .data[[mean_col]], group = 1)) +
  geom_line(color = "#FF46A2", linewidth = 1) +
  geom_point(color = "#FF46A2", size = 2) +
  geom_errorbar(
    aes(ymin = lower, ymax = upper),
    width = 0.2,
    linewidth = 0.8,
    color = "#FF46A2"
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

y_min <- min(data_df$lower, na.rm = TRUE)
y_max <- max(data_df$upper, na.rm = TRUE)
pad <- 0.05 * (y_max + y_min)

p <- p + coord_cartesian(ylim = c(y_min - pad, y_max + pad))

dir.create(dirname(output_file), recursive = TRUE, showWarnings = FALSE)

ggsave(
  output_file,
  p,
  width = 10,
  height = 6
)