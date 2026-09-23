#!/usr/bin/env Rscript

library(ggplot2)

args <- commandArgs(trailingOnly = TRUE)

input_file <- args[which(args == "--input") + 1]
output_file <- args[which(args == "--output") + 1]
model_name <- args[which(args == "--model") + 1]
metric <- args[which(args == "--metric") + 1]
plot_title <- args[which(args == "--title") + 1]

df <- read.csv(input_file, stringsAsFactors = FALSE)

mean_col <- paste0(metric, "_mean")
ci_col <- paste0(metric, "_ci")

model_df <- subset(df, model == model_name)

if (nrow(model_df) == 0) {
  stop(paste("Model not found:", model_name))
}

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

model_df[[mean_col]] <- as.numeric(model_df[[mean_col]])
model_df[[ci_col]] <- as.numeric(model_df[[ci_col]])

model_df$lower <- model_df[[mean_col]] - model_df[[ci_col]]
model_df$upper <- model_df[[mean_col]] + model_df[[ci_col]]

p <- ggplot(model_df, aes(x = dataset, y = .data[[mean_col]], group = 1)) +
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
    x = "Feature Set",
    y = metric
  ) +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

y_min <- min(model_df$lower, na.rm = TRUE)
y_max <- max(model_df$upper, na.rm = TRUE)
pad <- 0.05 * (y_max + y_min)

p <- p + coord_cartesian(ylim = c(y_min - pad, y_max + pad))

dir.create(dirname(output_file), recursive = TRUE, showWarnings = FALSE)

ggsave(
  output_file,
  p,
  width = 10,
  height = 6
)