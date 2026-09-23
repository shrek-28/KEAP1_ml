#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
input_file <- args[which(args == "--input") + 1]
output_file <- args[which(args == "--output") + 1]
plot_title <- args[which(args == "--title") + 1]

library(ggplot2)

df <- read.csv(input_file)
df$Feature <- gsub("_div_", "/", df$Feature, fixed = TRUE)

df <- df[order(df$MeanAbsSHAP, decreasing = TRUE), ]
df <- head(df, 20)
df$Feature <- factor(df$Feature, levels = rev(df$Feature))

p <- ggplot(df, aes(x = MeanAbsSHAP, y = Feature)) +
  geom_col(fill="deeppink") +
  geom_text(
    aes(label = sprintf("%.3f", MeanAbsSHAP)),
    hjust = -0.2,
    size = 3
  ) +
  labs(x = "Mean |SHAP value|", y = "Feature", title = plot_title) +
  theme_classic()

output_dir <- dirname(output_file)
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

ggsave(output_file, p, width = 10, height = 8)