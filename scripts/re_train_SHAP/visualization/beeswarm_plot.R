#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
input_file <- args[which(args == "--input") + 1]
output_file <- args[which(args == "--output") + 1]
plot_title <- args[which(args == "--title") + 1]

library(ggplot2)

df <- read.csv(input_file)

feature_order <- aggregate(abs(SHAPValue) ~ Feature, data = df, FUN = mean)
feature_order <- feature_order[order(feature_order$`abs(SHAPValue)`, decreasing = TRUE), "Feature"]
df$Feature <- factor(df$Feature, levels = rev(feature_order))

p <- ggplot(df, aes(x = SHAPValue, y = Feature, color = FeatureValue)) +
  geom_jitter(height = 0.25, width = 0, alpha = 0.7, size = 1.2) +
  labs(x = "SHAP value", y = "Feature", color = "Feature value", title = plot_title) +
  theme_classic() +
  theme(axis.text.y = element_text(size = 8))

ggsave(output_file, p, width = 10, height = 8)