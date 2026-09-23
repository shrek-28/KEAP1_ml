#!/usr/bin/env Rscript

library(tidyverse)
library(pheatmap)

args <- commandArgs(trailingOnly = TRUE)

input_file <- args[which(args == "--input") + 1]
output_file <- args[which(args == "--output") + 1]

df <- read.csv(input_file)

df$dataset <- str_replace(df$dataset, "\\.csv$", "")
colnames(df)[-1] <- str_replace(colnames(df)[-1], "\\.csv$", "")

rownames(df) <- df$dataset
df$dataset <- NULL

mat <- as.matrix(df)
mat <- apply(mat, 2, as.numeric)
rownames(mat) <- rownames(df)

row_order <- c(
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

mat <- mat[row_order, , drop = FALSE]

p <- pheatmap(
  mat,
  cluster_rows = FALSE,
  cluster_cols = TRUE,
  display_numbers = round(mat, 3),
  number_color = "black",
  color = colorRampPalette(c("#fde0dd", "#c51b8a"))(100),
  border_color = "white",
  main = "Clustered Heatmap of Feature Set Similarities",
  angle_col = 315
)

dir.create(dirname(output_file), recursive = TRUE, showWarnings = FALSE)

pdf(output_file, height = 8, width = 8)
print(p)
dev.off()