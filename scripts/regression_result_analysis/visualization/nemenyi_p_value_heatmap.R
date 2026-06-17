library(tidyverse)
library(pheatmap)

# -----------------------------
# Load data
# -----------------------------
df <- read.csv("/Users/shreyasree/Documents/GitHub/KEAP1_ml/data/stat_tests/nemenyi_pvalues.csv")

# -----------------------------
# Clean BOTH row + column names
# -----------------------------
df$dataset <- str_replace(df$dataset, "\\.csv$", "")
colnames(df)[-1] <- str_replace(colnames(df)[-1], "\\.csv$", "")

# -----------------------------
# Row names
# -----------------------------
rownames(df) <- df$dataset
df$dataset <- NULL

# -----------------------------
# Convert to matrix
# -----------------------------
mat <- as.matrix(df)
mat <- apply(mat, 2, as.numeric)
rownames(mat) <- rownames(df)

# -----------------------------
# (Optional) fixed order
# -----------------------------
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

# -----------------------------
# Plot heatmap
# -----------------------------
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

ggsave("/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/statistical_tests/p_value_heatmap.pdf", plot=p, height=8, width=8)
