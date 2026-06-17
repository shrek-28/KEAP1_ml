library(tidyverse)

# -----------------------------
# Input data
# -----------------------------
df <- read.csv("/Users/shreyasree/Documents/GitHub/KEAP1_ml/data/stat_tests/average_ranks.csv")

# -----------------------------
# Clean names (remove .csv)
# -----------------------------
df <- df %>%
  mutate(dataset = str_replace(dataset, "\\.csv$", ""))

# -----------------------------
# Order by performance (rank)
# lower = better
# -----------------------------
df <- df %>%
  arrange(average_rank) %>%
  mutate(dataset = factor(dataset, levels = dataset))

# -----------------------------
# Plot
# -----------------------------
p <- ggplot(df, aes(x = dataset, y = average_rank)) +
  geom_col(fill = "#FF46A2") +
  
  geom_text(aes(label = round(average_rank, 2)),
            vjust = -0.4,
            size = 4) +
  
  labs(
    title = "Feature Set Comparison (Average Rank)",
    x = "Feature Set",
    y = "Average Rank (Lower is Better)"
  ) +
  
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    plot.title = element_text(face = "bold", hjust = 0.5),
    panel.grid.major.x = element_blank()
  )

ggsave("/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/statistical_tests/average_rank_comparison.pdf", plot=p, height=12, width=8)
