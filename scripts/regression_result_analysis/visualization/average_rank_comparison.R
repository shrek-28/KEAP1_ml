#!/usr/bin/env Rscript

library(tidyverse)

args <- commandArgs(trailingOnly = TRUE)

input_file <- args[which(args == "--input") + 1]
output_file <- args[which(args == "--output") + 1]

df <- read.csv(input_file)

df <- df %>%
  mutate(dataset = str_replace(dataset, "\\.csv$", "")) %>%
  arrange(average_rank) %>%
  mutate(dataset = factor(dataset, levels = dataset))

p <- ggplot(df, aes(x = dataset, y = average_rank)) +
  geom_col(fill = "#FF46A2") +
  geom_text(
    aes(label = round(average_rank, 2)),
    vjust = -0.4,
    size = 4
  ) +
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

dir.create(dirname(output_file), recursive = TRUE, showWarnings = FALSE)

ggsave(
  output_file,
  plot = p,
  height = 12,
  width = 8,
  dpi = 300
)