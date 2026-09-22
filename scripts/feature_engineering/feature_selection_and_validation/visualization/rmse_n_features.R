library(ggplot2)
library(ggrepel)

args <- commandArgs(trailingOnly = TRUE)

input_file <- args[1]
output_file <- args[2]
plot_title <- args[3]

df <- read.csv(input_file, stringsAsFactors = FALSE)
df <- df[order(df$n_features), ]
df$rmse_label <- sprintf("%.2f", df$rmse)

p <- ggplot(df, aes(x = n_features, y = rmse)) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 3) +
  geom_text_repel(aes(label = rmse_label), size = 3.5, max.overlaps = Inf, box.padding = 0.3, point.padding = 0.2, segment.alpha = 0.5) +
  scale_x_continuous(breaks = df$n_features) +
  labs(title = plot_title, x = "Number of Features", y = "RMSE") +
  theme_bw(base_size = 12) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"), axis.title = element_text(face = "bold"), panel.grid.minor = element_blank())

ggsave(output_file, p, width = 25, height = 8, dpi = 300)