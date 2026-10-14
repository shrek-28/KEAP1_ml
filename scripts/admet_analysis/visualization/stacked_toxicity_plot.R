#!/usr/bin/env Rscript

# Usage:
# Rscript plot_toxicity.R input.csv output.png
# Example:
# Rscript plot_toxicity.R toxicity_data.csv toxicity_plot.png

args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 2) {
  stop(
    "Usage: Rscript plot_toxicity.R <input.csv> <output.png/pdf/svg>",
    call. = FALSE
  )
}

input_file <- args[1]
output_file <- args[2]

# Install missing dependency if necessary
if (!requireNamespace("ggplot2", quietly = TRUE)) {
  stop("Missing package 'ggplot2'. Install it using: install.packages('ggplot2')",
       call. = FALSE)
}

library(ggplot2)

# Read input CSV
data <- read.csv(input_file, stringsAsFactors = FALSE, check.names = FALSE)

required_cols <- c("Endpoint", "Safe", "Toxic")

if (!all(required_cols %in% colnames(data))) {
  stop("Input CSV must contain columns: Endpoint, Safe, Toxic",
       call. = FALSE)
}

if (any(!is.finite(data$Safe)) || any(!is.finite(data$Toxic)) ||
    any(data$Safe < 0) || any(data$Toxic < 0)) {
  stop("Safe and Toxic must contain non-negative numeric counts.",
       call. = FALSE)
}

if (any(data$Safe + data$Toxic == 0)) {
  stop("Each endpoint must have at least one molecule.",
       call. = FALSE)
}

# Preserve the original endpoint order
data$Endpoint <- factor(data$Endpoint, levels = data$Endpoint)

# Convert counts into percentages
data$Total <- data$Safe + data$Toxic
data$Safe_pct <- 100 * data$Safe / data$Total
data$Toxic_pct <- 100 * data$Toxic / data$Total

# Long-format data for stacked bars
plot_data <- rbind(
  data.frame(
    Endpoint = data$Endpoint,
    Category = "Safe",
    Count = data$Safe,
    Percentage = data$Safe_pct
  ),
  data.frame(
    Endpoint = data$Endpoint,
    Category = "Toxic",
    Count = data$Toxic,
    Percentage = data$Toxic_pct
  )
)

# Safe at the bottom; Toxic at the top
plot_data$Category <- factor(
  plot_data$Category,
  levels = c("Safe", "Toxic")
)

# Calculate the vertical midpoint of each stack segment
plot_data <- plot_data[
  order(plot_data$Endpoint, plot_data$Category),
]

plot_data$Label_y <- ave(
  plot_data$Percentage,
  plot_data$Endpoint,
  FUN = function(x) cumsum(x) - x / 2
)

# Annotate counts for non-zero segments
plot_data$Label <- ifelse(
  plot_data$Count > 0,
  as.character(plot_data$Count),
  ""
)

# Create plot
p <- ggplot(
  plot_data,
  aes(x = Endpoint, y = Percentage, fill = Category)
) +
  geom_col(
    width = 0.78,
    color = "white",
    linewidth = 0.25
  ) +
  geom_text(
    aes(y = Label_y, label = Label),
    size = 3,
    color = "black",
    fontface = "bold"
  ) +
  scale_fill_manual(
    values = c(
      "Safe" = "#4C956C",
      "Toxic" = "#D1495B"
    ),
    breaks = c("Safe", "Toxic")
  ) +
  scale_y_continuous(
    limits = c(0, 100),
    breaks = seq(0, 100, 20),
    labels = function(x) paste0(x, "%"),
    expand = expansion(mult = c(0, 0))
  ) +
  labs(
    x = "Toxicity endpoint",
    y = "Percentage of molecules (%)",
    fill = NULL
  ) +
  theme_classic(base_size = 12) +
  theme(
    axis.text.x = element_text(
      angle = 55,
      hjust = 1,
      vjust = 1,
      color = "black",
      size = 9
    ),
    axis.text.y = element_text(color = "black"),
    axis.title = element_text(face = "bold"),
    legend.position = "top",
    legend.text = element_text(size = 10),
    axis.ticks.x = element_line(color = "black"),
    plot.margin = margin(10, 15, 10, 10)
  )

# Save according to output extension
extension <- tolower(tools::file_ext(output_file))

if (extension == "png") {
  ggsave(
    output_file, plot = p,
    width = 13, height = 7,
    dpi = 300, bg = "white"
  )
} else if (extension == "pdf") {
  ggsave(
    output_file, plot = p,
    width = 13, height = 7,
    device = "pdf"
  )
} else if (extension == "svg") {
  if (!requireNamespace("svglite", quietly = TRUE)) {
    stop("For SVG output, install svglite: install.packages('svglite')",
         call. = FALSE)
  }
  ggsave(
    output_file, plot = p,
    width = 13, height = 7,
    device = svglite::svglite
  )
} else {
  stop("Unsupported output format. Use .png, .pdf, or .svg",
       call. = FALSE)
}

message("Plot saved to: ", normalizePath(output_file))
