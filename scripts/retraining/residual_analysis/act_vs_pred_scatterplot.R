#!/usr/bin/env Rscript

library(ggplot2)

# Usage:
# Rscript scatterplot.R input.csv output.png
args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 2) {
  stop("Usage: Rscript scatterplot.R <input.csv> <output.png>")
}

input_file  <- args[1]
output_file <- args[2]

# Read dataset
df <- read.csv(input_file)

required_cols <- c("Actual", "Predicted")
if (!all(required_cols %in% colnames(df))) {
  stop("Input CSV must contain columns: Actual, Predicted")
}

df <- df[complete.cases(df[, required_cols]), ]

if (nrow(df) < 3) {
  stop("At least 3 complete observations are required.")
}

# Linear regression
model <- lm(Predicted ~ Actual, data = df)

m  <- unname(coef(model)[2])
c0 <- unname(coef(model)[1])
r2 <- summary(model)$r.squared
r  <- cor(df$Actual, df$Predicted)

# Equation and statistics
equation <- sprintf(
  "y = %.3fx %s %.3f",
  m, ifelse(c0 >= 0, "+", "-"), abs(c0)
)

annotation <- sprintf(
  "%s\nR² = %.3f\nR = %.3f",
  equation, r2, r
)

# Scatterplot
p <- ggplot(df, aes(x = Actual, y = Predicted)) +
  geom_point(
    color = "#E75480",
    size = 0.5,
    alpha = 0.9
  ) +
  geom_smooth(
    method = "lm",
    formula = y ~ x,
    se = FALSE,
    color = "#555555",
    linewidth = 0.8
  ) +
  annotate(
    "text",
    x = -Inf, y = Inf,
    label = annotation,
    hjust = -0.1,
    vjust = 1.5,
    size = 4.5,
    lineheight = 1.3
  ) +
  labs(
    x = "Actual Binding Affinity",
    y = "Predicted Binding Affinity",
    title = "Actual vs Predicted Binding Affinities - Scatterplot"
  ) +
  coord_cartesian(clip = "off") +
  theme_classic(base_size = 13) +
  theme(
    axis.title = element_text(face = "bold"),
    axis.text = element_text(color = "black"),
    plot.margin = margin(12, 25, 12, 12)
  )

# Save output
# Save output as PDF
ggsave(
  filename = output_file,
  plot = p,
  device = "pdf",
  width = 7,
  height = 6,
  bg = "white"
)

cat("R²:", round(r2, 4), "\n")
cat("R:", round(r, 4), "\n")
cat("Equation:", equation, "\n")
cat("Plot saved to:", output_file, "\n")
