library(ggplot2)

# --------- HARD-CODED INPUTS ---------
input_file  <- "/Users/shreyasree/Documents/GitHub/KEAP1drugdiscovery/data/combined_scores/combined_valid_scores.csv"          # path to your CSV
column_name <- "Score"       # numeric column
output_file <- "/Users/shreyasree/Documents/GitHub/KEAP1drugdiscovery/plots/representative_analysis/docking_score_histogram.pdf"     # output image
binwidth    <- 0.5                # or set a number like 25

# --------- Load data ---------
df <- read.csv(input_file, stringsAsFactors = FALSE)

if (!(column_name %in% colnames(df))) {
  stop(paste("Column", column_name, "not found"))
}

x <- df[[column_name]]
x <- x[is.finite(x)]

if (!is.numeric(x)) {
  stop("Selected column must be numeric")
}

n <- length(x)

# --------- Statistics ---------
mean_x <- mean(x)
median_x <- median(x)
sd_x <- sd(x)
var_x <- var(x)

skewness_x <- sum((x - mean_x)^3) / (n * sd_x^3)
kurtosis_x <- sum((x - mean_x)^4) / (n * sd_x^4) - 3

label_text <- paste0(
  "Mean = ", round(mean_x, 3), " kcal/mol \n",
  "Median = ", round(median_x, 3), " kcal/mol \n",
  "SD = ", round(sd_x, 3), "\n",
  "Var = ", round(var_x, 3), "\n",
  "Skew = ", round(skewness_x, 3), "\n",
  "Kurtosis = ", round(kurtosis_x, 3)
)

# --------- Clean binning ---------
xmin <- min(x)
xmax <- max(x)

if (is.na(binwidth)) {
  raw_bw <- (xmax - xmin) / 20
  scale <- 10^floor(log10(raw_bw))
  candidates <- c(1, 2, 5, 10) * scale
  binwidth <- candidates[which.min(abs(candidates - raw_bw))]
}

xmin_clean <- floor(xmin / binwidth) * binwidth
xmax_clean <- ceiling(xmax / binwidth) * binwidth

bin_edges <- seq(xmin_clean, xmax_clean, by = binwidth)

counts <- hist(x, breaks = bin_edges, plot = FALSE)

interval_labels <- paste0(
  bin_edges[-length(bin_edges)],
  " - ",
  bin_edges[-1]
)

count_df <- data.frame(
  mids = counts$mids,
  counts = counts$counts,
  labels = interval_labels
)

# Save table
table_file <- sub("\\.png$", "_bin_table.csv", output_file)
write.csv(
  data.frame(
    lower = bin_edges[-length(bin_edges)],
    upper = bin_edges[-1],
    counts = counts$counts
  ),
  table_file,
  row.names = FALSE
)

# --------- Plot ---------
p <- ggplot(data.frame(x = x), aes(x = x)) +
  geom_histogram(
    aes(y = after_stat(count)),
    breaks = bin_edges,
    fill = "#FF46A2",
    color = "white",
    alpha = 0.9
  ) +
  geom_vline(
    aes(xintercept = mean_x, linetype = "Mean"),
    linewidth = 0.5
  ) +
  geom_vline(
    aes(xintercept = median_x, linetype = "Median"),
    linewidth = 0.5
  ) +
  scale_linetype_manual(
    name = "Statistics",
    values = c("Mean" = "dashed", "Median" = "dotted")
  ) +
  geom_text(
    data = count_df,
    aes(x = mids, y = counts, label = counts),
    vjust = -0.3,
    size = 3
  ) +
  annotate(
    "text",
    x = Inf, y = Inf,
    label = label_text,
    hjust = 1.1, vjust = 1.1,
    size = 4
  ) +
  labs(
    title = paste("Distribution of Docking Scores of Representative Molecules"),
    x = "Docking Score",
    y = "Count"
  ) +
  coord_cartesian(clip = "off") +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5),
    axis.title = element_text(face = "bold"),
    panel.grid.minor = element_blank(),
    legend.position = "top"
  )

print(p)
ggsave(output_file, plot = p, width = 10, height = 6, dpi = 300)