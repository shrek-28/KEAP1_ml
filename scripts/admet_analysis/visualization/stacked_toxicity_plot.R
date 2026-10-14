
#!/usr/bin/env Rscript

# Usage:
# Rscript scripts/admet_analysis/visualization/stacked_toxicity_plot.R \
#   data/admet/grouped_nt_summary.csv \
#   plots/admet_analysis/count_bar_plot.pdf

args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 2) {
  stop(
    "Usage: Rscript <script.R> <input.csv> <output.pdf/png/svg>",
    call. = FALSE
  )
}

input_file <- args[1]
output_file <- args[2]

if (!file.exists(input_file)) {
  stop("Input file does not exist: ", input_file, call. = FALSE)
}

if (!requireNamespace("ggplot2", quietly = TRUE)) {
  stop("Install ggplot2 using install.packages('ggplot2')",
       call. = FALSE)
}

library(ggplot2)

# --------------------------------------------------
# 1. Read and validate data
# --------------------------------------------------

data <- read.csv(
  input_file,
  stringsAsFactors = FALSE,
  check.names = FALSE
)

required_cols <- c("Endpoint", "Safe", "Toxic")

if (!all(required_cols %in% names(data))) {
  stop("CSV must contain Endpoint, Safe, and Toxic columns.",
       call. = FALSE)
}

data$Safe <- as.numeric(data$Safe)
data$Toxic <- as.numeric(data$Toxic)

if (
  anyNA(data$Endpoint) ||
  any(data$Endpoint == "") ||
  anyNA(data$Safe) ||
  anyNA(data$Toxic) ||
  any(!is.finite(data$Safe)) ||
  any(!is.finite(data$Toxic)) ||
  any(data$Safe < 0) ||
  any(data$Toxic < 0)
) {
  stop("Endpoints must be present and counts must be non-negative.",
       call. = FALSE)
}

if (anyDuplicated(data$Endpoint)) {
  stop("Each endpoint must appear only once.", call. = FALSE)
}

if (any(data$Safe != floor(data$Safe)) ||
    any(data$Toxic != floor(data$Toxic))) {
  stop("Counts must be whole numbers.", call. = FALSE)
}

data$Endpoint <- factor(
  data$Endpoint,
  levels = data$Endpoint
)

data$Total <- data$Safe + data$Toxic

# --------------------------------------------------
# 2. Calculate percentages
# --------------------------------------------------

data$Safe_pct <- ifelse(
  data$Total > 0,
  100 * data$Safe / pmax(data$Total, 1),
  0
)

data$Toxic_pct <- ifelse(
  data$Total > 0,
  100 * data$Toxic / pmax(data$Total, 1),
  0
)

# Explicit stack order: Safe first (bottom), Toxic second (top)
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

plot_data$Category <- factor(
  plot_data$Category,
  levels = c("Safe", "Toxic")
)

# --------------------------------------------------
# 3. Prepare annotations
# --------------------------------------------------

# Safe count: below the bar baseline
safe_labels <- data[data$Safe > 0, ]

safe_labels$Label_y <- -4

# Toxic count: above the top of the stacked bar
toxic_labels <- data[data$Toxic > 0, ]

toxic_labels$Label_y <- 104

# --------------------------------------------------
# 4. Plot
# --------------------------------------------------

p <- ggplot(
  plot_data,
  aes(x = Endpoint, y = Percentage, fill = Category)
) +
  geom_col(
    width = 0.8,
    color = "white",
    linewidth = 0.25,
    position = position_stack(reverse = TRUE)
  ) +

  # Safe counts below bars
  geom_text(
    data = safe_labels,
    aes(
      x = Endpoint,
      y = Label_y,
      label = Safe
    ),
    inherit.aes = FALSE,
    color = "black",
    fontface = "bold",
    size = 3,
    vjust = 0.5
  ) +

  # Toxic counts above bars
  geom_text(
    data = toxic_labels,
    aes(
      x = Endpoint,
      y = Label_y,
      label = Toxic
    ),
    inherit.aes = FALSE,
    color = "black",
    fontface = "bold",
    size = 3,
    vjust = 0.5
  ) +

  scale_fill_manual(
    values = c(
      "Safe" = "#4C956C",
      "Toxic" = "#D1495B"
    ),
    breaks = c("Safe", "Toxic"),
    limits = c("Safe", "Toxic"),
    drop = FALSE
  ) +

  scale_y_continuous(
    breaks = seq(0, 100, 20),
    labels = function(x) paste0(x, "%"),
    expand = expansion(mult = c(0, 0))
  ) +

  # Extra space below and above bars for counts
  coord_cartesian(
    ylim = c(-8, 110),
    clip = "off"
  ) +

  labs(
    x = "Toxicity endpoint",
    y = "Percentage of molecules (%)",
    title="Distribution of Toxic and Safe Molecules across Different Toxicity Endpoints",
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
    plot.margin = margin(10, 15, 15, 10),
    plot.title = element_text(
      face = "bold",
      hjust = 0.5,
      size = 14
    )
  )

# --------------------------------------------------
# 5. Save plot
# --------------------------------------------------

output_dir <- dirname(output_file)

if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

extension <- tolower(tools::file_ext(output_file))

if (extension == "pdf") {
  ggsave(
    output_file, plot = p,
    width = 13, height = 7,
    device = "pdf"
  )
} else if (extension == "png") {
  ggsave(
    output_file, plot = p,
    width = 13, height = 7,
    dpi = 300, bg = "white"
  )
} else if (extension == "svg") {
  if (!requireNamespace("svglite", quietly = TRUE)) {
    stop("Install svglite using install.packages('svglite')",
         call. = FALSE)
  }

  ggsave(
    output_file, plot = p,
    width = 13, height = 7,
    device = svglite::svglite
  )
} else {
  stop("Unsupported format. Use .pdf, .png, or .svg",
       call. = FALSE)
}

message("Plot saved to: ", normalizePath(output_file))
