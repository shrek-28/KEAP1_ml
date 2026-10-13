
#!/usr/bin/env Rscript

# ============================================================
# SHAP Importance Heatmap
# X-axis: Models
# Y-axis: Features
# Cell annotations: Mean absolute SHAP, rounded to 2 decimals
# ============================================================

# Install if required:
# install.packages("ggplot2")

library(ggplot2)

# ---------------------- INPUTS -------------------------------

args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 1) {
  stop("Usage: Rscript shap_heatmap.R <input_folder> [output.pdf]")
}

input_dir <- args[1]

output_pdf <- if (length(args) >= 2) {
  args[2]
} else {
  file.path(input_dir, "SHAP_importance_heatmap.pdf")
}

# ---------------------- CUSTOMIZATION ------------------------

# Plot labels
PLOT_TITLE <- "Mean Absolute SHAP Feature Importance"
X_LABEL <- "Models"
Y_LABEL <- "Features"
LEGEND_TITLE <- "Mean |SHAP|"

# Heatmap colors: low, midpoint, high
COLOR_LOW  <- "#F7FBFF"
COLOR_MID  <- "#6BAED6"
COLOR_HIGH <- "#08306B"

# Text sizes
TITLE_SIZE <- 16
AXIS_TITLE_SIZE <- 12
AXIS_TEXT_SIZE <- 9
ANNOTATION_SIZE <- 2.8
LEGEND_TEXT_SIZE <- 9

# Annotation decimal places
ANNOTATION_DIGITS <- 2

# PDF dimensions in inches
PDF_WIDTH <- 12
PDF_HEIGHT <- 18

# Ordering options:
# FEATURE_ORDER: "importance" or "alphabetical"
# MODEL_ORDER:   "alphabetical" or "importance"
FEATURE_ORDER <- "importance"
MODEL_ORDER <- "alphabetical"

# X-axis model label angle
X_LABEL_ANGLE <- 45

# ---------------------- READ CSV FILES -----------------------

if (!dir.exists(input_dir)) {
  stop(paste("Input folder does not exist:", input_dir))
}

files <- list.files(
  input_dir,
  pattern = "\\.csv$",
  full.names = TRUE,
  recursive = FALSE
)

if (length(files) == 0) {
  stop("No CSV files found in the input folder.")
}

data_list <- lapply(files, function(file) {

  df <- read.csv(file, check.names = FALSE)

  # Identify feature column
  feature_col <- which(
    tolower(trimws(names(df))) == "feature"
  )[1]

  if (is.na(feature_col)) {
    stop(paste("No 'feature' column in:", file))
  }

  # Identify importance column
  importance_candidates <- c(
    "meanabsshap",
    "mean_abs_shap",
    "mean_abs_shap_value",
    "meanabsshapvalue",
    "importance"
  )

  importance_col <- which(
    tolower(trimws(names(df))) %in% importance_candidates
  )[1]

  if (is.na(importance_col)) {
    stop(paste(
      "No recognized SHAP importance column in:", file,
      "\nAvailable columns:", paste(names(df), collapse = ", ")
    ))
  }

  # Model name = CSV filename without extension
  model_name <- tools::file_path_sans_ext(basename(file))

  result <- data.frame(
    Model = model_name,
    Feature = trimws(as.character(df[[feature_col]])),
    Importance = suppressWarnings(
      as.numeric(df[[importance_col]])
    ),
    stringsAsFactors = FALSE
  )

  # Remove missing or invalid entries
  result <- result[
    !is.na(result$Feature) &
      result$Feature != "" &
      !is.na(result$Importance) &
      is.finite(result$Importance),
  ]

  result
})

shap_data <- do.call(rbind, data_list)

if (is.null(shap_data) || nrow(shap_data) == 0) {
  stop("No valid feature-importance records found.")
}

# ---------------------- HANDLE DUPLICATES --------------------

# Average repeated model-feature combinations, if any
shap_data <- aggregate(
  Importance ~ Model + Feature,
  data = shap_data,
  FUN = mean
)

# Replace _div_ with / for display only
shap_data$FeatureLabel <- gsub(
  "_div_",
  "/",
  shap_data$Feature,
  ignore.case = TRUE
)

# ---------------------- ORDER AXES ---------------------------

# Feature ordering
feature_means <- aggregate(
  Importance ~ Feature,
  data = shap_data,
  FUN = mean
)

if (FEATURE_ORDER == "importance") {

  feature_levels <- feature_means$Feature[
    order(feature_means$Importance, decreasing = TRUE)
  ]

} else {

  feature_levels <- sort(unique(shap_data$Feature))

}

# Model ordering
model_means <- aggregate(
  Importance ~ Model,
  data = shap_data,
  FUN = mean
)

if (MODEL_ORDER == "importance") {

  model_levels <- model_means$Model[
    order(model_means$Importance, decreasing = TRUE)
  ]

} else {

  model_levels <- sort(unique(shap_data$Model))

}

# Map original feature names to display labels
feature_label_map <- setNames(
  shap_data$FeatureLabel[
    match(feature_levels, shap_data$Feature)
  ],
  feature_levels
)

# Features on y-axis, highest mean importance at the top
shap_data$FeatureLabel <- factor(
  unname(feature_label_map[shap_data$Feature]),
  levels = rev(unname(feature_label_map[feature_levels]))
)

# Models on x-axis
shap_data$Model <- factor(
  shap_data$Model,
  levels = model_levels
)

# ---------------------- CREATE HEATMAP -----------------------

p <- ggplot(
  shap_data,
  aes(
    x = Model,
    y = FeatureLabel,
    fill = Importance
  )
) +
  geom_tile(
    color = "white",
    linewidth = 0.3
  ) +
  geom_text(
    aes(
      label = formatC(
        Importance,
        format = "f",
        digits = ANNOTATION_DIGITS
      )
    ),
    size = ANNOTATION_SIZE
  ) +
  scale_fill_gradientn(
    colors = c(
      COLOR_LOW,
      COLOR_MID,
      COLOR_HIGH
    ),
    name = LEGEND_TITLE,
    na.value = "white"
  ) +
  labs(
    title = PLOT_TITLE,
    x = X_LABEL,
    y = Y_LABEL
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.title = element_text(
      size = TITLE_SIZE,
      face = "bold",
      hjust = 0.5
    ),

    axis.title.x = element_text(
      size = AXIS_TITLE_SIZE,
      face = "bold"
    ),

    axis.title.y = element_text(
      size = AXIS_TITLE_SIZE,
      face = "bold"
    ),

    axis.text.x = element_text(
      size = AXIS_TEXT_SIZE,
      angle = X_LABEL_ANGLE,
      hjust = 1,
      vjust = 1,
      color = "black"
    ),

    axis.text.y = element_text(
      size = AXIS_TEXT_SIZE,
      color = "black"
    ),

    legend.position = "top",

    legend.title = element_text(
      size = AXIS_TITLE_SIZE
    ),

    legend.text = element_text(
      size = LEGEND_TEXT_SIZE
    ),

    panel.grid = element_blank(),

    plot.margin = margin(
      12, 12, 12, 12
    )
  )

# ---------------------- SAVE PDF -----------------------------

output_dir <- dirname(output_pdf)

if (!dir.exists(output_dir)) {
  dir.create(
    output_dir,
    recursive = TRUE
  )
}

ggsave(
  filename = output_pdf,
  plot = p,
  width = PDF_WIDTH,
  height = PDF_HEIGHT,
  units = "in",
  device = "pdf",
  limitsize = FALSE
)

# ---------------------- SUMMARY -----------------------------

cat("Heatmap saved to:", output_pdf, "\n")
cat("Models:", length(unique(shap_data$Model)), "\n")
cat("Unique features:", length(unique(shap_data$Feature)), "\n")
cat("Annotation precision:", ANNOTATION_DIGITS, "decimal places\n")
