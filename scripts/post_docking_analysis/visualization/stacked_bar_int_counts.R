#!/usr/bin/env Rscript

# Usage:
# Rscript scripts/admet_analysis/visualization/interaction_count_plot.R \
#   data/admet/your_interaction_data.csv \
#   data/admet/bonds_of_interest.txt \
#   plots/admet_analysis/interaction_count_plot.pdf

args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 3) {
  stop(
    paste(
      "Usage: Rscript interaction_count_plot.R",
      "<input.csv> <bonds_of_interest.txt> <output.pdf/png/svg>"
    ),
    call. = FALSE
  )
}

input_file <- args[1]
bonds_file <- args[2]
output_file <- args[3]

# Check input files
if (!file.exists(input_file)) {
  stop("Input CSV not found: ", input_file, call. = FALSE)
}

if (!file.exists(bonds_file)) {
  stop("Bonds-of-interest file not found: ", bonds_file, call. = FALSE)
}

if (!requireNamespace("ggplot2", quietly = TRUE)) {
  stop("Install ggplot2 using install.packages('ggplot2')", call. = FALSE)
}

library(ggplot2)

# ------------------------------------------------------------
# 1. Read and clean input data
# ------------------------------------------------------------

data <- read.csv(
  input_file,
  stringsAsFactors = FALSE,
  check.names = FALSE,
  na.strings = c("", "NA")
)

# Remove Markdown asterisks from column names, if present
names(data) <- gsub("^\\*+|\\*+$", "", names(data))

required_cols <- c(
  "CNP_ID",
  "n_conformations_with_bonds_of_interest"
)

missing_cols <- setdiff(required_cols, names(data))

if (length(missing_cols) > 0) {
  stop(
    "Missing required columns: ",
    paste(missing_cols, collapse = ", "),
    call. = FALSE
  )
}

# ------------------------------------------------------------
# 2. Read bonds/interactions of interest
# ------------------------------------------------------------

bonds <- trimws(readLines(bonds_file, warn = FALSE))

# Remove empty lines and comments
bonds <- bonds[
  nzchar(bonds) & !grepl("^#", bonds)
]

# Remove optional surrounding quotation marks
bonds <- gsub('^["\']|["\']$', "", bonds)
bonds <- unique(trimws(bonds))
bonds <- bonds[nzchar(bonds)]

if (length(bonds) == 0) {
  stop("No interactions found in the bonds-of-interest file.", call. = FALSE)
}

# ------------------------------------------------------------
# 3. Exclude molecules with exactly 9 conformations
# ------------------------------------------------------------

data$n_conformations_with_bonds_of_interest <- as.numeric(
  data$n_conformations_with_bonds_of_interest
)

n_before <- nrow(data)

data <- data[
  !is.na(data$n_conformations_with_bonds_of_interest) &
    data$n_conformations_with_bonds_of_interest != 9,
  ,
  drop = FALSE
]

message("Molecules before filtering: ", n_before)
message("Molecules after filtering: ", nrow(data))
message("Molecules excluded: ", n_before - nrow(data))

# ------------------------------------------------------------
# 4. Identify hydrogen-bond and hydrophobic-interaction columns
# ------------------------------------------------------------

hb_cols <- grep(
  "^conf_[0-9]+_h_bonds_of_interest$",
  names(data),
  value = TRUE
)

hyd_cols <- grep(
  "^conf_[0-9]+_hydrophobic_bonds_of_interest$",
  names(data),
  value = TRUE
)

if (length(hb_cols) == 0) {
  stop("No hydrogen-bond interaction columns found.", call. = FALSE)
}

if (length(hyd_cols) == 0) {
  stop("No hydrophobic-interaction columns found.", call. = FALSE)
}

message("Hydrogen-bond columns detected: ", length(hb_cols))
message("Hydrophobic-interaction columns detected: ", length(hyd_cols))

# ------------------------------------------------------------
# 5. Parse interaction strings
# ------------------------------------------------------------

parse_interactions <- function(x) {

  if (
    length(x) == 0 ||
    is.na(x[1]) ||
    !nzchar(trimws(as.character(x[1])))
  ) {
    return(character(0))
  }

  tokens <- unlist(
    strsplit(as.character(x[1]), "|", fixed = TRUE),
    use.names = FALSE
  )

  tokens <- trimws(tokens)

  tokens <- tokens[
    nzchar(tokens) &
      tokens != "**" &
      !grepl("^0$", tokens)
  ]

  unique(tokens)
}

# Determine whether a molecule contains a target interaction
# in any of its conformations.

has_interaction <- function(row, columns, target) {

  present <- unlist(
    lapply(
      columns,
      function(column) {
        parse_interactions(row[[column]][1])
      }
    ),
    use.names = FALSE
  )

  target %in% present
}

# ------------------------------------------------------------
# 6. Count unique molecules for each interaction
# ------------------------------------------------------------

results <- data.frame(
  Interaction = bonds,
  Hydrogen_Bond = integer(length(bonds)),
  Hydrophobic = integer(length(bonds)),
  stringsAsFactors = FALSE
)

for (i in seq_along(bonds)) {

  target <- bonds[i]

  hb_present <- vapply(
    seq_len(nrow(data)),
    function(j) {
      has_interaction(
        data[j, , drop = FALSE],
        hb_cols,
        target
      )
    },
    logical(1)
  )

  hyd_present <- vapply(
    seq_len(nrow(data)),
    function(j) {
      has_interaction(
        data[j, , drop = FALSE],
        hyd_cols,
        target
      )
    },
    logical(1)
  )

  # Each molecule is counted at most once per interaction category,
  # regardless of how many conformations contain that interaction.

  results$Hydrogen_Bond[i] <- sum(hb_present)
  results$Hydrophobic[i] <- sum(hyd_present)
}

# Total is the sum of the two category-specific molecule counts.
# A molecule present in both categories contributes to both.

results$Total_Interactions <- (
  results$Hydrogen_Bond + results$Hydrophobic
)

# Save counts table alongside the plot
output_dir <- dirname(output_file)

if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

counts_file <- paste0(
  tools::file_path_sans_ext(output_file),
  "_counts.csv"
)

write.csv(results, counts_file, row.names = FALSE)

# ------------------------------------------------------------
# 7. Calculate bar and annotation positions
# ------------------------------------------------------------

results$x <- seq_len(nrow(results))

# Hydrogen bonds form the bottom (green) segment
results$hb_bottom <- 0
results$hb_top <- results$Hydrogen_Bond

# Hydrophobic interactions form the top (red) segment
results$hyd_bottom <- results$Hydrogen_Bond
results$hyd_top <- (
  results$Hydrogen_Bond + results$Hydrophobic
)

max_count <- max(results$hyd_top, 1)

# Spacing for count labels
label_offset <- max(0.5, max_count * 0.025)

# Hydrogen-bond count: below the bar baseline
results$hb_label_y <- -label_offset

# Hydrophobic count: immediately above the complete stacked bar
results$hyd_label_y <- results$hyd_top + label_offset


# ------------------------------------------------------------
# 8. Create stacked bar plot
# ------------------------------------------------------------

p <- ggplot() +

  # Hydrogen-bond segment (bottom)
  geom_rect(
    data = results,
    aes(
      xmin = x - 0.4,
      xmax = x + 0.4,
      ymin = hb_bottom,
      ymax = hb_top,
      fill = "Hydrogen bond"
    ),
    color = "white",
    linewidth = 0.25
  ) +

  # Hydrophobic-interaction segment (top)
  geom_rect(
    data = results,
    aes(
      xmin = x - 0.4,
      xmax = x + 0.4,
      ymin = hyd_bottom,
      ymax = hyd_top,
      fill = "Hydrophobic interaction"
    ),
    color = "white",
    linewidth = 0.25
  ) +

  # Hydrogen-bond count below the bar
  geom_text(
    data = results,
    aes(
      x = x,
      y = hb_label_y,
      label = Hydrogen_Bond
    ),
    inherit.aes = FALSE,
    color = "black",
    fontface = "bold",
    size = 3,
    vjust = 0.5
  ) +

  # Hydrophobic-interaction count above the bar
  geom_text(
    data = results,
    aes(
      x = x,
      y = hyd_label_y,
      label = Hydrophobic
    ),
    inherit.aes = FALSE,
    color = "black",
    fontface = "bold",
    size = 3,
    vjust = 0.5
  ) +

  # Total interaction count at the top of the plotting area
  geom_text(
    data = results,
    aes(
        x = x,
        y = Inf,
        label = Total_Interactions
    ),
    inherit.aes = FALSE,
    color = "black",
    fontface = "bold",
    size = 3.2,
    vjust = 1.5
  ) +

  # Segment colors and legend
  scale_fill_manual(
    name = NULL,
    values = c(
      "Hydrogen bond" = "#4C956C",
      "Hydrophobic interaction" = "#D1495B"
    ),
    breaks = c(
      "Hydrogen bond",
      "Hydrophobic interaction"
    ),
    drop = FALSE
  ) +

  # Interaction labels on x-axis
  scale_x_continuous(
    breaks = results$x,
    labels = results$Interaction,
    expand = expansion(add = 0.6)
  ) +

  # Avoid requiring the scales package
  scale_y_continuous(
    breaks = pretty(c(0, max(results$total_label_y)), n = 6),
    expand = expansion(mult = c(0, 0))
  ) +

  labs(
    title = "Interaction-wise Molecule Counts",
    x = "Interaction of interest",
    y = "Number of molecules"
  ) +

  # Reserve room below the baseline and above the total labels
  coord_cartesian(
  ylim = c(-2 * label_offset, max(results$hyd_top) * 1.15),
  clip = "off")  +

  theme_classic(base_size = 12) +

  theme(
    plot.title = element_text(
      face = "bold",
      size = 14,
      hjust = 0.5
    ),
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
    legend.direction = "horizontal",
    legend.text = element_text(size = 10),
    plot.margin = margin(12, 15, 18, 10)
  )

# ------------------------------------------------------------
# 9. Save plot
# ------------------------------------------------------------

extension <- tolower(tools::file_ext(output_file))

plot_width <- max(10, 0.65 * nrow(results))

if (extension == "pdf") {

  ggsave(
    filename = output_file,
    plot = p,
    width = plot_width,
    height = 7,
    device = "pdf"
  )

} else if (extension == "png") {

  ggsave(
    filename = output_file,
    plot = p,
    width = plot_width,
    height = 7,
    dpi = 300
  )

} else if (extension == "svg") {

  if (!requireNamespace("svglite", quietly = TRUE)) {
    stop(
      "Install svglite using install.packages('svglite') to save SVG.",
      call. = FALSE
    )
  }

  ggsave(
    filename = output_file,
    plot = p,
    width = plot_width,
    height = 7,
    device = svglite::svglite
  )

} else {

  stop(
    "Unsupported output format. Use .pdf, .png, or .svg.",
    call. = FALSE
  )
}

message("Plot saved to: ", normalizePath(output_file))
message("Counts table saved to: ", normalizePath(counts_file))