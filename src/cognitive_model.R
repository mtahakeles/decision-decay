# =============================================================================
# cognitive_model.R
# Decision Decay: Modeling Late-Game Cognitive Fatigue
# AQX Sports Analytics Data Bowl 3.0
#
# Reads the cleaned pass event data produced by data_processor.py and fits a
# logistic regression model estimating Expected Pass Completion (xP) as a
# function of match minute, pass distance, and defensive pressure. Then
# visualizes the modeled decline in decision quality as the match approaches
# the 90th minute -- the "decision decay" curve.
# =============================================================================

# ---- 0. Setup ---------------------------------------------------------------
required_packages <- c("ggplot2", "dplyr")
missing_packages <- setdiff(required_packages, rownames(installed.packages()))
if (length(missing_packages) > 0) {
  install.packages(missing_packages, repos = "https://cloud.r-project.org")
}

library(ggplot2)
library(dplyr)

if (!dir.exists("visuals")) dir.create("visuals", recursive = TRUE)

# ---- 1. Load data -------------------------------------------------------
passes <- read.csv("clean_passes.csv", stringsAsFactors = FALSE)

cat("Loaded", nrow(passes), "pass events\n")
cat("Overall completion rate:", round(mean(passes$outcome), 3), "\n\n")

# ---- 2. Fit the Expected Pass Completion (xP) model ----------------------
# outcome  : 1 = pass completed, 0 = incomplete
# minute   : match minute (proxy for accumulated cognitive/physical load)
# distance : pass length in pitch units (longer passes are inherently riskier)
# pressure : 1 if the passer was under defensive pressure
xp_model <- glm(
  outcome ~ minute + distance + pressure,
  data = passes,
  family = binomial(link = "logit")
)

cat("=== Expected Pass Completion (xP) Model Summary ===\n")
print(summary(xp_model))

# Odds-ratio view: easier to communicate in a business/analyst setting
cat("\n=== Odds Ratios ===\n")
print(exp(coef(xp_model)))

# ---- 3. Quantify the "decision decay" effect --------------------------
# Hold distance and pressure at their sample means, and compare predicted
# completion probability in the first 15 minutes vs. the last 15 minutes.
mean_distance <- mean(passes$distance, na.rm = TRUE)
mean_pressure <- mean(passes$pressure, na.rm = TRUE)

early_window <- data.frame(minute = 0:15, distance = mean_distance, pressure = mean_pressure)
late_window  <- data.frame(minute = 75:90, distance = mean_distance, pressure = mean_pressure)

early_xp <- mean(predict(xp_model, newdata = early_window, type = "response"))
late_xp  <- mean(predict(xp_model, newdata = late_window, type = "response"))
decay_pp <- (early_xp - late_xp) * 100

cat(sprintf(
  "\nExpected completion, minutes 0-15:  %.1f%%\n", early_xp * 100
))
cat(sprintf(
  "Expected completion, minutes 75-90: %.1f%%\n", late_xp * 100
))
cat(sprintf(
  "Modeled 'Decision Decay': %.1f percentage-point drop late in the match\n",
  decay_pp
))

# ---- 4. Visualize the decay curve ----------------------------------------
minute_range <- data.frame(
  minute = 0:90,
  distance = mean_distance,
  pressure = mean_pressure
)

minute_range$predicted_xp <- predict(xp_model, newdata = minute_range, type = "response")

decay_plot <- ggplot(minute_range, aes(x = minute, y = predicted_xp)) +
  geom_line(color = "#1b4965", linewidth = 1.4) +
  geom_ribbon(
    aes(ymin = predicted_xp - 0.02, ymax = predicted_xp + 0.02),
    fill = "#1b4965", alpha = 0.12
  ) +
  geom_vline(xintercept = 75, linetype = "dashed", color = "#c1121f", linewidth = 0.6) +
  annotate(
    "text", x = 76, y = max(minute_range$predicted_xp),
    label = "Fatigue window (75'+)", hjust = 0, color = "#c1121f", size = 3.4
  ) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  labs(
    title = "Decision Decay: Expected Pass Completion Across the Match",
    subtitle = sprintf(
      "Modeled xP declines by ~%.1f percentage points from the opening to closing 15 minutes",
      decay_pp
    ),
    x = "Match Minute",
    y = "Expected Pass Completion (xP)",
    caption = "Source: StatsBomb Open Data | Logistic regression: outcome ~ minute + distance + pressure"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", size = 15),
    plot.subtitle = element_text(color = "grey35", margin = margin(b = 10)),
    panel.grid.minor = element_blank(),
    plot.caption = element_text(color = "grey50", size = 8)
  )

# Guard the scales::percent_format call in case the 'scales' package isn't present
if (!requireNamespace("scales", quietly = TRUE)) {
  install.packages("scales", repos = "https://cloud.r-project.org")
}

ggsave("visuals/decision_decay_curve.png", plot = decay_plot, width = 9, height = 5.5, dpi = 300)
cat("\nSaved plot -> visuals/decision_decay_curve.png\n")
