# Does pass completion drop late in matches once we account for
# pass difficulty, pressure and score state?
#
# Run from the repo root:  Rscript src/cognitive_model.R

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(lme4)
})

passes <- read.csv("data/passes.csv") |>
  filter(period <= 2) |>                       # extra time only in a few knockout games
  mutate(
    minute = pmin(minute, 95),                 # stoppage time: very few rows past 95'
    score_state = relevel(factor(score_state), ref = "level"),
    height = factor(height),
    dist10 = distance / 10,                    # coefficients per 10 yards
    min15 = minute / 15,                       # coefficients per 15 minutes
    x10 = start_x / 10                         # pitch position, per 10 yards
  )

cat(sprintf("%d open-play passes, %d matches, %d players\n\n",
            nrow(passes), n_distinct(passes$match_id), n_distinct(passes$player)))

# --- 1. raw completion by 15-minute window --------------------------------
raw <- passes |>
  mutate(window = cut(minute, c(-1, 15, 30, 45, 60, 75, 95),
                      labels = c("0-15", "16-30", "31-45", "46-60", "61-75", "76-95"))) |>
  group_by(window) |>
  summarise(n = n(), completion = mean(outcome), mean_dist = mean(distance),
            pressure = mean(pressure), .groups = "drop")
print(raw)

# --- 2. models ----------------------------------------------------------------
# m1: minute only. m2: add pass difficulty. m3: add score state.
# m4: same as m3 with a random intercept per player, because the same
# players pass hundreds of times and those rows are not independent.
m1 <- glm(outcome ~ min15, data = passes, family = binomial)
m2 <- glm(outcome ~ min15 + dist10 + pressure + x10 + height,
          data = passes, family = binomial)
m3 <- update(m2, . ~ . + score_state)
m4 <- glmer(outcome ~ min15 + dist10 + pressure + x10 + height + score_state +
              (1 | player),
            data = passes, family = binomial,
            control = glmerControl(optimizer = "bobyqa"))

minute_row <- function(m, name) {
  est <- summary(m)$coefficients["min15", ]
  data.frame(model = name,
             odds_ratio_per_15min = exp(est[1]),
             ci_low  = exp(est[1] - 1.96 * est[2]),
             ci_high = exp(est[1] + 1.96 * est[2]),
             p_value = est[4])
}
minute_effect <- rbind(
  minute_row(m1, "minute only"),
  minute_row(m2, "+ distance, pressure, position, height"),
  minute_row(m3, "+ score state"),
  minute_row(m4, "+ player random intercept")
)
cat("\nEffect of +15 minutes on the odds of completing a pass:\n")
print(minute_effect, digits = 3, row.names = FALSE)

cat("\nFull mixed model:\n")
print(summary(m4)$coefficients, digits = 3)

dir.create("results", showWarnings = FALSE)
write.csv(raw, "results/completion_by_window.csv", row.names = FALSE)
write.csv(minute_effect, "results/minute_effect.csv", row.names = FALSE)
write.csv(as.data.frame(summary(m4)$coefficients), "results/mixed_model_coefficients.csv")

# --- 3. chart: raw rate vs. model-adjusted rate --------------------------------
# Adjusted curve: average predicted probability over the real passes, with
# only the minute changed (fixed effects, population level).
grid <- 0:95
adjusted <- sapply(grid, function(m) {
  nd <- passes; nd$min15 <- m / 15
  mean(predict(m4, newdata = nd, type = "response", re.form = NA))
})

# Bootstrap-free CI: shift the minute coefficient by +-1.96 SE
b  <- fixef(m4)["min15"]
se <- sqrt(vcov(m4)["min15", "min15"])
base_lp <- predict(m4, newdata = passes, type = "link", re.form = NA) - b * passes$min15
band <- t(sapply(grid, function(m) {
  c(lo = mean(plogis(base_lp + (b - 1.96 * se) * m / 15)),
    hi = mean(plogis(base_lp + (b + 1.96 * se) * m / 15)))
}))
curve <- data.frame(minute = grid, adjusted = adjusted,
                    lo = pmin(band[, "lo"], band[, "hi"]),
                    hi = pmax(band[, "lo"], band[, "hi"]))

by_minute <- passes |>
  mutate(bin = pmin(floor(minute / 5) * 5 + 2.5, 92.5)) |>
  group_by(bin) |>
  summarise(rate = mean(outcome), n = n(), .groups = "drop")

p <- ggplot() +
  geom_point(data = by_minute, aes(bin, rate, size = n), colour = "grey55", alpha = 0.8) +
  geom_ribbon(data = curve, aes(minute, ymin = lo, ymax = hi), fill = "#1b4965", alpha = 0.15) +
  geom_line(data = curve, aes(minute, adjusted), colour = "#1b4965", linewidth = 1.1) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  scale_size_continuous(range = c(1.5, 4), guide = "none") +
  labs(
    title = "Pass completion across the match, 2018 World Cup",
    subtitle = "Grey: raw completion per 5-minute bin. Blue: mixed-model estimate holding pass mix fixed (95% CI)",
    x = "Match minute", y = "Pass completion",
    caption = "Data: StatsBomb Open Data. Open-play passes, regulation time."
  ) +
  theme_minimal(base_size = 12) +
  theme(panel.grid.minor = element_blank(),
        plot.subtitle = element_text(colour = "grey35"))

dir.create("visuals", showWarnings = FALSE)
ggsave("visuals/completion_by_minute.png", p, width = 9, height = 5.2, dpi = 200)
cat("\nSaved visuals/completion_by_minute.png\n")
