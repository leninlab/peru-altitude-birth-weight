# ENDES 2020–2024: primary analysis of residential altitude and birth weight
#
# Input: data/processed/ENDES_2020_2024_birth_weight_card.csv
# The input file is an analysis-ready sample exported from the public ENDES
# microdata. It contains the pooled survey weight and year-specific PSU/stratum
# identifiers created during data preparation.
#
# Run this script from the repository root. It produces the main descriptive
# estimates, linear and spline models, adjusted predictions, and Figure 1.

# 0. Setup --------------------------------------------------------------------

required_packages <- c(
  "readr", "dplyr", "survey", "splines", "gtsummary", "gt",
  "ggplot2", "scales", "car"
)

missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_packages) > 0) {
  stop(
    "Install these packages before running the script: ",
    paste(missing_packages, collapse = ", ")
  )
}

library(readr)
library(dplyr)
library(survey)
library(splines)
library(gtsummary)
library(gt)
library(ggplot2)
library(scales)

options(survey.lonely.psu = "adjust")
options(survey.adjust.domain.lonely = TRUE)

input_file <- file.path(
  "data", "processed", "ENDES_2020_2024_birth_weight_card.csv"
)
if (!file.exists(input_file)) {
  stop("Input file not found: ", input_file,
       ". Run the script from the repository root.")
}

endes <- read_csv(input_file, show_col_types = FALSE)

required_variables <- c(
  "M19_peso", "Bajo_peso_nacer_num", "HV040", "Altitud_cat2",
  "V012", "HV270_etiquetada", "HV025", "HV026", "SHREGION",
  "V228_etiquetada", "Numero_hijos_cat", "QI422A_B_etiquetada",
  "Low_birth_weight", "AÑO", "peso_muestral",
  "peso_pool", "PSU_pool", "estrato_pool"
)
missing_variables <- setdiff(required_variables, names(endes))
if (length(missing_variables) > 0) {
  stop("Required variables missing from input: ",
       paste(missing_variables, collapse = ", "))
}

# The pooled weight was defined in data preparation as V005 / 1,000,000 / 5.
# Confirm that the two stored weights retain this relationship.
if (!isTRUE(all.equal(
  endes$peso_pool,
  endes$peso_muestral / 5,
  tolerance = 1e-10,
  check.attributes = FALSE
))) {
  stop("peso_pool is not equal to peso_muestral / 5. Check data preparation.")
}

# 1. Survey design ------------------------------------------------------------

# The file contains the primary analytic sample (health-card birth weights).
# PSU and stratum identifiers are unique across survey years.
design_card <- svydesign(
  ids = ~PSU_pool,
  strata = ~estrato_pool,
  weights = ~peso_pool,
  data = endes,
  nest = TRUE
)

# Create analysis variables and set category order/labels.
design_card <- update(
  design_card,
  altitude_1000m = HV040 / 1000,
  Altitud_cat2 = factor(
    as.character(Altitud_cat2),
    levels = c("<1500 m", "1500–2499 m", "2500–3499 m", "≥3500 m")
  ),
  wealth_quintile = factor(
    HV270_etiquetada,
    levels = c("Los más pobres", "Pobre", "Medio", "Rico", "Más rico"),
    labels = c("Poorest", "Poorer", "Middle", "Richer", "Richest")
  ),
  residence_area = factor(
    HV025,
    levels = c("Urbano", "Rural"),
    labels = c("Urban", "Rural")
  ),
  residence_type = factor(
    HV026,
    levels = c("Capital, gran ciudad", "Pequeña ciudad", "Pueblo", "Campo"),
    labels = c("Capital or large city", "Small city", "Town", "Countryside")
  ),
  natural_region = factor(
    SHREGION,
    levels = c("Lima metropolitana", "Resto Costa", "Sierra", "Selva"),
    labels = c("Metropolitan Lima", "Rest of the Coast", "Highlands", "Amazon region")
  ),
  previous_pregnancy_loss = factor(
    V228_etiquetada,
    levels = c("No", "Sí"),
    labels = c("No", "Yes")
  ),
  number_children_category = factor(
    Numero_hijos_cat,
    levels = c("0–1", "2–3", "≥4")
  ),
  survey_year = factor(AÑO)
)

# Basic design and analytic-sample checks.
print(
  design_card$variables |>
    summarise(
      observations = n(),
      survey_years = n_distinct(AÑO),
      PSUs = n_distinct(PSU_pool),
      strata = n_distinct(estrato_pool),
      missing_weights = sum(is.na(peso_pool)),
      nonpositive_weights = sum(peso_pool <= 0, na.rm = TRUE)
    )
)
print(degf(design_card))

# 2. Descriptive estimates ----------------------------------------------------

# Overall weighted mean birth weight and weighted LBW prevalence.
mean_birth_weight <- svymean(~M19_peso, design_card, na.rm = TRUE)
lbw_prevalence <- svymean(~Bajo_peso_nacer_num, design_card, na.rm = TRUE)

print(mean_birth_weight)
print(confint(mean_birth_weight))
print(lbw_prevalence)
print(confint(lbw_prevalence))

# Weighted birth-weight means and LBW prevalence by altitude category.
birth_weight_by_altitude <- svyby(
  ~M19_peso, ~Altitud_cat2, design_card, svymean,
  na.rm = TRUE, vartype = c("se", "ci"), keep.names = FALSE
)
lbw_by_altitude <- svyby(
  ~Bajo_peso_nacer_num, ~Altitud_cat2, design_card, svymean,
  na.rm = TRUE, vartype = c("se", "ci"), keep.names = FALSE
)
print(birth_weight_by_altitude)
print(lbw_by_altitude)

# Weighted distribution across the four altitude categories.
altitude_distribution <- svymean(~Altitud_cat2, design_card, na.rm = TRUE)
print(altitude_distribution)
print(confint(altitude_distribution))

# Survey-weighted Table 1.
design_card <- update(
  design_card,
  Low_birth_weight = factor(Low_birth_weight, levels = c("No", "Yes")),
  anemia_pregnancy = factor(
    QI422A_B_etiquetada,
    levels = c("No", "Sí"),
    labels = c("No", "Yes")
  )
)

table1 <- tbl_svysummary(
  design_card,
  by = Altitud_cat2,
  include = c(
    HV040, V012, M19_peso, Low_birth_weight, anemia_pregnancy,
    residence_area, residence_type, natural_region, wealth_quintile,
    previous_pregnancy_loss, number_children_category, survey_year
  ),
  label = list(
    HV040 ~ "Cluster altitude, median (IQR), m",
    V012 ~ "Maternal age, mean (SD), years",
    M19_peso ~ "Birth weight, mean (SD), g",
    Low_birth_weight ~ "Low birth weight",
    anemia_pregnancy ~ "Anemia diagnosed during pregnancy",
    residence_area ~ "Area of residence",
    residence_type ~ "Type of place of residence",
    natural_region ~ "Natural region",
    wealth_quintile ~ "Wealth quintile",
    previous_pregnancy_loss ~ "History of miscarriage or stillbirth",
    number_children_category ~ "Number of living children/current pregnancy reported",
    survey_year ~ "Survey year"
  ),
  type = list(
    Low_birth_weight ~ "categorical",
    anemia_pregnancy ~ "categorical",
    previous_pregnancy_loss ~ "categorical"
  ),
  statistic = list(
    HV040 ~ "{median} ({p25}–{p75})",
    V012 ~ "{mean} ({sd})",
    M19_peso ~ "{mean} ({sd})",
    all_categorical() ~ "{n_unweighted} ({p}%)"
  ),
  digits = list(
    HV040 ~ 0,
    V012 ~ 1,
    M19_peso ~ 1,
    all_categorical() ~ c(0, 1)
  ),
  missing = "ifany",
  missing_text = "Missing data"
) |>
  add_overall(last = FALSE) |>
  modify_header(
    label ~ "**Characteristic**",
    all_stat_cols() ~ "**{level}**"
  ) |>
  bold_labels()
print(table1)

# 3. Crude linear and spline models ------------------------------------------

model_crude_linear <- svyglm(
  M19_peso ~ altitude_1000m,
  design = design_card,
  family = gaussian()
)
print(summary(model_crude_linear))
print(confint(model_crude_linear, "altitude_1000m"))

# Survey-weighted 5th, 35th, 65th, and 95th percentile knots.
knots_weighted <- svyquantile(
  ~altitude_1000m,
  design = design_card,
  quantiles = c(0.05, 0.35, 0.65, 0.95),
  ci = FALSE,
  na.rm = TRUE
)
knots_4 <- as.numeric(coef(knots_weighted))
names(knots_4) <- c("P5", "P35", "P65", "P95")
boundary_knots <- knots_4[c("P5", "P95")]
interior_knots <- knots_4[c("P35", "P65")]

spline_basis <- ns(
  design_card$variables$altitude_1000m,
  knots = unname(interior_knots),
  Boundary.knots = unname(boundary_knots),
  intercept = FALSE
)
design_card$variables$spline_1 <- spline_basis[, 1]
design_card$variables$spline_2 <- spline_basis[, 2]
design_card$variables$spline_3 <- spline_basis[, 3]

model_crude_spline <- svyglm(
  M19_peso ~ spline_1 + spline_2 + spline_3,
  design = design_card,
  family = gaussian()
)
crude_global_test <- regTermTest(
  model_crude_spline,
  ~spline_1 + spline_2 + spline_3
)
crude_nonlinearity_test <- anova(
  model_crude_linear,
  model_crude_spline,
  method = "Wald",
  force = TRUE
)
print(crude_global_test)
print(crude_nonlinearity_test)

# 4. Adjusted linear and spline models ---------------------------------------

weighted_mean_age <- as.numeric(coef(svymean(~V012, design_card, na.rm = TRUE)))
design_card <- update(
  design_card,
  maternal_age_centered = V012 - weighted_mean_age,
  maternal_age_centered_sq = (V012 - weighted_mean_age)^2
)

model_adjusted_linear <- svyglm(
  M19_peso ~ altitude_1000m + maternal_age_centered +
    maternal_age_centered_sq + wealth_quintile + residence_area + survey_year,
  design = design_card,
  family = gaussian(),
  na.action = na.omit
)
model_adjusted_spline <- svyglm(
  M19_peso ~ spline_1 + spline_2 + spline_3 + maternal_age_centered +
    maternal_age_centered_sq + wealth_quintile + residence_area + survey_year,
  design = design_card,
  family = gaussian(),
  na.action = na.omit
)

adjusted_global_test <- regTermTest(
  model_adjusted_spline,
  ~spline_1 + spline_2 + spline_3
)
adjusted_nonlinearity_test <- anova(
  model_adjusted_linear,
  model_adjusted_spline,
  method = "Wald",
  force = TRUE
)
print(summary(model_adjusted_linear))
print(confint(model_adjusted_linear, "altitude_1000m"))
print(adjusted_global_test)
print(adjusted_nonlinearity_test)

# Percent attenuation of the linear coefficient after covariate adjustment.
beta_crude <- coef(model_crude_linear)["altitude_1000m"]
beta_adjusted <- coef(model_adjusted_linear)["altitude_1000m"]
attenuation_percent <- (abs(beta_crude) - abs(beta_adjusted)) /
  abs(beta_crude) * 100
print(attenuation_percent)

# 5. Standardized predictions and altitude contrasts -------------------------

# Average the model matrix over the survey-weighted covariate distribution.
model_matrix_observed <- model.matrix(model_adjusted_spline)
model_weights <- weights(model_adjusted_spline, type = "prior")
stopifnot(nrow(model_matrix_observed) == length(model_weights))
mean_model_matrix <- colSums(model_matrix_observed * model_weights) /
  sum(model_weights)

plot_limits <- as.numeric(coef(svyquantile(
  ~altitude_1000m,
  design = design_card,
  quantiles = c(0.01, 0.99),
  ci = FALSE,
  na.rm = TRUE
)))
altitude_grid <- seq(plot_limits[1], plot_limits[2], length.out = 300)
spline_grid <- predict(spline_basis, newx = altitude_grid)

prediction_matrix <- matrix(
  rep(mean_model_matrix, each = length(altitude_grid)),
  nrow = length(altitude_grid),
  byrow = FALSE
)
colnames(prediction_matrix) <- names(mean_model_matrix)
prediction_matrix[, "spline_1"] <- spline_grid[, 1]
prediction_matrix[, "spline_2"] <- spline_grid[, 2]
prediction_matrix[, "spline_3"] <- spline_grid[, 3]

beta <- coef(model_adjusted_spline)
vcov_beta <- vcov(model_adjusted_spline)
predicted_weight <- as.vector(prediction_matrix %*% beta)
prediction_se <- sqrt(rowSums((prediction_matrix %*% vcov_beta) * prediction_matrix))
critical_value <- qt(0.975, df = df.residual(model_adjusted_spline))

prediction_data <- data.frame(
  altitude_m = altitude_grid * 1000,
  predicted_weight = predicted_weight,
  lower_95 = predicted_weight - critical_value * prediction_se,
  upper_95 = predicted_weight + critical_value * prediction_se
)

selected_altitude_km <- c(0, 1, 2, 3, 4)
spline_selected <- predict(spline_basis, newx = selected_altitude_km)
selected_matrix <- matrix(
  rep(mean_model_matrix, each = length(selected_altitude_km)),
  nrow = length(selected_altitude_km),
  byrow = FALSE
)
colnames(selected_matrix) <- names(mean_model_matrix)
selected_matrix[, "spline_1"] <- spline_selected[, 1]
selected_matrix[, "spline_2"] <- spline_selected[, 2]
selected_matrix[, "spline_3"] <- spline_selected[, 3]
selected_predictions <- as.vector(selected_matrix %*% beta)
selected_se <- sqrt(rowSums((selected_matrix %*% vcov_beta) * selected_matrix))

prediction_at_selected_altitudes <- data.frame(
  altitude_m = selected_altitude_km * 1000,
  predicted_mean_g = selected_predictions,
  lower_95 = selected_predictions - critical_value * selected_se,
  upper_95 = selected_predictions + critical_value * selected_se
)
print(prediction_at_selected_altitudes)

contrast_rows <- lapply(seq_along(selected_altitude_km), function(i) {
  reference_contrast <- selected_matrix[i, ] - selected_matrix[1, ]
  reference_estimate <- as.numeric(reference_contrast %*% beta)
  reference_se <- sqrt(as.numeric(reference_contrast %*% vcov_beta %*% reference_contrast))

  if (i == 1) {
    sequential_estimate <- NA_real_
    sequential_lower <- NA_real_
    sequential_upper <- NA_real_
  } else {
    sequential_contrast <- selected_matrix[i, ] - selected_matrix[i - 1, ]
    sequential_estimate <- as.numeric(sequential_contrast %*% beta)
    sequential_se <- sqrt(as.numeric(sequential_contrast %*% vcov_beta %*% sequential_contrast))
    sequential_lower <- sequential_estimate - critical_value * sequential_se
    sequential_upper <- sequential_estimate + critical_value * sequential_se
  }

  data.frame(
    altitude_m = selected_altitude_km[i] * 1000,
    difference_vs_0m_g = reference_estimate,
    vs_0m_lower_95 = reference_estimate - critical_value * reference_se,
    vs_0m_upper_95 = reference_estimate + critical_value * reference_se,
    difference_vs_previous_1000m_g = sequential_estimate,
    vs_previous_lower_95 = sequential_lower,
    vs_previous_upper_95 = sequential_upper
  )
})
altitude_contrasts <- bind_rows(contrast_rows)
print(altitude_contrasts)

# Adjusted exposure–response figure.
figure1 <- ggplot(prediction_data, aes(x = altitude_m, y = predicted_weight)) +
  geom_ribbon(aes(ymin = lower_95, ymax = upper_95), fill = "#4472A8", alpha = 0.20) +
  geom_line(color = "#17365D", linewidth = 1.1) +
  scale_x_continuous(breaks = seq(0, 5000, 1000), labels = label_comma()) +
  scale_y_continuous(labels = label_comma()) +
  labs(
    x = "Residential altitude (m above sea level)",
    y = "Predicted birth weight (g)"
  ) +
  theme_classic(base_size = 11)
print(figure1)

dir.create(file.path("results", "figures"), recursive = TRUE, showWarnings = FALSE)
ggsave(
  filename = file.path("results", "figures", "Figure_1_ENDES_adjusted_spline.png"),
  plot = figure1,
  width = 8,
  height = 5.5,
  units = "in",
  dpi = 600,
  bg = "white"
)

# 6. Model diagnostics --------------------------------------------------------

# Generalized variance inflation factors for the adjusted linear covariates.
diagnostic_model <- lm(
  M19_peso ~ altitude_1000m + maternal_age_centered + maternal_age_centered_sq +
    wealth_quintile + residence_area + survey_year,
  data = design_card$variables,
  weights = peso_pool,
  na.action = na.omit
)
gvif_results <- car::vif(diagnostic_model)
print(gvif_results)

# Condition index for the adjusted spline model matrix (excluding intercept).
spline_model_matrix <- model.matrix(model_adjusted_spline)
spline_model_matrix <- spline_model_matrix[
  , colnames(spline_model_matrix) != "(Intercept)", drop = FALSE
]
condition_index <- kappa(scale(spline_model_matrix), exact = TRUE)
print(condition_index)

# Record software and package versions for reproducibility.
print(R.version.string)
print(sapply(required_packages, packageVersion))
