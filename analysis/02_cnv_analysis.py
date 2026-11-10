# %% [markdown]
# # CNV analysis: provincial-capital altitude and low birth weight
#
# This script reproduces the descriptive summaries and regression analyses for
# the province-year CNV dataset, Peru, 2020-2024. Run it from the repository
# root. The input is the analysis-ready CSV in `data/processed/`.

# %%
from pathlib import Path
from importlib.metadata import version

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.cov_struct import Exchangeable, Independence, Autoregressive
from patsy import build_design_matrices

# %% [markdown]
# ## 1. Input data and analysis setup
#
# The CSV contains one row per province-year. Birth-weight counts and
# provincial-capital altitude have already been harmonized and combined.

# %%
DATA_FILE = "cnv_province_year_2020_2024.csv"


def find_project_root(start: Path = Path.cwd()) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "data" / "processed" / DATA_FILE).exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find data/processed/{DATA_FILE}. "
        "Run this script from the repository or project directory."
    )


PROJECT_ROOT = find_project_root()
DATA_PATH = PROJECT_ROOT / "data" / "processed" / DATA_FILE
OUTPUT_DIR = PROJECT_ROOT / "results" / "cnv"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

cnv = pd.read_csv(DATA_PATH)
required_columns = {
    "id_provincia", "departamento", "provincia", "año",
    "altitud", "casos_bpn", "total_nacimientos"
}
missing_columns = required_columns.difference(cnv.columns)
if missing_columns:
    raise ValueError(f"Required columns are missing: {sorted(missing_columns)}")

cnv["año"] = pd.to_numeric(cnv["año"], errors="raise").astype(int)
for column in ["altitud", "casos_bpn", "total_nacimientos"]:
    cnv[column] = pd.to_numeric(cnv[column], errors="raise")

if cnv["id_provincia"].isna().any() or cnv["año"].isna().any():
    raise ValueError("Province identifiers and years must not be missing.")
if cnv.duplicated(["id_provincia", "año"]).any():
    raise ValueError("The dataset contains duplicate province-year rows.")
if cnv[["altitud", "casos_bpn", "total_nacimientos"]].isna().any().any():
    raise ValueError("Altitude and birth-count variables must not be missing.")
if (cnv["total_nacimientos"] <= 0).any():
    raise ValueError("All province-year records must have positive birth counts.")
if (cnv["casos_bpn"] < 0).any() or (cnv["casos_bpn"] > cnv["total_nacimientos"]).any():
    raise ValueError("LBW counts must be between zero and total births.")

# Counts in the input CSV may be read as floats; convert after checking validity.
cnv["casos_bpn"] = cnv["casos_bpn"].astype(int)
cnv["total_nacimientos"] = cnv["total_nacimientos"].astype(int)

print(f"Input file: {DATA_PATH.name}")
print(f"Province-year observations: {len(cnv):,}")
print(f"Provinces: {cnv['id_provincia'].nunique():,}")
print(f"Years: {sorted(cnv['año'].unique())}")
print(f"Python package versions: pandas {version('pandas')}, "
      f"NumPy {version('numpy')}, statsmodels {version('statsmodels')}, "
      f"patsy {version('patsy')}")

# %% [markdown]
# ## 2. Descriptive summaries
#
# Prevalence is calculated from summed LBW cases and summed registered births,
# rather than by averaging province-year percentages.

# %%
annual = (
    cnv.groupby("año", as_index=False)
    .agg(
        provinces=("id_provincia", "nunique"),
        province_years=("id_provincia", "size"),
        lbw_cases=("casos_bpn", "sum"),
        births=("total_nacimientos", "sum")
    )
)
annual["lbw_prevalence_pct"] = 100 * annual["lbw_cases"] / annual["births"]

overall = pd.DataFrame({
    "period": [f"{cnv['año'].min()}–{cnv['año'].max()}"],
    "provinces": [cnv["id_provincia"].nunique()],
    "province_years": [len(cnv)],
    "lbw_cases": [cnv["casos_bpn"].sum()],
    "births": [cnv["total_nacimientos"].sum()]
})
overall["lbw_prevalence_pct"] = 100 * overall["lbw_cases"] / overall["births"]

altitude_labels = ["<1500 m", "1500–2499 m", "2500–3499 m", "≥3500 m"]
cnv["altitude_category"] = pd.cut(
    cnv["altitud"],
    bins=[-np.inf, 1500, 2500, 3500, np.inf],
    labels=altitude_labels,
    right=False,
    ordered=True
)

by_altitude = (
    cnv.groupby("altitude_category", observed=False, as_index=False)
    .agg(
        provinces=("id_provincia", "nunique"),
        province_years=("id_provincia", "size"),
        lbw_cases=("casos_bpn", "sum"),
        births=("total_nacimientos", "sum")
    )
)
by_altitude["lbw_prevalence_pct"] = 100 * by_altitude["lbw_cases"] / by_altitude["births"]

cnv["altitude_2500"] = np.where(cnv["altitud"] < 2500, "<2500 m", "≥2500 m")
by_altitude_2500 = (
    cnv.groupby("altitude_2500", as_index=False)
    .agg(
        provinces=("id_provincia", "nunique"),
        lbw_cases=("casos_bpn", "sum"),
        births=("total_nacimientos", "sum")
    )
)
by_altitude_2500["lbw_prevalence_pct"] = (
    100 * by_altitude_2500["lbw_cases"] / by_altitude_2500["births"]
)

print("\nOverall prevalence")
print(overall.round(2).to_string(index=False))
print("\nAnnual prevalence")
print(annual.round(2).to_string(index=False))
print("\nPooled prevalence by altitude category")
print(by_altitude.round(2).to_string(index=False))
print("\nPooled prevalence below/above 2,500 m")
print(by_altitude_2500.round(2).to_string(index=False))

overall.to_csv(OUTPUT_DIR / "cnv_descriptive_overall.csv", index=False)
annual.to_csv(OUTPUT_DIR / "cnv_descriptive_annual.csv", index=False)
by_altitude.to_csv(OUTPUT_DIR / "cnv_descriptive_by_altitude.csv", index=False)
by_altitude_2500.to_csv(OUTPUT_DIR / "cnv_descriptive_altitude_2500m.csv", index=False)

# %% [markdown]
# ## 3. Continuous-altitude Poisson models
#
# The outcome is the number of LBW births. The log of total births is the
# offset. Models adjust for year, with province as the repeated-measures or
# clustering unit.

# %%
cnv["altitude_1000m"] = cnv["altitud"] / 1000
cnv["year_factor"] = pd.Categorical(
    cnv["año"], categories=sorted(cnv["año"].unique()), ordered=True
)
cnv["log_births"] = np.log(cnv["total_nacimientos"])


def extract_altitude_effect(model, model_name, correlation=np.nan):
    term = "altitude_1000m"
    beta = model.params[term]
    ci = model.conf_int().loc[term]
    return {
        "model": model_name,
        "PR_per_1000m": np.exp(beta),
        "CI95_lower": np.exp(ci.iloc[0]),
        "CI95_upper": np.exp(ci.iloc[1]),
        "p_value": model.pvalues[term],
        "working_correlation": correlation
    }


# Primary: Poisson GEE with exchangeable working correlation.
gee_exchangeable = smf.gee(
    "casos_bpn ~ altitude_1000m + C(year_factor)",
    groups="id_provincia",
    data=cnv,
    family=sm.families.Poisson(),
    cov_struct=Exchangeable(),
    offset=cnv["log_births"]
).fit()

rho_exchangeable = gee_exchangeable.cov_struct.dep_params
primary_result = extract_altitude_effect(
    gee_exchangeable, "Exchangeable Poisson GEE", rho_exchangeable
)
print("Primary continuous-altitude model")
print(pd.DataFrame([primary_result]).round(3).to_string(index=False))

# %% [markdown]
# ## 4. Categorical-altitude Poisson model
#
# Province-clustered robust standard errors are used. The <1,500-m category is
# the reference; an additional contrast compares the two upper categories.

# %%
category_formula = (
    'casos_bpn ~ C(altitude_category, Treatment(reference="<1500 m")) '
    '+ C(year_factor)'
)
category_model = smf.glm(
    category_formula,
    data=cnv,
    family=sm.families.Poisson(),
    offset=cnv["log_births"]
).fit(cov_type="cluster", cov_kwds={"groups": cnv["id_provincia"]})

category_terms = [
    name for name in category_model.params.index
    if "altitude_category" in name
]
category_results = pd.DataFrame({
    "contrast": category_terms,
    "PR": np.exp(category_model.params[category_terms].to_numpy()),
    "CI95_lower": np.exp(category_model.conf_int().loc[category_terms, 0].to_numpy()),
    "CI95_upper": np.exp(category_model.conf_int().loc[category_terms, 1].to_numpy()),
    "p_value": category_model.pvalues[category_terms].to_numpy()
})

category_names = list(category_model.params.index)
category_indices = [category_names.index(term) for term in category_terms]
R_category = np.zeros((len(category_indices), len(category_names)))
for row, column in enumerate(category_indices):
    R_category[row, column] = 1
category_global_test = category_model.wald_test(R_category, scalar=True)

upper_formula = (
    'casos_bpn ~ '
    'C(altitude_category, Treatment(reference="2500–3499 m")) '
    '+ C(year_factor)'
)
upper_contrast_model = smf.glm(
    upper_formula,
    data=cnv,
    family=sm.families.Poisson(),
    offset=cnv["log_births"]
).fit(cov_type="cluster", cov_kwds={"groups": cnv["id_provincia"]})
upper_term = next(
    name for name in upper_contrast_model.params.index
    if "altitude_category" in name and "≥3500 m" in name
)
upper_ci = upper_contrast_model.conf_int().loc[upper_term]
upper_result = pd.DataFrame([{
    "contrast": "≥3500 m vs 2500–3499 m",
    "PR": np.exp(upper_contrast_model.params[upper_term]),
    "CI95_lower": np.exp(upper_ci.iloc[0]),
    "CI95_upper": np.exp(upper_ci.iloc[1]),
    "p_value": upper_contrast_model.pvalues[upper_term]
}])

print("Categorical-altitude contrasts")
print(category_results.round(3).to_string(index=False))
print(f"Global altitude-category Wald chi-square={float(category_global_test.statistic):.2f}; "
      f"df={int(category_global_test.df_denom)}; p={float(category_global_test.pvalue):.3f}")
print("Upper-category contrast")
print(upper_result.round(3).to_string(index=False))
category_results.to_csv(OUTPUT_DIR / "cnv_categorical_altitude_model.csv", index=False)
upper_result.to_csv(OUTPUT_DIR / "cnv_upper_altitude_contrast.csv", index=False)

# %% [markdown]
# ## 5. Restricted cubic spline model
#
# Four knots are placed at birth-weighted 5th, 35th, 65th, and 95th percentiles
# of provincial-capital altitude. The lower and upper knots are boundaries.

# %%
def weighted_percentiles(values, percentiles, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    percentiles = np.asarray(percentiles, dtype=float)
    order = np.argsort(values)
    values_sorted = values[order]
    weights_sorted = weights[order]
    cumulative = (np.cumsum(weights_sorted) - 0.5 * weights_sorted) / weights_sorted.sum()
    return np.interp(percentiles, cumulative, values_sorted)


def restricted_cubic_spline_basis(x, knots):
    x = np.asarray(x, dtype=float)
    knots = np.asarray(knots, dtype=float)
    if len(knots) != 4 or np.any(np.diff(knots) <= 0):
        raise ValueError("Four unique, ordered spline knots are required.")
    last, penultimate = knots[-1], knots[-2]
    basis = []
    for knot in knots[:-2]:
        term = (
            np.maximum(x - knot, 0) ** 3
            - np.maximum(x - penultimate, 0) ** 3 * (last - knot) / (last - penultimate)
            + np.maximum(x - last, 0) ** 3 * (penultimate - knot) / (last - penultimate)
        )
        term = term / (knots[-1] - knots[0]) ** 2
        basis.append(term)
    return np.column_stack(basis)


knots = weighted_percentiles(
    cnv["altitude_1000m"],
    [0.05, 0.35, 0.65, 0.95],
    cnv["total_nacimientos"]
)
spline_basis = restricted_cubic_spline_basis(cnv["altitude_1000m"], knots)
cnv["spline_nl1"] = spline_basis[:, 0]
cnv["spline_nl2"] = spline_basis[:, 1]

spline_model = smf.glm(
    "casos_bpn ~ altitude_1000m + spline_nl1 + spline_nl2 + C(year_factor)",
    data=cnv,
    family=sm.families.Poisson(),
    offset=cnv["log_births"]
).fit(cov_type="cluster", cov_kwds={"groups": cnv["id_provincia"]})


def joint_wald_test(model, terms):
    names = list(model.params.index)
    indices = [names.index(term) for term in terms]
    contrast = np.zeros((len(indices), len(names)))
    for row, column in enumerate(indices):
        contrast[row, column] = 1
    return model.wald_test(contrast, scalar=True)


overall_spline_test = joint_wald_test(
    spline_model, ["altitude_1000m", "spline_nl1", "spline_nl2"]
)
nonlinearity_test = joint_wald_test(spline_model, ["spline_nl1", "spline_nl2"])
print("Spline knot locations (m):", np.round(knots * 1000, 1))
print(f"Overall spline Wald chi-square={float(overall_spline_test.statistic):.2f}; "
      f"df={int(overall_spline_test.df_denom)}; p={float(overall_spline_test.pvalue):.3g}")
print(f"Nonlinearity Wald chi-square={float(nonlinearity_test.statistic):.2f}; "
      f"df={int(nonlinearity_test.df_denom)}; p={float(nonlinearity_test.pvalue):.3f}")

# %% [markdown]
# ## 6. Standardized spline predictions and figure
#
# Predicted prevalences are standardized to the birth-weighted distribution of
# years. The curve is displayed from the birth-weighted 1st to 99th percentile.

# %%
plot_limits = weighted_percentiles(
    cnv["altitude_1000m"], [0.01, 0.99], cnv["total_nacimientos"]
)
altitude_grid = np.linspace(plot_limits[0], plot_limits[1], 300)
years = list(cnv["year_factor"].cat.categories)
year_weights = (
    cnv.groupby("year_factor", observed=False)["total_nacimientos"]
    .sum().reindex(years).to_numpy(dtype=float)
)
year_weights /= year_weights.sum()
design_info = spline_model.model.data.design_info
beta = spline_model.params.to_numpy()
cov_beta = spline_model.cov_params().to_numpy()
predictions = []

for altitude_value in altitude_grid:
    basis = restricted_cubic_spline_basis(np.array([altitude_value]), knots)
    new_data = pd.DataFrame({
        "altitude_1000m": [altitude_value] * len(years),
        "spline_nl1": [basis[0, 0]] * len(years),
        "spline_nl2": [basis[0, 1]] * len(years),
        "year_factor": pd.Categorical(years, categories=years, ordered=True)
    })
    X = np.asarray(build_design_matrices([design_info], new_data)[0])
    predicted_by_year = np.exp(X @ beta)
    prevalence = np.sum(year_weights * predicted_by_year)
    gradient = np.sum(year_weights[:, None] * predicted_by_year[:, None] * X, axis=0)
    variance = gradient @ cov_beta @ gradient
    se = np.sqrt(max(variance, 0))
    se_log = se / prevalence
    predictions.append({
        "altitude_m": altitude_value * 1000,
        "predicted_prevalence_pct": prevalence * 100,
        "CI95_lower_pct": prevalence * np.exp(-1.96 * se_log) * 100,
        "CI95_upper_pct": prevalence * np.exp(1.96 * se_log) * 100
    })

predictions = pd.DataFrame(predictions)
cnv["altitude_bin_250m"] = np.floor(cnv["altitud"] / 250) * 250
observed_binned = (
    cnv.groupby("altitude_bin_250m", as_index=False)
    .agg(lbw_cases=("casos_bpn", "sum"), births=("total_nacimientos", "sum"))
)
observed_binned["observed_prevalence_pct"] = (
    100 * observed_binned["lbw_cases"] / observed_binned["births"]
)
observed_binned = observed_binned[
    observed_binned["altitude_bin_250m"].between(
        predictions["altitude_m"].min(), predictions["altitude_m"].max()
    )
]

fig, ax = plt.subplots(figsize=(8.5, 5.5))
ax.fill_between(
    predictions["altitude_m"], predictions["CI95_lower_pct"],
    predictions["CI95_upper_pct"], color="#9BB8D8", alpha=0.35,
    linewidth=0, label="95% CI"
)
ax.plot(
    predictions["altitude_m"], predictions["predicted_prevalence_pct"],
    color="#173F6D", linewidth=2.6, label="Year-adjusted spline"
)
ax.scatter(
    observed_binned["altitude_bin_250m"], observed_binned["observed_prevalence_pct"],
    facecolor="white", edgecolor="#B33A3A", linewidth=0.9, s=35,
    zorder=3, label="Observed prevalence (250-m bins)"
)
ax.set_xlabel("Altitude of the provincial capital (m above sea level)")
ax.set_ylabel("Low birth weight prevalence (%)")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "cnv_altitude_lbw_spline.png", dpi=600, bbox_inches="tight")
plt.show()

predictions.to_csv(OUTPUT_DIR / "cnv_spline_predictions.csv", index=False)

# %% [markdown]
# ## 7. Robustness analyses: alternative correlation structures
#
# These models assess sensitivity to the working correlation and variance
# specification. They are supplementary to the primary exchangeable GEE.

# %%
gee_independence = smf.gee(
    "casos_bpn ~ altitude_1000m + C(year_factor)",
    groups="id_provincia", data=cnv, family=sm.families.Poisson(),
    cov_struct=Independence(), offset=cnv["log_births"]
).fit()

cnv_ar = cnv.sort_values(["id_provincia", "año"]).copy()
gee_ar1 = smf.gee(
    "casos_bpn ~ altitude_1000m + C(year_factor)",
    groups="id_provincia", time="año", data=cnv_ar,
    family=sm.families.Poisson(), cov_struct=Autoregressive(grid=True),
    offset=cnv_ar["log_births"]
).fit()

glm_cluster = smf.glm(
    "casos_bpn ~ altitude_1000m + C(year_factor)",
    data=cnv, family=sm.families.Poisson(), offset=cnv["log_births"]
).fit(cov_type="cluster", cov_kwds={"groups": cnv["id_provincia"]})

robustness = pd.DataFrame([
    extract_altitude_effect(gee_exchangeable, "Exchangeable Poisson GEE", rho_exchangeable),
    extract_altitude_effect(gee_ar1, "AR(1) Poisson GEE", gee_ar1.cov_struct.dep_params),
    extract_altitude_effect(gee_independence, "Independence Poisson GEE"),
    extract_altitude_effect(glm_cluster, "Cluster-robust Poisson GLM")
])
print("Robustness models")
print(robustness.round(3).to_string(index=False))
robustness.to_csv(OUTPUT_DIR / "cnv_robustness_models.csv", index=False)

# %% [markdown]
# ## 8. Save key model summaries
#
# The printed estimates above and exported tables correspond to the models
# reported in the manuscript and supplementary analyses.

# %%
print("\nAnalysis completed. Results saved under:", OUTPUT_DIR)
