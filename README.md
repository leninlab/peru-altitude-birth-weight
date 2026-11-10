# Residential altitude and birth weight in Peru

This repository contains the analysis code and processed ENDES datasets used to examine the relationship between maternal residential altitude and birth weight in Peru, 2020–2024.

## Repository contents

- `analysis/01_endes_primary_analysis.R`: R script for the primary ENDES analysis.
- `data/processed/ENDES_2020_2024_birth_weight_card.csv`: primary analytic dataset using birth weights recorded on the health card.
- `data/processed/ENDES_2020_2024_birth_weight_card_recall.csv`: expanded dataset including health-card and maternally recalled birth weights.

## Reproducing the primary ENDES analysis

The analysis was conducted using R 4.4.2 and the `survey` package (version 4.5).

Open the repository folder as the working directory in R or RStudio, then run:

```r
source("analysis/01_endes_primary_analysis.R")
