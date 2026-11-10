# Residential altitude and birth weight in Peru

This repository contains analysis code and processed datasets for a study of residential altitude and birth weight in Peru, using national data from 2020–2024.

## Repository contents

- `analysis/01_endes_primary_analysis.R`: R script for the primary individual-level ENDES analysis.
- `analysis/02_cnv_analysis.py`: Python script for the CNV descriptive and regression analyses.
- `analysis/02_cnv_analysis.ipynb`: Jupyter notebook version of the CNV analysis.
- `data/processed/ENDES_2020_2024_birth_weight_card.csv`: primary ENDES analytic dataset using birth weights recorded on the child’s health card.
- `data/processed/ENDES_2020_2024_birth_weight_card_recall.csv`: expanded ENDES dataset including health-card-recorded and maternally recalled birth weights.
- `data/processed/cnv_province_year_2020_2024.csv`: ecological CNV dataset with 954 province-year observations from 193 provinces.

## Reproducing the ENDES analysis

The ENDES analysis was conducted using R 4.4.2 and the `survey` package (version 4.5).

Open the repository folder as the working directory in R or RStudio, then run: ```r
source("analysis/01_endes_primary_analysis.R")

Reproducing the CNV analysis
The CNV analysis was conducted in Python. The script uses pandas, numpy, matplotlib, patsy, and statsmodels.

From the repository root, run:**
python analysis/02_cnv_analysis.py

Alternatively, open analysis/02_cnv_analysis.ipynb in Jupyter or Google Colab and run its cells. The analysis uses the province-year dataset in data/processed/.
Data sources

The ENDES datasets were derived from publicly available ENDES microdata for 2020–2024. The CNV dataset was derived from publicly available national live birth registry data for the same period. In the CNV analysis, provincial-capital altitude was used as an ecological proxy for altitude.
