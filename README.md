# Pakistan Wheat Yield Analysis and Forecasting

Project for the **Junior Data Science Analyst: Agriculture & Agribusiness** internship (09 Oct 2026 to 13 Nov 2026).

The project studies wheat area, production and yield in Pakistan (with a focus on Punjab) and the factors that influence them. Each weekly task is one stage of the same workflow.

## Data sources
| Source | Used for | Link |
|---|---|---|
| FAOSTAT, Crops and livestock products | National yield, area, production (1961 to 2024) | https://data.fao.org/catalog/iso/d24a448b-3b62-4c09-8c1d-4a39bb599876 |
| Pakistan Bureau of Statistics / State Bank of Pakistan | Provincial yield, irrigation, fertiliser, prices | https://www.pbs.gov.pk/node/249 |
| NASA POWER (Agroclimatology) | Daily weather variables (from 1981) | https://power.larc.nasa.gov/docs/tutorials/data-access-viewer/quick-start/ |

## Repository structure
| Folder | Contents | Status |
|---|---|---|
| `week1_strategy/` | Dataset research, preliminary FAOSTAT analysis, methods and strategy (report), script explore_week1.py | Done |
| `week2_eda/` | EDA report, script (eda_wheat.py) and 6 figures | Done |
| `week3_modeling_plan/` | Modelling plan report, 5 figures, baseline forecast script | Done |
| `week4_data_cleaning/` | Cleaning protocol report, scripts, stress test, 3 figures | Done |
| `week5_evaluation/` | Final evaluation report, metrics script, 4 figures | Done |
| `data/raw/` | Original downloaded files (not edited) | |
| `data/processed/` | Cleaned datasets | |

## Tools
Python, pandas, matplotlib, seaborn, scikit-learn, statsmodels, Jupyter.

## How to reproduce
1. `pip install -r requirements.txt`
2. Download the datasets listed above into `data/raw/`.
3. Run the notebooks in each week folder in order.

## Running the Week 2 analysis
```
cd week2_eda
python eda_wheat.py ../data/raw/FAOSTAT_data_en_10-9-2026.csv ./figures
```
The script checks data integrity, builds the analysis table and saves the six figures.

## Running the Week 3 baseline test
```
cd week3_modeling_plan
python baseline_models.py ../data/raw/FAOSTAT_data_en_10-9-2026.csv
python diagrams.py
```
`baseline_models.py` runs expanding-window one-step-ahead forecasts (1991 to 2024) for four baseline methods. Run it inside the `week3_modeling_plan` folder; it writes `baseline_forecasts.csv` and `baseline_metrics.csv` there. `diagrams.py` redraws the flowcharts and the baseline chart from those files.

## Running the Week 4 cleaning protocol
```
cd week4_data_cleaning
python stress_test.py ../data/raw/FAOSTAT_data_en_10-9-2026.csv
python make_figs.py
```
`cleaning_pipeline.py` holds the reusable cleaning functions. `stress_test.py` injects 14 known errors into a copy of the real data (simulated, for testing only), cleans it, and scores the result. The real FAOSTAT data needed no repairs.

## Running the Week 5 evaluation
```
cd week5_evaluation
python evaluation_metrics.py ../data/raw/FAOSTAT_data_en_10-9-2026.csv ../week3_modeling_plan/baseline_forecasts.csv
python make_figs5.py ../data/raw/FAOSTAT_data_en_10-9-2026.csv ../week3_modeling_plan/baseline_forecasts.csv figures
```
Run Week 3 first so that `baseline_forecasts.csv` exists.

## Summary of results
- Wheat production in Pakistan grew 8.3 times from 1961 to 2024, about two thirds through yield and one third through area.
- A simple linear-trend forecast (one year ahead, 1991 to 2024) has a typical error of about 3.4 percent (RMSE about 105 kg/ha), better than the naive benchmark.
- Weather, soil, input and provincial data are planned next steps and have not been analysed yet.

## Running the Week 1 preliminary analysis
```
cd week1_strategy
python explore_week1.py ../data/raw/FAOSTAT_data_en_10-9-2026.csv figures
```
Prints the growth and decomposition table (also saved as `growth_decomposition.csv`) and redraws the three Week 1 figures.
