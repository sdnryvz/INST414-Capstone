# INST414 Capstone

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Predicting Spotify Song Popularity using audio features. 

This project predicts Spotify song popularity using machine learning models built from audio features such as danceability, energy, loudness, valence, and tempo, along with playlist metadata like genre and subgenre. The dataset was split using an 80/20 train–test split, with no stratification since popularity is a continuous target. I also used 5-fold cross-validation to evaluate model generalization and ensure that performance was not dependent on a single train/test split.

I tested three models: Linear Regression as a simple benchmark, Random Forest as a nonlinear ensemble method, and CatBoost as a gradient boosting model optimized for categorical data. A naive baseline model that predicts the mean popularity for all songs served as a minimum performance benchmark; it achieved an MAE of 20.77, RMSE of 24.82, and an R² near zero. This confirmed that the baseline captured no meaningful variation in popularity, so any effective model needed to significantly outperform it.

Among all models tested, the Random Forest performed best, achieving an MAE of 16.97, RMSE of 21.16, and an R² of 0.273. It trained quickly (3–5 seconds) and offered reasonable interpretability through feature importance analysis, which showed that danceability, energy, and valence were among the strongest predictors of popularity. I also completed required regression diagnostics, including residual analysis, Q-Q plots, error distributions, cross-validation results, and worst-prediction analysis. Feature engineering focused primarily on encoding categorical variables, and attempts at feature elimination actually reduced performance.

Looking ahead to Sprint 4, I plan to finalize my model, refine visualizations, write a clear non-technical explanation of results, and potentially explore a simple ensemble of Random Forest and CatBoost. I will also prepare a presentation-ready analysis and begin drafting my final paper. I am currently on track, with my biggest win being the strong performance of the Random Forest model and my biggest challenge being the inherent noise in Spotify popularity data. I feel moderately confident in the model (7/10) and may need instructor feedback on interpretation and communicating results effectively.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         inst414_capstone and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── inst414_capstone   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes inst414_capstone a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

