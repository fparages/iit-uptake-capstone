Failure Modelling
==============================

Analysis and enhancement of supervised failure models for industrial time-series data with limited labels

Project Organization
------------

              
    ├── README.md          <- The top-level README for developers using this project.
    |
    ├── data      
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    |
    ├── trained_models     <- Trained and serialized models, model predictions, or model summaries.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data.
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling.
    │   │   └── transform_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions.
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── eda            <- Scripts to create exploratory and results oriented visualizations.
    │       └── eda.py
    │
    └── tests              <- Scipts to perform unit tests.



