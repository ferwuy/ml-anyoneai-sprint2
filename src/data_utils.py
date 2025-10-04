import os
from typing import Tuple

import gdown
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config


def get_datasets() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Download from GDrive all the needed datasets for the project.

    Returns:
        app_train : pd.DataFrame
            Training dataset

        app_test : pd.DataFrame
            Test dataset

        columns_description : pd.DataFrame
            Extra dataframe with detailed description about dataset features
    """
    # Download HomeCredit_columns_description.csv
    if not os.path.exists(config.DATASET_DESCRIPTION):
        gdown.download(
            config.DATASET_DESCRIPTION_URL, config.DATASET_DESCRIPTION, quiet=False
        )

    # Download application_test_aai.csv
    if not os.path.exists(config.DATASET_TEST):
        gdown.download(config.DATASET_TEST_URL, config.DATASET_TEST, quiet=False)

    # Download application_train_aai.csv
    if not os.path.exists(config.DATASET_TRAIN):
        gdown.download(config.DATASET_TRAIN_URL, config.DATASET_TRAIN, quiet=False)

    app_train = pd.read_csv(config.DATASET_TRAIN)
    app_test = pd.read_csv(config.DATASET_TEST)
    columns_description = pd.read_csv(config.DATASET_DESCRIPTION_URL)

    return app_train, app_test, columns_description


def get_feature_target(
    app_train: pd.DataFrame, app_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Separates our train and test datasets columns between Features
    (the input to the model) and Targets (what the model has to predict with the
    given features).

    Arguments:
        app_train : pd.DataFrame
            Training datasets
        app_test : pd.DataFrame
            Test datasets

    Returns:
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
        X_test : pd.DataFrame
            Test features
        y_test : pd.Series
            Test target
    """
    # Features: all columns except TARGET | Target: the TARGET column
    X_train, y_train = app_train.drop(columns=["TARGET"]), app_train["TARGET"]
    X_test, y_test = app_test.drop(columns=["TARGET"]), app_test["TARGET"]

    return X_train, y_train, X_test, y_test


def get_train_val_sets(
    X_train: pd.DataFrame, y_train: pd.Series
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Split training dataset in two new sets used for train and validation.

    Arguments:
        X_train : pd.DataFrame
            Original training features
        y_train: pd.Series
            Original training labels/target

    Returns:
        X_train : pd.DataFrame
            Training features
        X_val : pd.DataFrame
            Validation features
        y_train : pd.Series
            Training target
        y_val : pd.Series
            Validation target
    """

    X_tr, X_val, y_tr, y_val = train_test_split(  # split into train/validation subsets
        X_train,  # feature matrix to be partitioned
        y_train,  # target vector to be partitioned
        test_size=0.2,  # allocate 20% of samples to validation
        random_state=42,  # fixed seed for reproducibility across runs
        shuffle=True,  # shuffle before splitting to avoid ordered bias
    )

    return X_tr, X_val, y_tr, y_val
