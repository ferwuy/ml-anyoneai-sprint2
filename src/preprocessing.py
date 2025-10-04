from typing import Tuple  # function return annotation

import numpy as np  # numerical ops and NaN
import pandas as pd  # dataframe handling
from sklearn.impute import SimpleImputer  # median imputation
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder  # encoders + scaler


def preprocess_data(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pre processes data for modeling. Receives train, val and test dataframes
    and returns numpy ndarrays of cleaned up dataframes with feature engineering
    already performed.

    Arguments:
        train_df : pd.DataFrame
        val_df : pd.DataFrame
        test_df : pd.DataFrame

    Returns:
        train : np.ndarrary
        val : np.ndarrary
        test : np.ndarrary
    """
    # Print shape of input data
    print("Input train data shape: ", train_df.shape)  # log train rows/cols
    print("Input val data shape: ", val_df.shape)      # log val rows/cols
    print("Input test data shape: ", test_df.shape, "\n")  # log test rows/cols

    # Make a copy of the dataframes
    working_train_df = train_df.copy()  # avoid mutating caller data
    working_val_df = val_df.copy()      # idem for validation
    working_test_df = test_df.copy()    # idem for test

    # 1. Correct outliers/anomalous values in numerical
    # columns (`DAYS_EMPLOYED` column).
    working_train_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)  # replace sentinel with NaN (train)
    working_val_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)    # replace sentinel with NaN (val)
    working_test_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)   # replace sentinel with NaN (test)

    # 2. TODO Encode string categorical features (dytpe `object`):
    #     - If the feature has 2 categories encode using binary encoding,
    #       please use `sklearn.preprocessing.OrdinalEncoder()`. Only 4 columns
    #       from the dataset should have 2 categories.
    #     - If it has more than 2 categories, use one-hot encoding, please use
    #       `sklearn.preprocessing.OneHotEncoder()`. 12 columns
    #       from the dataset should have more than 2 categories.
    # Take into account that:
    #   - You must apply this to the 3 DataFrames (working_train_df, working_val_df,
    #     working_test_df).
    #   - In order to prevent overfitting and avoid Data Leakage you must use only
    #     working_train_df DataFrame to fit the OrdinalEncoder and
    #     OneHotEncoder classes, then use the fitted models to transform all the
    #     datasets.

    # Identify categorical columns to be encoded
    def get_categorical_columns(df, list1, list2):
        col = df.select_dtypes(include=["object"]).nunique() == 2  # True where exactly 2 uniques
        for name, value in col.items():                            # iterate categorical cols
            if value is True:                                      # binary categorical
                list1.append(name)                                 # collect for ordinal encoding
            else:                                                  # multi-category (or single unique)
                list2.append(name)                                 # collect for one-hot encoding
        return None                                                # lists filled by side-effect

    ordinal_col = []  # binary categorical columns (2 unique values)
    onehot_col = []   # multi-category columns (>2 unique values)

    get_categorical_columns(working_train_df, ordinal_col, onehot_col)  # detect on train only

    # Apply Ordinal Encoder to columns with 2 categories
    ordinal_encoder = OrdinalEncoder(                                 # map categories to numeric levels
        handle_unknown="use_encoded_value",                           # safe transform when unseen appears
        unknown_value=np.nan                                          # unseen becomes NaN (imputed later)
    )
    ordinal_encoder.fit(working_train_df[ordinal_col])                # fit on train only

    datasets = [working_train_df, working_val_df, working_test_df]    # convenient iterable over splits

    for df in datasets:                                               # transform all splits in-place
        df[ordinal_col] = ordinal_encoder.transform(df[ordinal_col])  # replace binary cats with numbers

    # Apply OneHot Encoder to columns with more than 2 categories
    onehot_encoder = OneHotEncoder(                                   # expand multi-cat to indicator columns
        handle_unknown="ignore",                                      # ignore unseen levels at transform
        sparse_output=False                                                  # dense output for easier concat
    )
    onehot_encoder.fit(working_train_df[onehot_col])                  # fit categories on train only

    # Helper function to apply OneHot Encoder
    def oe_encoding(df, col=onehot_col):
        working_onehot = onehot_encoder.transform(df[col])            # OHE array for given df
        df_onehot = pd.DataFrame(
            working_onehot,                                           # dense matrix
            columns=onehot_encoder.get_feature_names_out(col),        # stable OHE column names
            index=df.index,                                           # align with original index
        )
        df_onehot = pd.concat([df_onehot, df], axis=1).drop(col, axis=1)  # add OHE cols, drop originals
        return df_onehot                                              # return expanded dataframe

    df_test = oe_encoding(working_test_df)     # test with OHE applied
    df_train = oe_encoding(working_train_df)   # train with OHE applied
    df_val = oe_encoding(working_val_df)       # val with OHE applied

    # 3. TODO Impute values for all columns with missing data or, just all the columns.
    # Use median as imputing value. Please use sklearn.impute.SimpleImputer().
    # Again, take into account that:
    #   - You must apply this to the 3 DataFrames (working_train_df, working_val_df,
    #     working_test_df).
    #   - In order to prevent overfitting and avoid Data Leakage you must use only
    #     working_train_df DataFrame to fit the SimpleImputer and then use the fitted
    #     model to transform all the datasets.'
    # Create a copy of the DataFrames to avoid modifications to the originals

    # Use SimpleImputer to handle missing values and prevent data leakage
    imputer = SimpleImputer(strategy="median", 
                            missing_values=np.nan)  # per-column median
    imputer.fit(df_train)                                              # fit on train only

    # Helper function to apply imputation
    def impute_data(imputer, *dataframes):
        imputed_dataframes = [imputer.transform(df) for df in dataframes]  # transform to numpy arrays
        return imputed_dataframes                                          # keep order as given

    working_train_array, working_test_array, working_val_array = impute_data(
        imputer, 
        df_train, 
        df_test, 
        df_val
    )  # get arrays for train/test/val in that order

    # 4. TODO Feature scaling with Min-Max scaler. Apply this to all the columns.
    # Please use sklearn.preprocessing.MinMaxScaler().
    # Again, take into account that:
    #   - You must apply this to the 3 DataFrames (working_train_df, working_val_df,
    #     working_test_df).
    #   - In order to prevent overfitting and avoid Data Leakage you must use only
    #     working_train_df DataFrame to fit the MinMaxScaler and then use the fitted
    #     model to transform all the datasets.
    # Create the Min-Max scaler

    min_max = MinMaxScaler()                           # scales each feature to [0, 1]
    min_max.fit(working_train_array)                   # fit on train only

    # Helper function to apply scaling
    def scale_data(scaler, *dataarrays):
        scaled_data = [scaler.transform(data) for data in dataarrays]  # transform arrays
        return scaled_data                                             # keep order

    working_train_array, working_test_array, working_val_array = scale_data(
        min_max, 
        working_train_array, 
        working_test_array, 
        working_val_array
    )  # scaled arrays

    # Print shape of output data
    print("Output train data shape: ", working_train_array.shape)  # log train rows/cols
    print("Output val data shape: ", working_val_array.shape)      # log val rows/cols
    print("Output test data shape: ", working_test_array.shape, "\n")    # log test rows/cols

    return working_train_array, working_val_array, working_test_array  # return (train, val, test)