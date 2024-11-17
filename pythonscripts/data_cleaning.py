import pandas as pd
import numpy as np

def integer_impute_na_mean(column):
    """Replace NA values with mean"""
    return column.fillna(column.mean())

def categorical_impute_na_mode(column):
    """Replace NA values with mode"""
    return column.fillna(column.mode()[0])

def numeric_remove_outliers_zscore(column):
    """Remove outliers using z-score"""
    z_scores = np.abs((column - column.mean()) / column.std())
    return column[z_scores <= 3]

def numeric_cap_outliers_zscore(column):
    """Cap outliers using z-score"""
    mean = column.mean()
    std = column.std()
    return column.clip(lower=mean - 3*std, upper=mean + 3*std)

def remove_duplicates_keep_first(column):
    """Remove duplicates keeping first"""
    return column.drop_duplicates()