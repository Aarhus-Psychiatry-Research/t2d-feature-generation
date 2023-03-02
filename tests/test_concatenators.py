"""Test cases for the main module."""
import pandas as pd
import pytest
from t2d_feature_generation.steps.concatenators import validate_indeces_match_and_concat


def test_validate_indeces_match_and_concat():
    # Create test dataframes with matching indices
    df1 = pd.DataFrame({"A": [1, 2], "B": [3, 4]}, index=[0, 1])
    df2 = pd.DataFrame({"A": [1, 2], "C": [7, 8]}, index=[0, 1])
    df3 = pd.DataFrame({"A": [1, 2], "D": [11, 12]}, index=[0, 1])

    # Call the function with the test dataframes
    result = validate_indeces_match_and_concat([df1, df2, df3], shared_cols=["A"])

    # Check that the concatenated dataframe has the expected shape and columns
    assert result.shape == (2, 4)

    # Create test dataframes with non-matching indices
    df4 = pd.DataFrame({"A": [1, 2], "B": [3, 4]}, index=[0, 1])
    df5 = pd.DataFrame({"A": [1, 2], "C": [7, 8]}, index=[1, 2])

    # Call the function with the test dataframes and check that it raises a ValueError
    with pytest.raises(ValueError):
        validate_indeces_match_and_concat([df4, df5])

    # Create test dataframes with different lengths
    df6 = pd.DataFrame({"A": [1, 2], "B": [0, 1]}, index=[0, 1])
    df7 = pd.DataFrame({"A": [1]}, index=[0])
    df8 = pd.DataFrame({"A": [1, 2], "E": [9, 10]}, index=[0, 1])

    # Call the function with the test dataframes and check that it raises a ValueError
    with pytest.raises(ValueError):
        validate_indeces_match_and_concat([df6, df7, df8])
