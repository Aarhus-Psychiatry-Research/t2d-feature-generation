import pandas as pd
import pytest
from steps.concatenators import validate_indeces_match_and_concat


def test_validate_indeces_match_and_concat():
    # create sample dataframes
    df1 = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    df2 = pd.DataFrame({"A": [7, 8, 9], "B": [10, 11, 12]})

    # call the function with a list of dataframes
    result = validate_indeces_match_and_concat([df1, df2])

    # check that the output is a concatenated dataframe
    assert isinstance(result, pd.DataFrame)

    # check that the output dataframe has the expected columns
    expected_columns = ["A", "B", "A", "B"]
    assert result.columns.tolist() == expected_columns

    # check that the output dataframe has the expected values
    expected_values = [[1, 4, 7, 10], [2, 5, 8, 11], [3, 6, 9, 12]]
    assert result.values.tolist() == expected_values

    # check that the function raises an error if the indices are not equal
    df3 = pd.DataFrame({"A": [13, 14, 15], "B": [16, 17, 18]}, index=[1, 2, 3])

    with pytest.raises(ValueError):
        validate_indeces_match_and_concat([df1, df3])
