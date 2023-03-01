from typing import List

import pandas as pd
from pipelines.dynamic_pipelines.gather_step import GatherStepsParameters
from steps.loaders.predictor_loader import PredictorOutputParameters
from zenml.steps import step


class ConcatenatorParams(GatherStepsParameters):
    """Parameters for the concatenator step."""

    pass


@step
def feature_concatenator(params: ConcatenatorParams) -> pd.DataFrame:
    # Check that all indeces match and concat the dataframes
    inputs = PredictorOutputParameters().gather(params)

    dfs = [p.flattened_df for p in inputs]

    concatenated_df = validate_indeces_match_and_concat(dfs)

    return concatenated_df


def validate_indeces_match_and_concat(dfs):
    if not all(dfs[0].index.equals(f.index) for f in dfs):
        raise ValueError("All features must have the same indeces.")

    concatenated_df = pd.concat(dfs, axis=1)

    return concatenated_df


@step
def combined_concatenator(
    outcomes: pd.DataFrame, statics: pd.DataFrame, predictors: pd.DataFrame
) -> pd.DataFrame:
    features = [outcomes, statics, predictors]

    concatenated_df = validate_indeces_match_and_concat(features)

    return concatenated_df
