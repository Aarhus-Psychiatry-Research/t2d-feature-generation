import logging
from collections.abc import Hashable, Sequence

import pandas as pd
from zenml.steps import step

from t2d_feature_generation.pipelines.dynamic_pipelines.gather_step import (
    GatherStepsParameters,
)
from t2d_feature_generation.steps.loaders.predictor_loader import (
    PredictorOutputParameters,
)

log = logging.getLogger(__name__)


class ConcatenatorParams(GatherStepsParameters):
    """Parameters for the concatenator step."""


def validate_indeces_match_and_concat(
    dfs: list[pd.DataFrame],
    shared_cols: Sequence[Hashable] = (
        "dw_ek_borger",
        "timestamp",
        "prediction_time_uuid",
    ),
):
    log.info(f"Validating indices for {len(dfs)} dataframes.")
    if not all(dfs[0].index.equals(f.index) for f in dfs):
        raise ValueError("All features must have the same indices.")

    log.info("Concatenating dataframes.")
    dfs_without_shared_cols = [df.drop(columns=list(shared_cols)) for df in dfs]

    dfs_to_concat = dfs_without_shared_cols[1:] + [dfs[0]]

    concatenated_df = pd.concat(dfs_to_concat, axis=1)

    return concatenated_df


@step
def feature_concatenator(params: ConcatenatorParams) -> pd.DataFrame:
    # Check that all indices match and concat the dataframes
    inputs = PredictorOutputParameters.gather(params)

    dfs = [p.flattened_df for p in inputs]

    concatenated_df = validate_indeces_match_and_concat(dfs)

    return concatenated_df


@step
def combined_concatenator(
    outcomes: pd.DataFrame,
    statistics: pd.DataFrame,
    predictors: pd.DataFrame,
) -> pd.DataFrame:
    features = [outcomes, statistics, predictors]

    concatenated_df = validate_indeces_match_and_concat(features)

    return concatenated_df
