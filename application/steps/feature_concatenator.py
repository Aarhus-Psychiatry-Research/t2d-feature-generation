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

    features = [p.flattened_df for p in inputs]

    if not all(features[0].index.equals(f.index) for f in features):
        raise ValueError("All features must have the same indeces.")

    concatenated_df = pd.concat(features, axis=1)

    return concatenated_df
