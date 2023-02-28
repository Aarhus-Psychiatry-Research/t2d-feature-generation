from typing import Any, List, Union

import pandas as pd
from steps.flatten_from_specs import FlattenFromParamsConf, flatten_from_specs
from zenml.steps import BaseParameters, step

from timeseriesflattener.feature_spec_objects import PredictorGroupSpec


class PredictorLoaderParams(BaseParameters):
    values_loader: List[str]
    lookbehind_days: List[Union[int, float]]
    resolve_multiple_fn: List[str]
    fallback: List[Any]
    allowed_nan_value_prop: List[float]
    flattening_conf: FlattenFromParamsConf = FlattenFromParamsConf()


@step
def load_and_flatten_predictors(
    params: PredictorLoaderParams,
    prediction_times: pd.DataFrame,
) -> pd.DataFrame:
    specs = PredictorGroupSpec(
        values_loader=params.values_loader,
        lookbehind_days=params.lookbehind_days,
        resolve_multiple_fn=params.resolve_multiple_fn,
        fallback=params.fallback,
        allowed_nan_value_prop=params.allowed_nan_value_prop,
        prefix=params.flattening_conf.predictor_prefix,
    ).create_combinations()

    flattened_df = flatten_from_specs(
        specs=specs,
        prediction_times=prediction_times,
        flattening_conf=params.flattening_conf,
    )
    return flattened_df
