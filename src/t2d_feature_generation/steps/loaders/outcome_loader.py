from typing import Any, Union

import pandas as pd
from zenml.steps import BaseParameters, Output, step

from t2d_feature_generation.steps.flatten_from_specs import (
    FlattenFromParamsConf,
    flatten_from_specs,
)
from timeseriesflattener.feature_spec_objects import (
    OutcomeGroupSpec,  # pylint: disable=no-name-in-module
)


class OutcomeLoaderParams(BaseParameters):
    values_loader: list[str]
    lookahead_days: list[Union[int, float]]
    resolve_multiple_fn: list[str]
    fallback: list[Any]
    incident: list[bool]
    allowed_nan_value_prop: list[float]
    flattening_conf: FlattenFromParamsConf = FlattenFromParamsConf()


@step
def load_and_flatten_outcomes(
    params: OutcomeLoaderParams,
    prediction_times: pd.DataFrame,
) -> Output(flattened_df=pd.DataFrame, filtered_prediction_times=pd.DataFrame):
    specs = OutcomeGroupSpec(
        values_loader=params.values_loader,
        lookahead_days=params.lookahead_days,
        resolve_multiple_fn=params.resolve_multiple_fn,
        fallback=params.fallback,
        incident=params.incident,
        allowed_nan_value_prop=params.allowed_nan_value_prop,
        prefix=params.flattening_conf.outcome_prefix,
    ).create_combinations()

    flattened_outcomes = flatten_from_specs(
        specs=specs,
        prediction_times=prediction_times,
        flattening_conf=params.flattening_conf,
    )

    filtered_prediction_times = flattened_outcomes[
        ["dw_ek_borger", "timestamp", "prediction_time_uuid"]
    ]

    return flattened_outcomes, filtered_prediction_times
