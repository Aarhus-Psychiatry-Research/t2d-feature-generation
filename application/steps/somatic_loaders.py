from sys import prefix
from typing import Any, Callable, Iterable, List, Literal, Optional, Tuple, Union

import pandas as pd
import psutil
from psycop_feature_generation.application_modules.flatten_dataset import (
    filter_prediction_times,
)
from psycop_feature_generation.application_modules.project_setup import ProjectInfo
from psycop_feature_generation.loaders.raw.load_moves import (
    load_move_into_rm_for_exclusion,
)
from psycop_feature_generation.loaders.raw.load_visits import (
    physical_visits_to_psychiatry,
)
from pydantic import BaseModel
from zenml.steps import BaseParameters, step

from timeseriesflattener.feature_cache.cache_to_disk import DiskCache
from timeseriesflattener.feature_spec_objects import (
    OutcomeGroupSpec,
    PredictorGroupSpec,
    StaticSpec,
    TemporalSpec,
)
from timeseriesflattener.flattened_dataset import TimeseriesFlattener


class PredTimeParams(BaseParameters):
    quarantine_days: int
    entity_id_col_name: str


@step
def prediction_times_loader(
    params: PredTimeParams, quarantine_df: pd.DataFrame
) -> pd.DataFrame:
    """Loader for prediction times."""
    df = physical_visits_to_psychiatry(timestamps_only=True)

    df = filter_prediction_times(
        prediction_times_df=df,
        quarantine_df=quarantine_df,
        quarantine_days=params.quarantine_days,
    )

    return df


@step
def quarantine_df_loader() -> pd.DataFrame:
    df = load_move_into_rm_for_exclusion()
    return df


class FlattenFromParamsConf(BaseParameters):
    predictor_prefix: str = "pred"
    outcome_prefix: str = "outc"
    timestamp_col_name: str = "timestamp"
    entity_id_col_name: str = "dw_ek_borger"


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
        prefix=params.predictor_prefix,
    ).create_combinations()

    flattened_df = flatten_from_specs(
        specs=specs,
        prediction_times=prediction_times,
        flattening_conf=params.flattening_conf,
    )
    return flattened_df


class StaticLoaderParams(BaseParameters):
    flattening_conf: FlattenFromParamsConf = FlattenFromParamsConf()


@step
def load_and_flatten_static_specs(
    params: StaticLoaderParams,
    prediction_times: pd.DataFrame,
) -> pd.DataFrame:
    specs = [
        StaticSpec(
            values_loader="t2d",
            input_col_name_override="timestamp",
            output_col_name_override="timestamp_first_t2d_hba1c",
            prefix="",
        ),
        StaticSpec(
            values_loader="timestamp_exclusion",
            input_col_name_override="timestamp",
            output_col_name_override="timestamp_exclusion",
            prefix="",
        ),
    ]

    flattened_df = flatten_from_specs(
        specs=specs,
        prediction_times=prediction_times,
        flattening_conf=params.flattening_conf,
    )
    return flattened_df


class OutcomeLoaderParams(BaseParameters):
    values_loader: List[str]
    lookahead_days: List[Union[int, float]]
    resolve_multiple_fn: List[str]
    fallback: List[Any]
    incident: List[bool]
    allowed_nan_value_prop: List[float]
    flattening_conf: FlattenFromParamsConf = FlattenFromParamsConf()


@step
def load_and_flatten_outcomes(
    params: OutcomeLoaderParams,
    prediction_times: pd.DataFrame,
) -> pd.DataFrame:
    specs = OutcomeGroupSpec(
        values_loader=params.values_loader,
        lookahead_days=params.lookahead_days,
        resolve_multiple_fn=params.resolve_multiple_fn,
        fallback=params.fallback,
        incident=params.incident,
        allowed_nan_value_prop=params.allowed_nan_value_prop,
        prefix=params.flattening_conf.outcome_prefix,
    ).create_combinations()

    flattened_df = flatten_from_specs(
        specs=specs,
        prediction_times=prediction_times,
        flattening_conf=params.flattening_conf,
    )
    return flattened_df


def flatten_from_specs(
    specs: List[Union[StaticSpec, TemporalSpec]],
    prediction_times: pd.DataFrame,
    flattening_conf: FlattenFromParamsConf,
):
    flattened_dataset = TimeseriesFlattener(
        prediction_times_df=prediction_times,
        n_workers=min(
            len(specs),
            psutil.cpu_count(logical=True),
        ),
        drop_pred_times_with_insufficient_look_distance=False,
        predictor_col_name_prefix=flattening_conf.predictor_prefix,
        outcome_col_name_prefix=flattening_conf.outcome_prefix,
        timestamp_col_name=flattening_conf.timestamp_col_name,
        entity_id_col_name=flattening_conf.entity_id_col_name,
    )

    flattened_dataset.add_spec(spec=specs)

    return flattened_dataset.get_df()
