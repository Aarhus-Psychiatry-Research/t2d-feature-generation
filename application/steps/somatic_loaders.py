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
    id_col_name: str


@step
def prediction_times_loader(
    params: PredTimeParams, quarantine_df: pd.DataFrame
) -> pd.DataFrame:
    """Loader for prediction times."""
    df = physical_visits_to_psychiatry(timestamps_only=True)

    df = filter_prediction_times(
        prediction_times_df=df,
        project_info=params.project_info,
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
    id_col_name: str = "dw_ek_borger"


class PredictorLoaderParams(BaseParameters):
    values_loader: List[str]
    lookbehind_days: List[Union[int, float]]
    resolve_multiple_fn: List[str]
    fallback: List[Any]
    allowed_nan_value_prop: List[float]
    flatten_from_params_config: FlattenFromParamsConf


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
        prefix=params.prefix,
    ).create_combinations()

    flattened_df = flatten_from_specs(
        specs=specs,
        prediction_times=prediction_times,
        predictor_prefix=params.predictor_prefix,
        outcome_prefix=params.outcome_prefix,
        timestamp_col_name=params.timestamp_col_name,
        entity_id_col_name=params.entity_id_col_name,
    )
    return flattened_df


class StaticLoaderParams(BaseParameters):
    project_info: ProjectInfo


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
        specs=specs, prediction_times=prediction_times, project_info=params.project_info
    )
    return flattened_df


class OutcomeLoaderParams(BaseParameters):
    project_info: ProjectInfo
    values_loader: List[str]
    lookahead_days: List[Union[int, float]]
    resolve_multiple_fn: List[str]
    fallback: List[Any]
    incident: List[bool]
    allowed_nan_value_prop: List[float]


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
        prefix=params.project_info.prefix.outcome,
    ).create_combinations()

    flattened_df = flatten_from_specs(
        specs=specs, prediction_times=prediction_times, project_info=params.project_info
    )
    return flattened_df


def flatten_from_specs(
    specs: List[Union[StaticSpec, TemporalSpec]],
    prediction_times: pd.DataFrame,
    predictor_prefix: str,
    outcome_prefix: str,
    timestamp_col_name: str,
    entity_id_col_name: str,
):
    flattened_dataset = TimeseriesFlattener(
        prediction_times_df=prediction_times,
        n_workers=min(
            len(specs),
            psutil.cpu_count(logical=True),
        ),
        drop_pred_times_with_insufficient_look_distance=False,
        predictor_col_name_prefix=predictor_prefix,
        outcome_col_name_prefix=outcome_prefix,
        timestamp_col_name=timestamp_col_name,
        entity_id_col_name=entity_id_col_name,
    )

    flattened_dataset.add_spec(spec=specs)

    return flattened_dataset.get_df()
