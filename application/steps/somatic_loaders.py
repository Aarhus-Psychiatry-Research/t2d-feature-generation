from typing import Any, Callable, Iterable, List, Tuple

import pandas as pd
from psycop_feature_generation.application_modules.flatten_dataset import (
    create_flattened_dataset,
)
from psycop_feature_generation.application_modules.project_setup import ProjectInfo
from psycop_feature_generation.loaders.raw.load_moves import (
    load_move_into_rm_for_exclusion,
)
from psycop_feature_generation.loaders.raw.load_visits import (
    physical_visits_to_psychiatry,
)
from zenml.steps import BaseParameters, step

from timeseriesflattener.feature_spec_objects import PredictorGroupSpec


class LoaderParams(BaseParameters):
    values_loader: List[str]
    project_info: ProjectInfo
    interval_days: List[int]
    resolve_multiple: Iterable[str]
    fallback: List[Any]
    allowed_nan_value_prop: List[float]
    quarantine_days: int


@step
def prediction_times_loader() -> pd.DataFrame:
    df = physical_visits_to_psychiatry(timestamps_only=True)
    return df


@step
def quarantine_df_loader() -> pd.DataFrame:
    df = load_move_into_rm_for_exclusion()
    return df


@step
def load_and_flatten_somatic_medications(
    params: LoaderParams,
    prediction_times: pd.DataFrame,
    quarantine_df: pd.DataFrame,
) -> pd.DataFrame:
    flattened_df = flatten_from_specs(params, prediction_times, quarantine_df)
    return flattened_df


def flatten_from_specs(
    params: LoaderParams, prediction_times: pd.DataFrame, quarantine_df: pd.DataFrame
):
    specs = PredictorGroupSpec(
        values_loader=params.values_loader,
        lookbehind_days=params.interval_days,
        resolve_multiple_fn=params.resolve_multiple,
        fallback=params.fallback,
        allowed_nan_value_prop=params.allowed_nan_value_prop,
    ).create_combinations()

    flattened_df = create_flattened_dataset(
        feature_specs=specs,
        prediction_times_df=prediction_times,
        drop_pred_times_with_insufficient_look_distance=False,
        project_info=params.project_info,
        quarantine_df=quarantine_df,
        quarantine_days=params.quarantine_days,
    )

    return flattened_df
