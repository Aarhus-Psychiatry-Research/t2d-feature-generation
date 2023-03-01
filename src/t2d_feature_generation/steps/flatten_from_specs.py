from typing import List, Union

import pandas as pd
import psutil
from timeseriesflattener.feature_spec_objects import StaticSpec, TemporalSpec
from timeseriesflattener.flattened_dataset import TimeseriesFlattener
from zenml.steps import BaseParameters

from t2d_feature_generation.steps.loaders.t2d_loaders import (  # noqa pylint: disable=unused-import
    timestamp_exclusion,
)


class FlattenFromParamsConf(BaseParameters):
    predictor_prefix: str = "pred"
    outcome_prefix: str = "outc"
    timestamp_col_name: str = "timestamp"
    entity_id_col_name: str = "dw_ek_borger"


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
