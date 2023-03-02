import pandas as pd
from timeseriesflattener.feature_spec_objects import StaticSpec
from zenml.steps import BaseParameters, step

from t2d_feature_generation.steps.flatten_from_specs import (
    FlattenFromParamsConf,
    flatten_from_specs,
    flatten_with_age,
)


class StaticLoaderParams(BaseParameters):
    flattening_conf: FlattenFromParamsConf = FlattenFromParamsConf()


@step
def load_and_flatten_static_specs(
    params: StaticLoaderParams,
    prediction_times: pd.DataFrame,
) -> pd.DataFrame:
    specs = [
        StaticSpec(
            values_loader="sex_female",
            input_col_name_override="sex_female",
            prefix="pred",
        ),
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

    flattened_df = flatten_with_age(
        specs=specs,
        prediction_times=prediction_times,
        flattening_conf=params.flattening_conf,
        add_age=True,
    )
    return flattened_df
