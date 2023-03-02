import pandas as pd
from steps.flatten_from_specs import FlattenFromParamsConf, flatten_from_specs
from timeseriesflattener.feature_spec_objects import StaticSpec
from zenml.steps import BaseParameters, step


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