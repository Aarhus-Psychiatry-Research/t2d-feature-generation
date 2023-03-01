import pandas as pd
from psycop_feature_generation.application_modules.flatten_dataset import (
    filter_prediction_times,
)
from psycop_feature_generation.loaders.raw.load_visits import (
    physical_visits_to_psychiatry,
)
from zenml.steps import BaseParameters, step


class PredTimeParams(BaseParameters):
    quarantine_days: int
    entity_id_col_name: str


@step
def prediction_times_loader(
    params: PredTimeParams,
    quarantine_df: pd.DataFrame,
) -> pd.DataFrame:
    """Loader for prediction times."""
    df = physical_visits_to_psychiatry(timestamps_only=True)

    df = filter_prediction_times(
        prediction_times_df=df,
        quarantine_df=quarantine_df,
        quarantine_days=params.quarantine_days,
        entity_id_col_name=params.entity_id_col_name,
    )

    return df
