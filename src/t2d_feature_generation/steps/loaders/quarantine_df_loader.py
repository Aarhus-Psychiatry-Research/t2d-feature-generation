import pandas as pd
from psycop_feature_generation.loaders.raw.load_moves import (
    load_move_into_rm_for_exclusion,
)
from zenml.steps import step


@step
def quarantine_df_loader() -> pd.DataFrame:
    df = load_move_into_rm_for_exclusion()
    return df
