"""Main feature generation."""
import logging

import numpy as np
from psycop_feature_generation.application_modules.flatten_dataset import (
    create_flattened_dataset,
)
from psycop_feature_generation.application_modules.project_setup import get_project_info
from psycop_feature_generation.loaders.raw.load_moves import (
    load_move_into_rm_for_exclusion,
)
from psycop_feature_generation.loaders.raw.load_visits import (
    physical_visits_to_psychiatry,
)

from timeseriesflattener.feature_spec_objects import PredictorSpec

log = logging.getLogger()


def main():
    """Main function for loading, generating and evaluating a flattened
    dataset."""
    project_info = get_project_info(
        project_name="t2d-testing",
    )

    feature_specs = [
        PredictorSpec(
            values_loader="hba1c",
            fallback=np.nan,
            lookbehind_days=9999,
            resolve_multiple_fn="count",
            allowed_nan_value_prop=0.0,
            prefix="eval",
        ),
    ]

    flattened_df = create_flattened_dataset(
        feature_specs=feature_specs,
        prediction_times_df=physical_visits_to_psychiatry(
            timestamps_only=True,
            timestamp_for_output="end",
        ),
        drop_pred_times_with_insufficient_look_distance=False,
        project_info=project_info,
        quarantine_df=load_move_into_rm_for_exclusion(),
        quarantine_days=720,
    )

    prop_na = (
        flattened_df["eval_hba1c_within_9999_days_count_fallback_nan"].isna().mean()
    )
    prop_na_for_display = round(prop_na, 2)

    print(prop_na_for_display)

    pass


if __name__ == "__main__":
    main()
