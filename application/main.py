"""Main feature generation."""
import logging
from typing import Callable

import numpy as np
from modules.specify_features import FeatureSpecifier
from psycop_feature_generation.application_modules.describe_flattened_dataset import (
    save_flattened_dataset_description_to_disk,
)
from psycop_feature_generation.application_modules.flatten_dataset import (
    create_flattened_dataset,
)
from psycop_feature_generation.application_modules.loggers import init_root_logger
from psycop_feature_generation.application_modules.project_setup import (
    get_project_info,
    init_wandb,
)
from psycop_feature_generation.application_modules.save_dataset_to_disk import (
    split_and_save_dataset_to_disk,
)
from psycop_feature_generation.application_modules.wandb_utils import (
    wandb_alert_on_exception,
)
from psycop_feature_generation.loaders.raw.load_moves import (
    load_move_into_rm_for_exclusion,
)
from psycop_feature_generation.loaders.raw.load_visits import (
    physical_visits_to_psychiatry,
)
from steps.somatic_loaders import (
    LoaderParams,
    load_and_flatten_somatic_medications,
    prediction_times_loader,
    quarantine_df_loader,
)
from zenml.pipelines import pipeline

log = logging.getLogger()
import pandas as pd


@wandb_alert_on_exception
def main():
    """Main function for loading, generating and evaluating a flattened
    dataset."""

    feature_specs = FeatureSpecifier(
        project_info=project_info,
        min_set_for_debug=False,  # Remember to set to False when generating full dataset
    ).get_feature_specs()

    flattened_df = create_flattened_dataset(
        feature_specs=feature_specs,
        prediction_times_df=physical_visits_to_psychiatry(timestamps_only=True),
        drop_pred_times_with_insufficient_look_distance=False,
        project_info=project_info,
        quarantine_df=load_move_into_rm_for_exclusion(),
        quarantine_days=720,
    )

    split_and_save_dataset_to_disk(
        flattened_df=flattened_df,
        project_info=project_info,
    )

    save_flattened_dataset_description_to_disk(
        project_info=project_info,
        feature_specs=feature_specs,
    )

@pipeline(enable_cache=True)
def main_pipeline(
    prediction_times_loader: Callable,
    quarantine_df_loader: Callable,
    load_and_flatten_somatic_medications: Callable,
    # load_and_flatten_somatic_diagnoses: Callable,
    # load_and_flatten_somatic_lab_results: Callable,
    # load_and_flatten_psychiatric_medications: Callable,
    # load_and_flatten_psychiatric_diagnoses: Callable,
    # load_and_flatten_psychiatric_lab_results: Callable,
    # split_and_save_dataset_to_disk: Callable,
    # save_flattened_dataset_description_to_disk: Callable,
):
    prediction_times = prediction_times_loader()
    quarantine_df = quarantine_df_loader()

    features = [
        load_and_flatten_somatic_medications(prediction_times=prediction_times, quarantine_df=quarantine_df),
        # load_and_flatten_somatic_diagnoses(prediction_times=prediction_times, quarantine_df=quarantine_df),
        # load_and_flatten_somatic_lab_results(prediction_times=prediction_times, quarantine_df=quarantine_df),
        # load_and_flatten_psychiatric_medications(prediction_times=prediction_times, quarantine_df=quarantine_df),
        # load_and_flatten_psychiatric_diagnoses(prediction_times=prediction_times, quarantine_df=quarantine_df),
        # load_and_flatten_psychiatric_lab_results(prediction_times=prediction_times, quarantine_df=quarantine_df),
        # load_and_flatten_metadata(prediction_times=prediction_times, quarantine_df=quarantine_df),
    ]

    # Concatenate all the datasets
    # combined_predictors = pd.concat(
    #     features,
    #     axis=1,
    # )

    # split_and_save_dataset_to_disk()
    
    # save_flattened_dataset_description_to_disk()

    pass

if __name__ == "__main__":
    # Run elements that are required before wandb init first,
    # then run the rest in main so you can wrap it all in
    # wandb_alert_on_exception, which will send a slack alert
    # if you have wandb alerts set up in wandb
    project_info = get_project_info(
        project_name="t2d",
    )

    init_root_logger(project_info=project_info)

    log.info(f"Stdout level is {logging.getLevelName(log.level)}")
    log.debug("Debugging is still captured in the log file")

    # Use wandb to keep track of your dataset generations
    # Makes it easier to find paths on wandb, as well as
    # allows monitoring and automatic slack alert on failure
    # init_wandb(
    #     project_info=project_info,
    # )
    
    resolve_multiple = ["max", "min", "mean", "latest"]
    interval_days = [30, 90, 180, 365, 730]
    allowed_nan_value_prop = [0]

    loader_params = LoaderParams(
            values_loader=["gerd_drugs", "statins", "antihypertensives", "diuretics"], 
            project_info=project_info,
            interval_days=interval_days,
            resolve_multiple=resolve_multiple,
            fallback=["NaN"],
            allowed_nan_value_prop=[0],
            quarantine_days=730,
        )

    main_pipeline_instance = main_pipeline(
        prediction_times_loader=prediction_times_loader(),
        quarantine_df_loader=quarantine_df_loader(),
        load_and_flatten_somatic_medications=load_and_flatten_somatic_medications(params=loader_params),
    )
    
    main_pipeline_instance.run(unlisted=True)
        