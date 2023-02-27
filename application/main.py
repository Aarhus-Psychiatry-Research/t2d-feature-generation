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
    PredTimeParams,
    load_and_flatten_predictors,
    prediction_times_loader,
    quarantine_df_loader,
)
from zenml.pipelines import pipeline

from timeseriesflattener import resolve_multiple_functions

log = logging.getLogger()

@pipeline(enable_cache=True)
def main_pipeline(
    prediction_times_loader: Callable,
    quarantine_df_loader: Callable,
    load_and_flatten_somatic_medications: Callable,
    load_and_flatten_somatic_diagnoses: Callable,
    load_and_flatten_somatic_lab_results: Callable,
    load_and_flatten_psychiatric_medications: Callable,
    load_and_flatten_psychiatric_diagnoses: Callable,
    load_and_flatten_psychiatric_lab_results: Callable,
    split_and_save_dataset_to_disk: Callable,
    save_flattened_dataset_description_to_disk: Callable,
):
    quarantine_df = quarantine_df_loader()
    prediction_times = prediction_times_loader(quarantine_df=quarantine_df)

    features = [
        load_and_flatten_somatic_medications(prediction_times=prediction_times),
        # load_and_flatten_somatic_diagnoses(prediction_times=prediction_times),
        # load_and_flatten_somatic_lab_results(prediction_times=prediction_times),
        # load_and_flatten_psychiatric_medications(prediction_times=prediction_times),
        # load_and_flatten_psychiatric_diagnoses(prediction_times=prediction_times),
        # load_and_flatten_psychiatric_lab_results(prediction_times=prediction_times),
        # load_and_flatten_metadata(prediction_times=prediction_times),
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
    
    lookbehind_days = [30, 90, 180, 365, 730]
    resolve_multiple = ["max", "min", "mean", "latest"]
    
    
    allowed_nan_value_prop = [0]
    
    loader_params = {"somatic": {
                            "medications": LoaderParams(
                                project_info=project_info,
                                values_loader=["gerd_drugs", "statins", "antihypertensives", "diuretics"],
                                lookbehind_days=lookbehind_days,
                                resolve_multiple_fn=resolve_multiple,
                                fallback=[np.nan],
                                allowed_nan_value_prop=allowed_nan_value_prop,
                            ), 
                            "diagnoses": LoaderParams(
                                project_info=project_info,
                                values_loader=[
                                    "essential_hypertension",
                                    "hyperlipidemia",
                                    "polycystic_ovarian_syndrome",
                                    "sleep_apnea",
                                    "gerd",
                                ],
                                lookbehind_days=lookbehind_days,
                                resolve_multiple_fn=resolve_multiple,
                                fallback=[0],
                                allowed_nan_value_prop=allowed_nan_value_prop,
                            ),
                            "lab_results": LoaderParams(
                                project_info=project_info,
                                values_loader=[
                                    "alat",
                                    "hdl",
                                    "ldl",
                                    "triglycerides",
                                    "fasting_ldl",
                                    "crp",
                                ],
                                lookbehind_days=lookbehind_days,
                                resolve_multiple_fn=resolve_multiple,
                                fallback=[np.nan],
                                allowed_nan_value_prop=allowed_nan_value_prop,
                            )
                        },
                    }
    
    main_pipeline_instance = main_pipeline(
        quarantine_df_loader=quarantine_df_loader(),
        prediction_times_loader=prediction_times_loader(params=PredTimeParams(quarantine_days=730, project_info=project_info)),
        load_and_flatten_somatic_medications=load_and_flatten_predictors(params=loader_params["somatic"]["medications"]),
        load_and_flatten_somatic_diagnoses=load_and_flatten_predictors(params=loader_params["somatic"]["diagnoses"]),
        load_and_flatten_somatic_lab_results=load_and_flatten_predictors(params=loader_params["somatic"]["lab_results"]),
        load_and_flatten_psychiatric_medications=load_and_flatten_predictors(params=loader_params["psychiatric"]["medications"]),
        load_and_flatten_psychiatric_diagnoses=load_and_flatten_predictors(params=loader_params["psychiatric"]["diagnoses"]),
        load_and_flatten_psychiatric_lab_results=load_and_flatten_predictors(params=loader_params["psychiatric"]["lab_results"]),
    )
    
    main_pipeline_instance.run(unlisted=True)
        