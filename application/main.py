"""Main feature generation."""
import logging
from typing import Callable

import numpy as np
from application.steps.somatic_loaders import StaticLoaderParams, load_and_flatten_static_specs
from steps.somatic_loaders import OutcomeLoaderParams
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
    PredictorLoaderParams,
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
    load_and_flatten_general_labs: Callable,
    load_and_flatten_diabetes_labs: Callable,
    load_and_flatten_psychiatric_medications: Callable,
    load_and_flatten_psychiatric_diagnoses: Callable,
    load_and_flatten_psychiatric_lab_results: Callable,
    load_and_flatten_static_metadata: Callable,
    load_and_flatten_outcomes: Callable,
    split_and_save_dataset_to_disk: Callable,
    save_flattened_dataset_description_to_disk: Callable,
):
    quarantine_df = quarantine_df_loader()
    prediction_times = prediction_times_loader(quarantine_df=quarantine_df)

    features = [
        load_and_flatten_somatic_medications(prediction_times=prediction_times),
        load_and_flatten_somatic_diagnoses(prediction_times=prediction_times),
        load_and_flatten_general_labs(prediction_times=prediction_times),
        load_and_flatten_diabetes_labs(prediction_times=prediction_times),
        load_and_flatten_psychiatric_medications(prediction_times=prediction_times),
        load_and_flatten_psychiatric_diagnoses(prediction_times=prediction_times),
        load_and_flatten_psychiatric_lab_results(prediction_times=prediction_times),
        load_and_flatten_metadata(prediction_times=prediction_times),
        load_and_flatten_static_metadata(prediction_times=prediction_times),
        load_and_flatten_outcomes(prediction_times=prediction_times),    
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

    loader_params = {
        "somatic": {
            "medications": PredictorLoaderParams(
                project_info=project_info,
                values_loader=[
                    "gerd_drugs",
                    "statins",
                    "antihypertensives",
                    "diuretics",
                ],
                lookbehind_days=lookbehind_days,
                resolve_multiple_fn=resolve_multiple,
                fallback=[np.nan],
                allowed_nan_value_prop=allowed_nan_value_prop,
            ),
            "diagnoses": PredictorLoaderParams(
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
            "general_labs": PredictorLoaderParams(
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
            ),
            "diabetes_labs": PredictorLoaderParams(
                project_info=project_info,
                values_loader=[
                    "hba1c",
                    "scheduled_glc",
                    "unscheduled_p_glc",
                    "egfr",
                    "albumine_creatinine_ratio",
                ],
                lookbehind_days=lookbehind_days,
                resolve_multiple_fn=resolve_multiple,
                fallback=[np.nan],
                allowed_nan_value_prop=allowed_nan_value_prop,
            ),
        },
        "psychiatric": {
            "medications": PredictorLoaderParams(
                project_info=project_info,
                values_loader=[
                    "antipsychotics",
                    "clozapine",
                    "top_10_weight_gaining_antipsychotics",
                    "lithium",
                    "valproate",
                    "lamotrigine",
                    "benzodiazepines",
                    "pregabaline",
                    "ssri",
                    "snri",
                    "tca",
                    "selected_nassa",
                    "benzodiazepine_related_sleeping_agents",
                ],
                lookbehind_days=lookbehind_days,
                resolve_multiple_fn=resolve_multiple,
                fallback=[np.nan],
                allowed_nan_value_prop=allowed_nan_value_prop,
            ),
            "diagnoses": PredictorLoaderParams(
                project_info=project_info,
                values_loader=[
                    "f0_disorders",
                    "f1_disorders",
                    "f2_disorders",
                    "f3_disorders",
                    "f4_disorders",
                    "f5_disorders",
                    "f6_disorders",
                    "f7_disorders",
                    "f8_disorders",
                    "hyperkinetic_disorders",
                ],
                lookbehind_days=lookbehind_days,
                resolve_multiple_fn=resolve_multiple,
                fallback=[0],
                allowed_nan_value_prop=allowed_nan_value_prop,
            ),
        },
        "metadata_from_predictors": PredictorLoaderParams(
            project_info=project_info,
            values_loader=["hba1c"],
            fallback=[np.nan],
            lookbehind_days=9999,
            resolve_multiple=["count"],
            allowed_nan_value_prop=allowed_nan_value_prop,
            prefix=project_info.prefix.eval,
        ),
        "static_metadata": PredictorLoaderParams(project_info=project_info), # See load_and_flatten_static_specs for spec definitions
        "outcomes": OutcomeLoaderParams(
            project_info=project_info,
            values_loader=["t2d"],
            lookahead_days=[year * 365 for year in (1, 2, 3, 4, 5)],
            resolve_multiple_fn=["max"],
            fallback=[0],
            incident=[True],
            allowed_nan_value_prop=[0],
        )
            
    }

    main_pipeline_instance = main_pipeline(
        quarantine_df_loader=quarantine_df_loader(),
        prediction_times_loader=prediction_times_loader(
            params=PredTimeParams(quarantine_days=730, project_info=project_info)
        ),
        load_and_flatten_somatic_medications=load_and_flatten_predictors(
            params=loader_params["somatic"]["medications"]
        ),
        load_and_flatten_somatic_diagnoses=load_and_flatten_predictors(
            params=loader_params["somatic"]["diagnoses"]
        ),
        load_and_flatten_somatic_general_labs=load_and_flatten_predictors(
            params=loader_params["somatic"]["general_labs"]
        ),
        load_and_flatten_somatic_diabetes_labs=load_and_flatten_predictors(
            params=loader_params["somatic"]["diabetes_labs"]
        ),
        load_and_flatten_psychiatric_medications=load_and_flatten_predictors(
            params=loader_params["psychiatric"]["medications"]
        ),
        load_and_flatten_psychiatric_diagnoses=load_and_flatten_predictors(
            params=loader_params["psychiatric"]["diagnoses"]
        ),
        load_and_flatten_metadata_from_predictor=load_and_flatten_predictors(
            params=loader_params["metadata"]
        )
        load_and_flatten_static_metadata=load_and_flatten_static_specs(StaticLoaderParamsparams=loader_params["static_metadata"]),
        load_and_flatten_outcomes=load_and_flatten_outcomes(
            params=loader_params["outcomes"]
        )
        
    )

    main_pipeline_instance.run(unlisted=True)
