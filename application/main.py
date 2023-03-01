"""Main feature generation."""
import logging

from get_predictor_params import get_predictor_params
from pipelines.dynamic_main_pipeline import FeatureGeneration
from pipelines.main_pipeline import main_pipeline
from psycop_feature_generation.application_modules.loggers import init_root_logger
from psycop_feature_generation.application_modules.project_setup import get_project_info
from steps.dataset_saver import DatasetSaverParams, dataset_saver
from steps.feature_concatenator import feature_concatenator
from steps.loaders.outcome_loader import OutcomeLoaderParams, load_and_flatten_outcomes
from steps.loaders.prediction_times_loader import (
    PredTimeParams,
    prediction_times_loader,
)
from steps.loaders.predictor_loader import load_and_flatten_predictors
from steps.loaders.quarantine_df_loader import quarantine_df_loader
from steps.loaders.static_loader import (
    StaticLoaderParams,
    load_and_flatten_static_specs,
)

log = logging.getLogger()


def functional_pipeline():
    outcome_params = OutcomeLoaderParams(
        values_loader=["t2d"],
        lookahead_days=[year * 365 for year in (1, 2, 3, 4, 5)],
        resolve_multiple_fn=["max"],
        fallback=[0],
        incident=[True],
        allowed_nan_value_prop=[0],
    )

    MAIN_PIPELINE_INSTANCE = main_pipeline(  # pylint: disable=no-value-for-parameter
        quarantine_df_loader=quarantine_df_loader(),
        prediction_times_loader=prediction_times_loader(  # pylint: disable=no-value-for-parameter
            params=PredTimeParams(
                quarantine_days=730, entity_id_col_name="dw_ek_borger"
            )
        ),
        load_and_flatten_somatic_medications=predictor_steps["somatic_medications"],
        load_and_flatten_somatic_diagnoses=predictor_steps["somatic_diagnoses"],
        load_and_flatten_general_labs=predictor_steps["general_labs"],
        load_and_flatten_diabetes_labs=predictor_steps["diabetes_labs"],
        load_and_flatten_psychiatric_medications=predictor_steps[
            "psychiatric_medications"
        ],
        load_and_flatten_psychiatric_diagnoses=predictor_steps["psychiatric_diagnoses"],
        load_and_flatten_metadata_from_predictor=predictor_steps[
            "metadata_from_predictors"
        ],
        load_and_flatten_static_metadata=load_and_flatten_static_specs(  # pylint: disable=no-value-for-parameter
            params=StaticLoaderParams()
        ),  # pylint: disable=no-value-for-parameter
        load_and_flatten_outcomes=load_and_flatten_outcomes(  # pylint: disable=no-value-for-parameter
            params=outcome_params
        ),  # pylint: disable=no-value-for-parameter
        feature_concatenator=feature_concatenator(),
        dataset_saver=dataset_saver(
            params=DatasetSaverParams(project_info=project_info)
        ),
    )

    MAIN_PIPELINE_INSTANCE.run(unlisted=True)


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

    predictor_params = get_predictor_params(eval_prefix=project_info.prefix.eval)

    FeatureGeneration(
        quarantine_df_loader=quarantine_df_loader(),
        prediction_time_loader=prediction_times_loader(
            params=PredTimeParams(
                quarantine_days=730, entity_id_col_name="dw_ek_borger"
            )
        ),
        predictor_confs=predictor_params,
        feature_concatenator=feature_concatenator,
        dataset_saver=dataset_saver(
            params=DatasetSaverParams(project_info=project_info)
        ),
    ).run(unlisted=True)
