"""Main feature generation."""
import logging

from get_predictor_params import get_predictor_params
from pipelines.main_pipeline import main_pipeline
from psycop_feature_generation.application_modules.loggers import init_root_logger
from psycop_feature_generation.application_modules.project_setup import get_project_info
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
from steps.split_and_save_to_disk import dataset_saver

from application.steps.dataset_saver import DatasetSaverParams

log = logging.getLogger()


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

    predictor_params = get_predictor_params()
    outcome_params = OutcomeLoaderParams(
        project_info=project_info,
        values_loader=["t2d"],
        lookahead_days=[year * 365 for year in (1, 2, 3, 4, 5)],
        resolve_multiple_fn=["max"],
        fallback=[0],
        incident=[True],
        allowed_nan_value_prop=[0],
    )

    predictor_steps = {
        pred_name: load_and_flatten_predictors(  # pylint: disable=no-value-for-parameter
            params=pred_params
        ).configure(
            name=pred_name
        )
        for pred_name, pred_params in predictor_params.items()
    }

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
        dataset_saver=dataset_saver(DatasetSaverParams(project_info=project_info)),
    )

    MAIN_PIPELINE_INSTANCE.run(unlisted=True)
