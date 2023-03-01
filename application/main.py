"""Main feature generation."""
import logging

from modules.get_predictor_params import get_predictor_params
from psycop_feature_generation.application_modules.loggers import init_root_logger
from psycop_feature_generation.application_modules.project_setup import get_project_info

from t2d_feature_generation.pipelines.dynamic_main_pipeline import FeatureGeneration
from t2d_feature_generation.pipelines.main_pipeline import main_pipeline
from t2d_feature_generation.steps.concatenators import (
    combined_concatenator,
    feature_concatenator,
)
from t2d_feature_generation.steps.dataset_saver import DatasetSaverParams, dataset_saver
from t2d_feature_generation.steps.loaders.outcome_loader import (
    OutcomeLoaderParams,
    load_and_flatten_outcomes,
)
from t2d_feature_generation.steps.loaders.prediction_times_loader import (
    PredTimeParams,
    prediction_times_loader,
)
from t2d_feature_generation.steps.loaders.predictor_loader import (
    load_and_flatten_predictors,
)
from t2d_feature_generation.steps.loaders.quarantine_df_loader import (
    quarantine_df_loader,
)
from t2d_feature_generation.steps.loaders.static_loader import (
    StaticLoaderParams,
    load_and_flatten_static_specs,
)

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

    predictor_params = get_predictor_params(eval_prefix=project_info.prefix.eval)
    static_loader = load_and_flatten_static_specs(params=StaticLoaderParams())
    outcome_loader = load_and_flatten_outcomes(
        params=OutcomeLoaderParams(
            values_loader=["t2d"],
            lookahead_days=[year * 365 for year in (1, 2, 3, 4, 5)],
            resolve_multiple_fn=["max"],
            fallback=[0],
            incident=[True],
            allowed_nan_value_prop=[0],
        ),
    )

    FeatureGeneration(
        quarantine_df_loader=quarantine_df_loader(),
        prediction_time_loader=prediction_times_loader(
            params=PredTimeParams(
                quarantine_days=730,
                entity_id_col_name="dw_ek_borger",
            ),
        ),
        predictor_confs=predictor_params,
        predictor_concatenator=feature_concatenator,
        outcome_loader=outcome_loader,
        static_loader=static_loader,
        combined_concatenator=combined_concatenator(),
        dataset_saver=dataset_saver(
            params=DatasetSaverParams(project_info=project_info),
        ),
    ).run(unlisted=True)
