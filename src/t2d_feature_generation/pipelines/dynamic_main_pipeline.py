from itertools import chain

from zenml.steps import BaseStep

from t2d_feature_generation.pipelines.dynamic_pipelines.dynamic_pipeline import (
    DynamicPipeline,
)
from t2d_feature_generation.steps.concatenators import ConcatenatorParams
from t2d_feature_generation.steps.loaders.predictor_loader import (
    PredictorLoaderParams,
    load_and_flatten_predictors,
)


class FeatureGeneration(DynamicPipeline):
    """Generates the steps of the hyperparameter tuning pipeline dynamically based on the input, and connects
    the steps."""

    def __init__(
        self,
        quarantine_df_loader: BaseStep,
        prediction_time_loader: BaseStep,
        predictor_confs: list[PredictorLoaderParams],
        predictor_concatenator: BaseStep,
        outcome_loader: BaseStep,
        static_loader: BaseStep,
        combined_concatenator: BaseStep,
        dataset_saver: BaseStep,
        **kwargs,
    ) -> None:
        """
        Initialize the pipeline by creating the step instances used by it.
        """
        self.quarantine_df_loader = quarantine_df_loader
        self.prediction_time_loader = prediction_time_loader

        self.predictor_loading_steps = [
            [
                load_and_flatten_predictors(params=pred_params).configure(
                    name=pred_params.predictor_group_name,
                ),
            ]
            for pred_params in predictor_confs
        ]
        predictor_step_names = [step[-1].name for step in self.predictor_loading_steps]
        self.predictor_concatenator = predictor_concatenator(
            params=ConcatenatorParams(output_steps_names=predictor_step_names),
        )

        self.outcome_loader = outcome_loader
        self.static_loader = static_loader
        self.combined_concatenator = combined_concatenator
        self.dataset_saver = dataset_saver

        super().__init__(
            self.quarantine_df_loader,
            self.prediction_time_loader,
            self.predictor_concatenator,
            self.outcome_loader,
            self.static_loader,
            self.combined_concatenator,
            self.dataset_saver,
            *chain.from_iterable(self.predictor_loading_steps),
            **kwargs,
        )

    def connect(self, **kwargs: BaseStep) -> None:  # pylint: disable=arguments-differ
        """
        The method connects the input and outputs of the hyperparameter tuning pipeline.

        Args:
            **kwargs: the step instances of the pipeline.
        """
        quarantine_df = self.quarantine_df_loader()
        prediction_times = self.prediction_time_loader(quarantine_df=quarantine_df)

        outcomes, prediction_times_filtered_by_outcome = self.outcome_loader(
            prediction_times=prediction_times,
        )

        for predictor_step in self.predictor_loading_steps:
            step = predictor_step[0]
            step(prediction_times=prediction_times_filtered_by_outcome)
            self.predictor_concatenator.after(step)

        concatenated_predictors = self.predictor_concatenator()

        statistics = self.static_loader(
            prediction_times=prediction_times_filtered_by_outcome,
        )

        combined = self.combined_concatenator(
            outcomes=outcomes,
            statistics=statistics,
            predictors=concatenated_predictors,
        )

        self.dataset_saver(df=combined)
