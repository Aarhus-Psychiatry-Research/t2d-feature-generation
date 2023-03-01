from itertools import chain
from typing import List

from pipelines.dynamic_pipelines.dynamic_pipeline import DynamicPipeline
from steps.feature_concatenator import ConcatenatorParams
from steps.loaders.predictor_loader import (
    PredictorLoaderParams,
    load_and_flatten_predictors,
)
from zenml.steps import BaseParameters, BaseStep


class FeatureGeneration(DynamicPipeline):
    """Generates the steps of the hyperparameter tuning pipeline dynamically based on the input, and connects
    the steps."""

    def __init__(
        self,
        quarantine_df_loader: BaseStep,
        prediction_time_loader: BaseStep,
        predictor_confs: List[PredictorLoaderParams],
        feature_concatenator: BaseStep,
        **kwargs
    ) -> None:
        """
        Initialize the pipeline by creating the step instances used by it.
        """
        self.quarantine_df_loader = quarantine_df_loader
        self.prediction_time_loader = prediction_time_loader

        self.predictor_loading_steps = [
            [load_and_flatten_predictors(params=pred_params).configure(
                name=pred_params.predictor_group_name
            )]
            for pred_params in predictor_confs
        ]
        
        output_step_names = [step[-1].name for step in self.predictor_loading_steps]
        self.feature_concatenator = feature_concatenator(params=ConcatenatorParams(output_steps_names=output_step_names))

        super().__init__(
            self.quarantine_df_loader,
            self.prediction_time_loader,
            self.feature_concatenator,
            *chain.from_iterable(self.predictor_loading_steps),
            **kwargs
        )

    def connect(self, **kwargs: BaseStep) -> None:
        """
        The method connects the input and outputs of the hyperparameter tuning pipeline.

        Args:
            **kwargs: the step instances of the pipeline.
        """
        quarantine_df = self.quarantine_df_loader()
        prediction_times = self.prediction_time_loader(quarantine_df=quarantine_df)
        
        for predictor_step in self.predictor_loading_steps:
            step = predictor_step[0]
            step(prediction_times=prediction_times)
            self.feature_concatenator.after(step)
        
        concatenated_predictors = self.feature_concatenator()
        