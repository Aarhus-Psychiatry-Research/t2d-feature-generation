from typing import Callable

import pandas as pd
from zenml.pipelines import pipeline


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
    load_and_flatten_static_metadata: Callable,
    load_and_flatten_outcomes: Callable,
    load_and_flatten_metadata_from_predictor: Callable,
    dataset_saver: Callable,
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
        load_and_flatten_static_metadata(prediction_times=prediction_times),
        load_and_flatten_outcomes(prediction_times=prediction_times),
        load_and_flatten_metadata_from_predictor(prediction_times=prediction_times),
    ]

    # Check that all indeces match and concat the dataframes
    assert all(features[0].index.equals(f.index) for f in features)
    concatenated_df = pd.concat(features, axis=1)

    dataset_saver(df=concatenated_df)
