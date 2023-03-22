import pandas as pd
from t2d_feature_generation.outcome_specification.lab_results import (
    get_first_diabetes_lab_result_above_threshold,
)
from t2d_feature_generation.outcome_specification.medications import (
    get_first_antidiabetic_medication,
)
from t2d_feature_generation.outcome_specification.t1d_diagnoses import (
    get_first_type_1_diabetes_diagnosis,
)
from t2d_feature_generation.outcome_specification.t2d_diagnoses import (
    get_first_type_2_diabetes_diagnosis,
)
from timeseriesflattener.utils import data_loaders


@data_loaders.register("first_diabetes_indicator")
def get_first_diabetes_indicator() -> pd.DataFrame:
    t1d_diagnoses = get_first_type_1_diabetes_diagnosis()
    t2d_diagnoses = get_first_type_2_diabetes_diagnosis()
    medications = get_first_antidiabetic_medication()
    lab_results = get_first_diabetes_lab_result_above_threshold()

    combined = pd.concat(
        [
            t1d_diagnoses,
            t2d_diagnoses,
            medications,
            lab_results,
        ],
        axis=0,
    )

    first_diabetes_indicator = (
        combined.sort_values("timestamp")
        .groupby("dw_ek_borger")
        .first()
        .reset_index(drop=False)
    )

    first_diabetes_indicator["value"] = 1
    return first_diabetes_indicator[["dw_ek_borger", "timestamp", "value"]]


if __name__ == "__main__":
    df = get_first_diabetes_indicator()
