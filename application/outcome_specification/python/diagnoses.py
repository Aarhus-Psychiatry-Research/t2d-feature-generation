from psycop_feature_generation.loaders.raw.load_lab_results import (
    fasting_p_glc,
    hba1c,
    ogtt,
    unscheduled_p_glc,
)
import pandas as pd


def get_rows_above_value(value: float, df: pd.DataFrame, value_type: str):
    output_df = df[df["value"] > value]
    output_df["value_type"] = value_type

    return output_df


def get_hba1cs_above_threshold():
    return get_rows_above_value(df=hba1c(), value=48.0, value_type="hba1c")


def get_unscheduled_p_glc_above_threshold():
    return get_rows_above_value(
        df=unscheduled_p_glc(), value=11.0, value_type="unscheduled_p_glc"
    )


def get_fasting_glc_above_threshold():
    return get_rows_above_value(
        df=fasting_p_glc(), value=7.0, value_type="fasting_p_glc"
    )


def get_ogtt_above_threshold():
    return get_rows_above_value(df=ogtt(), value=11.0, value_type="ogtt")


def get_diabetes_lab_results_above_threshold():
    return pd.concat(
        [
            get_hba1cs_above_threshold(),
            get_unscheduled_p_glc_above_threshold(),
            get_fasting_glc_above_threshold(),
            get_ogtt_above_threshold(),
        ],
        axis=0,
    )


if __name__ == "__main__":
    pass
