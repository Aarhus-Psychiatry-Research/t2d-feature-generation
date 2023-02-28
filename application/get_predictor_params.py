import numpy as np
from steps.loaders.predictor_loader import PredictorLoaderParams


def get_predictor_params(
    project_info, lookbehind_days, resolve_multiple, allowed_nan_value_prop
):
    lookbehind_days = [30, 90, 180, 365, 730]
    resolve_multiple = ["max", "min", "mean", "latest"]
    allowed_nan_value_prop = [0]

    return {
        "somatic_medications": PredictorLoaderParams(
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
        "somatic_diagnoses": PredictorLoaderParams(
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
        "psychiatric_medications": PredictorLoaderParams(
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
        "psychiatric_diagnoses": PredictorLoaderParams(
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
        "metadata_from_predictors": PredictorLoaderParams(
            values_loader=["hba1c"],
            fallback=[np.nan],
            lookbehind_days=[9999],
            resolve_multiple_fn=["count"],
            allowed_nan_value_prop=allowed_nan_value_prop,
            prefix=project_info.prefix.eval,
        ),
    }
