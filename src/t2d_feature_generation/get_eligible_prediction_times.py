
# %%
from psycop_feature_generation.loaders.raw.load_visits import (
    physical_visits_to_psychiatry,
)

min_date = datetime(year=2013, month=1, day=1)

# %load_ext autoreload
# %autoreload 2

# %%
all_prediction_times = physical_visits_to_psychiatry(
    timestamps_only=True, timestamp_for_output="start"
)

# %%
prediction_times_after_min_date = all_prediction_times[all_prediction_times["timestamp"] > min_date]
print(prediction_times_after_min_date.shape[0])




# %%
#########################################
# Filtering based on prevalent diabetes #
#########################################
from t2d_feature_generation.outcome_specification.combined import (
    get_first_diabetes_indicator,
)

first_diabetes_indicator = get_first_diabetes_indicator()

# %%
from datetime import datetime

indicator_before_min_date = first_diabetes_indicator[first_diabetes_indicator["timestamp"] < min_date]

# %%
prediction_times_from_patients_with_diabetes = prediction_times_after_min_date.merge(indicator_before_min_date, how="inner", on="dw_ek_borger")

# %% Summarised
print(prediction_times_from_patients_with_diabetes.groupby("source").count()["value"])
print(prediction_times_from_patients_with_diabetes.shape[0])

# %%
#########################
# No prevalent diabetes #
#########################
no_prevalent_diabetes = prediction_times_after_min_date.merge(right=indicator_before_min_date, how="left", on="dw_ek_borger")
no_prevalent_diabetes = no_prevalent_diabetes[no_prevalent_diabetes["source"].isna()][["dw_ek_borger", "timestamp_x"]]
print(no_prevalent_diabetes.shape[0])

# %%
#############################################
# Contacts after  incident diagnosis of T2D #
#############################################
from t2d_feature_generation.outcome_specification.lab_results import get_first_diabetes_lab_result_above_threshold

results_above_threshold = get_first_diabetes_lab_result_above_threshold()
no_prevalent_diabetes_with_hba1c = no_prevalent_diabetes.merge(results_above_threshold, on="dw_ek_borger", how="left")
after_incident_diabetes = no_prevalent_diabetes_with_hba1c[no_prevalent_diabetes_with_hba1c["value"].notnull()]
print(after_incident_diabetes.shape[0])

not_after_incident_diabetes = no_prevalent_diabetes_with_hba1c[no_prevalent_diabetes_with_hba1c["value"].isna()][["dw_ek_borger", "timestamp_x"]].rename({"timestamp_x": "timestamp"}, axis=1)
print(not_after_incident_diabetes.shape[0])
# %%
############################################################################
# Contacts within 2 years from entering the central Central Denmark Region #
############################################################################
from psycop_feature_generation.application_modules.filter_prediction_times import (
    PredictionTimeFilterer,
)
from psycop_feature_generation.loaders.raw.load_moves import (
    load_move_into_rm_for_exclusion,
)
not_within_two_years_from_move = PredictionTimeFilterer(prediction_times_df=not_after_incident_diabetes, entity_id_col_name="dw_ek_borger", quarantine_timestamps_df=load_move_into_rm_for_exclusion(), quarantine_interval_days=730).run_filter()
print(f"Within 2 years from move: {not_after_incident_diabetes.shape[0] - not_within_two_years_from_move.shape[0]}")
print(f"Remaining: {not_within_two_years_from_move.shape[0]}")

# %%
###################################################
# Whether T2D in the 3 years following prediction #
###################################################