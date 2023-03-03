from psycop_feature_generation.loaders.raw.load_visits import (
    physical_visits_to_psychiatry,
)
from ydata_profiling import ProfileReport

if __name__ == "__main__":
    datasets = {
        "start": physical_visits_to_psychiatry(
            timestamps_only=True,
            timestamp_for_output="start",
        ),
        "end": physical_visits_to_psychiatry(
            timestamps_only=True,
            timestamp_for_output="end",
        ),
    }

    for ds_type in ("start", "end"):
        start_profile = ProfileReport(datasets[ds_type])
        output = start_profile.to_file(f"{ds_type}.html")

    pass
