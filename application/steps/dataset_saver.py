from psycop_feature_generation.application_modules.project_setup import ProjectInfo
from psycop_feature_generation.application_modules.save_dataset_to_disk import (
    split_and_save_dataset_to_disk,
)
from zenml.steps import BaseParameters, step


class DatasetSaverParams(BaseParameters):
    project_info: ProjectInfo
    

def dataset_saver(params: DatasetSaverParams, df: pd.DataFrame) -> None:
    split_and_save_dataset_to_disk(flattened_df=df, project_info=params.project_info)
    