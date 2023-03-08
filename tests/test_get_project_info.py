from psycop_feature_generation.application_modules.project_setup import (
    ProjectInfo,
    get_project_info,
)


def test_get_project_info():
    """Manual test to ensure that at least one test runs."""
    project_info = get_project_info(
        project_name="t2d-testing",
    )

    assert isinstance(project_info, ProjectInfo)
