"""Checked-in project assets and generated run outputs have distinct roots."""

from src.runtime.config import (
    PROJECT_ASSETS_DIR,
    PROJECTS_DIR,
    problem_definition_path,
    project_assets_dir,
    project_dir,
    project_prompts_dir,
    references_dir,
)


def test_project_asset_helpers_use_the_declared_asset_root():
    project_id = "subjective_randomness"
    assets = project_assets_dir(project_id)

    assert assets == PROJECT_ASSETS_DIR / project_id
    assert problem_definition_path(project_id) == assets / "problem_definition.md"
    assert references_dir(project_id) == assets / "references"
    assert project_prompts_dir(project_id) == assets / "prompts"
    assert project_dir(project_id) == PROJECTS_DIR / project_id


def test_all_checked_in_project_assets_live_under_one_root():
    assert problem_definition_path("think_aloud_game24").is_file()
    assert (project_prompts_dir("think_aloud_game24") / "1_theory.md").is_file()
    assert (references_dir("number_game") / "bigelow2016inferring.pdf").is_file()
