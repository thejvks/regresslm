"""Runtime configuration."""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="REGRESSLM_", env_file=".env", extra="ignore")

    runs_dir: str = ".regresslm/runs"
    # Gate thresholds (overridable per CI job).
    max_score_drop: float = 0.02
    max_newly_failing: int = 0
    # Default judge model when using the real LLMJudge.
    judge_model: str = "claude-haiku-4-5"


def get_settings() -> Settings:
    return Settings()
