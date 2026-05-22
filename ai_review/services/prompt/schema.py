from pydantic import BaseModel, Field, field_serializer

from ai_review.config import settings
from ai_review.libs.template.render import render_template

# Marker inserted between the stable (cacheable) and variable parts of a system
# prompt. Square brackets keep it disjoint from the default `<<value>>` template
# placeholder syntax, so render_template never touches it.
SYSTEM_PROMPT_CACHE_BOUNDARY = "\n\n[[AI_REVIEW_CACHE_BOUNDARY_DO_NOT_EDIT]]\n\n"


def split_system_prompt(prompt: str) -> tuple[str, str]:
    """Return (cached_prefix, variable). cached_prefix is empty when no boundary is present."""
    if SYSTEM_PROMPT_CACHE_BOUNDARY in prompt:
        cached, variable = prompt.split(SYSTEM_PROMPT_CACHE_BOUNDARY, 1)
        return cached, variable
    return "", prompt


def strip_system_prompt_boundary(prompt: str) -> str:
    """Strip the boundary and restore the legacy default-first ordering.

    Caching mode lays out the prompt as `user_prefix || BOUNDARY || stage_default`
    so the stable prefix matches the provider cache. Non-caching clients
    (and the agent loop's `original_prompt_system`) need the upstream
    v0.67.0 byte ordering of `stage_default + user_prefix` to keep model
    behavior unchanged when caching is off.
    """
    if SYSTEM_PROMPT_CACHE_BOUNDARY not in prompt:
        return prompt
    prefix, variable = prompt.split(SYSTEM_PROMPT_CACHE_BOUNDARY, 1)
    if prefix and variable:
        return f"{variable}\n\n{prefix}"
    return prefix or variable


class PromptContextSchema(BaseModel):
    review_title: str = ""
    review_description: str = ""

    review_author_name: str = ""
    review_author_username: str = ""

    review_reviewer: str = ""
    review_reviewers: list[str] = Field(default_factory=list)
    review_reviewers_usernames: list[str] = Field(default_factory=list)

    review_assignees: list[str] = Field(default_factory=list)
    review_assignees_usernames: list[str] = Field(default_factory=list)

    source_branch: str = ""
    target_branch: str = ""

    labels: list[str] = Field(default_factory=list)
    changed_files: list[str] = Field(default_factory=list)

    @field_serializer(
        "review_reviewers",
        "review_reviewers_usernames",
        "review_assignees",
        "review_assignees_usernames",
        "labels",
        "changed_files",
        when_used="always"
    )
    def list_of_strings_serializer(self, value: list[str]) -> str:
        return ", ".join(value)

    def apply_format(self, prompt: str) -> str:
        values = {**self.model_dump(), **settings.prompt.context}
        return render_template(prompt, values, settings.prompt.context_placeholder)
