from pydantic import BaseModel


class CalculateCostSchema(BaseModel):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0


class CostReportSchema(BaseModel):
    model: str
    prompt_tokens: int
    completion_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0

    @property
    def cache_hit(self) -> bool:
        return self.cache_read_tokens > 0

    @property
    def prompt_percent(self) -> float:
        return (self.input_cost / self.total_cost * 100) if self.total_cost else 0.0

    @property
    def completion_percent(self) -> float:
        return (self.output_cost / self.total_cost * 100) if self.total_cost else 0.0

    @property
    def pretty_total_line(self) -> str:
        return f"- {'Total:':<20} {'':>7}   {self.total_cost:12.6f} USD"

    @property
    def pretty_prompt_line(self) -> str:
        return (
            f"- {'Prompt tokens:':<20} {self.prompt_tokens:>7} → "
            f"{self.input_cost:12.6f} USD ({self.prompt_percent:.1f}%)"
        )

    @property
    def pretty_completion_line(self) -> str:
        return (
            f"- {'Completion tokens:':<20} {self.completion_tokens:>7} → "
            f"{self.output_cost:12.6f} USD ({self.completion_percent:.1f}%)"
        )

    @property
    def pretty_cache_line(self) -> str | None:
        if not (self.cache_creation_tokens or self.cache_read_tokens):
            return None
        parts = []
        if self.cache_creation_tokens:
            parts.append(f"created {self.cache_creation_tokens:,}")
        if self.cache_read_tokens:
            parts.append(f"read {self.cache_read_tokens:,}")
        return f"- {'Cache tokens:':<20} {'':>7}   {', '.join(parts)}"

    def pretty(self) -> str:
        lines = [
            f"\n💰 Estimated Cost for `{self.model}`",
            self.pretty_prompt_line,
            self.pretty_completion_line,
        ]
        cache_line = self.pretty_cache_line
        if cache_line:
            lines.append(cache_line)
        lines.append(self.pretty_total_line)
        return "\n".join(lines) + "\n"
