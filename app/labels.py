"""Human-readable labels for raw dataset / encoder class strings."""

OUTCOME_DISPLAY: dict[str, str] = {
    "Home Team": "Home win",
    "Away Team": "Away win",
    "Draw": "Draw",
}


def outcome_display(raw: str) -> str:
    return OUTCOME_DISPLAY.get(raw, raw)
