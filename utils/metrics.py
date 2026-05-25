import re
from typing import Optional


def extract_number(text: str, pattern: str) -> Optional[float]:
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return None
    raw = match.group(1).replace(",", "")
    multiplier = 1.0
    if raw.upper().endswith("B"):
        multiplier = 1e9
        raw = raw[:-1]
    elif raw.upper().endswith("M"):
        multiplier = 1e6
        raw = raw[:-1]
    try:
        return float(raw) * multiplier
    except ValueError:
        return None


def pct_change(current: Optional[float], previous: Optional[float]) -> Optional[float]:
    if current is None or previous is None or previous == 0:
        return None
    return round(((current - previous) / abs(previous)) * 100, 2)


def extract_financial_metrics(text: str) -> dict:
    metrics = {}

    rev = extract_number(text, r"revenue[^\d]*\$?([\d,.]+[BM]?)")
    if rev:
        metrics["revenue"] = rev

    eps = extract_number(text, r"(?:eps|earnings per share)[^\d]*\$?([\d,.]+)")
    if eps:
        metrics["eps"] = eps

    net = extract_number(text, r"net income[^\d]*\$?([\d,.]+[BM]?)")
    if net:
        metrics["net_income"] = net

    margin = extract_number(text, r"operating margin[^\d]*([\d,.]+)%")
    if margin:
        metrics["operating_margin_pct"] = margin

    gross = extract_number(text, r"gross margin[^\d]*([\d,.]+)%")
    if gross:
        metrics["gross_margin_pct"] = gross

    guidance = extract_number(text, r"guidance[^\d]*\$?([\d,.]+[BM]?)")
    if guidance:
        metrics["guidance_revenue"] = guidance

    return metrics


def compute_growth(current_metrics: dict, previous_metrics: dict) -> dict:
    growth = {}
    for key in current_metrics:
        if key in previous_metrics:
            change = pct_change(current_metrics[key], previous_metrics[key])
            if change is not None:
                growth[f"{key}_growth_pct"] = change
    return growth
