"""
EarthTrend Explorer — Plain-Language Scientific Storytelling (Checkpoint 8)
==========================================================================

Rule-based sentence builders that turn VERIFIED model outputs into short explanations
for a non-specialist reader (judges, students, decision makers).

Ground rules (scientific honesty):
  * Every number in a sentence is read from an analysis result — nothing is invented.
  * Wording strength follows the statistics: "clearly" only when the test is significant,
    "may" / "not detectable" otherwise.
  * Three kinds of statement are kept apart and labelled:
      - OBSERVED  : what the NASA record shows (deterministic description),
      - STATISTICAL: whether that could be chance (test result),
      - PROJECTED : what a model expects next (uncertain by construction).
  * No causal claims: we never say *why* something changed (e.g. "because of climate
    change") — the data alone cannot show that.

Each builder returns a list of (kind, sentence) tuples, kind in {"observed", "statistical",
"projected", "caution"}, so the UI can tag them visually.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import pandas as pd

Story = List[Tuple[str, str]]


def _chance_phrase(p: float) -> str:
    """Translate a p-value into everyday language."""
    if p < 0.001:
        return "less than a 1-in-1,000 chance"
    if p < 0.01:
        return f"about a 1-in-{max(2, int(round(1 / p))):,} chance"
    if p < 0.05:
        return f"about a 1-in-{max(2, int(round(1 / p)))} chance"
    return "a high chance"


# ------------------------------------------------------------------------------
# Rainfall
# ------------------------------------------------------------------------------
def rainfall_story(res: Dict[str, Any], region: str) -> Story:
    t = res["trends"]["annual_total"]
    cp = res["change_point"]
    out: Story = [(
        "observed",
        f"{region} gets about {t['mean']:.0f} mm of rain in a typical year — roughly "
        f"{t['mean'] / 10:.0f} cm of water if it all stayed on the ground — and "
        f"{res['dry_month_fraction'] * 100:.0f}% of months are practically dry.",
    )]
    if t["is_significant"]:
        direction = "wetter" if t["sen_slope_per_decade"] > 0 else "drier"
        out.append(("statistical",
                    f"Rainfall has been getting steadily {direction} "
                    f"({t['sen_slope_per_decade']:+.1f} mm per decade), and that drift is unlikely to be chance."))
    else:
        out.append(("statistical",
                    "There is no steady, year-by-year drift in rainfall: wet and dry years come and go "
                    "without a clear direction."))
    if cp["is_significant"]:
        out.append(("statistical",
                    f"Instead, the record shows a sudden step: before {cp['new_regime_start']} a typical year had about "
                    f"{cp['mean_before']:.0f} mm, after it about {cp['mean_after']:.0f} mm "
                    f"({cp['pct_change']:+.0f}%). A jump this large has {_chance_phrase(cp['p_value'])} of being random."))
        out.append(("caution",
                    "We can say WHEN rainfall changed, not WHY. A step this sharp could be real weather, or a change in "
                    "how the satellite/reanalysis system measures rain; it needs checking against rain gauges or the "
                    "GPM satellite record before anyone calls it climate change."))
    else:
        out.append(("statistical", "There is also no sudden jump in the rainfall record."))
    if t.get("low_reliability"):
        out.append(("caution",
                    "Rain here is so rare that most years are zeros — treat every rainfall statistic for this place "
                    "as low-confidence."))
    return out


# ------------------------------------------------------------------------------
# Temperature extremes
# ------------------------------------------------------------------------------
def extremes_story(res: Dict[str, Any], region: str) -> Story:
    tr = res["trends"]
    hot = tr["annual_hottest"]
    cold = tr["annual_coldest"]
    hm = tr["hot_months"]
    ann = res["annual"]
    early = ann[ann["year"] <= 1990]["hot_months"].mean()
    late = ann[ann["year"] >= 2011]["hot_months"].mean()
    last = int(ann["year"].max())

    out: Story = []
    if hot["is_significant"]:
        out.append(("statistical",
                    f"The hottest moment of each year in {region} is getting hotter by about "
                    f"{hot['sen_per_decade']:.2f} °C every 10 years — a real trend, not luck."))
    else:
        out.append(("statistical",
                    f"The hottest moment of the year in {region} moves around, but we cannot confirm a steady rise."))
    if early > 0 and not math.isnan(late):
        out.append(("observed",
                    f"In the 1980s a year had about {early:.1f} 'unusually hot' months (hot for that time of year); "
                    f"in 2011–{last} it had about {late:.1f} — roughly {late / early:.1f} times as many."))
    elif not math.isnan(late):
        out.append(("observed",
                    f"In 2011–{last} a year had about {late:.1f} 'unusually hot' months, compared with almost none in the 1980s."))
    if hm["is_significant"]:
        out.append(("statistical", "That rise in hot months is statistically significant."))
    if cold["is_significant"] and cold["sen_per_decade"] > 0:
        out.append(("statistical",
                    f"The coldest moments of the year are also getting milder (+{cold['sen_per_decade']:.2f} °C per decade)."))
    elif not cold["is_significant"]:
        out.append(("statistical",
                    "The coldest moment of the year has not changed in a way we can confirm — winters still bring cold snaps."))
    out.append(("caution",
                "'Hottest' here is the hottest hour NASA's model recorded in each month, for one ~50 km grid cell — "
                "a city street can be hotter."))
    return out


# ------------------------------------------------------------------------------
# Forecast
# ------------------------------------------------------------------------------
def forecast_story(fc_res: Dict[str, Any], variable: str, units: str, location: str) -> Story:
    k = fc_res["summary_kpis"]
    bt = fc_res["backtest_results"]
    win = bt["summary_metrics"][bt["tournament_winner_key"]]
    fc_df: pd.DataFrame = fc_res["forecast_df"]
    start = fc_df["date"].iloc[0].strftime("%b %Y")
    end = fc_df["date"].iloc[-1].strftime("%b %Y")
    out: Story = []

    if fc_res["is_precipitation"]:
        out.append(("projected",
                    f"For rain we show the forecast mainly as a warning sign: in testing on 2018–2022 no model beat the "
                    f"simple 'same as last year' guess by a useful margin (MASE = {win['mase']:.2f}; below 1 would mean "
                    "better). Rain in Egypt comes in rare, random storms that cannot be predicted months ahead."))
        out.append(("caution", "Use the rainfall bands as a range of what is plausible, not as a prediction."))
        return out

    mean_anom = k["mean_expected_anomaly"]
    warmer = "warmer" if mean_anom >= 0 else "cooler"
    out.append(("projected",
                f"For {start} – {end}, the best model expects {location} to be on average about "
                f"{abs(mean_anom):.1f} {units} {warmer} than the 1981–2010 normal."))
    if not math.isnan(k.get("mean_prob_above_normal", float("nan"))):
        out.append(("projected",
                    f"Month by month, the chance of being warmer than normal averages about "
                    f"{k['mean_prob_above_normal']:.0f}%, and {k['months_likely_warm']} of the 12 months are more likely "
                    "than not to be unusually warm (above the normal by one typical year-to-year swing)."))
    beat = "beat" if win["mase"] < 1 else "did not beat"
    out.append(("statistical",
                f"Why trust it? We hid the last five years from the model, asked it to forecast them, and checked: it "
                f"{beat} the simple 'same as last year' guess, missing by about {win['mean_mae']:.1f} {units} per month "
                f"on average."))
    c95 = win.get("coverage_95")
    if c95 is not None:
        out.append(("statistical",
                    f"Its 95% uncertainty band caught {c95:.0f}% of the months it had never seen — "
                    + ("close to the promised 95%, so the band is honest."
                       if abs(c95 - 95) <= 6 else "so read the band with some caution.")))
    out.append(("caution",
                "A forecast is a best guess with a range, not a promise — the shaded band is as important as the line."))
    return out


# ------------------------------------------------------------------------------
# Regional comparison
# ------------------------------------------------------------------------------
def regional_story(diff_df: pd.DataFrame, variable_label: str) -> Story:
    if diff_df is None or diff_df.empty:
        return [("caution", "Not enough regions loaded to compare warming speeds.")]
    out: Story = []
    sig = diff_df[diff_df["is_significant"]]
    for _, r in sig.iterrows():
        out.append(("statistical",
                    f"{r['faster_region']} is warming faster than {r['slower_region']}: about "
                    f"{r['difference_per_decade']:.2f} °C more every 10 years in {variable_label}. "
                    "This difference passed every test we ran, including the correction for comparing several pairs."))
    nonsig = diff_df[~diff_df["is_significant"]]
    if not nonsig.empty:
        pairs = ", ".join(f"{r['faster_region']} vs {r['slower_region']}" for _, r in nonsig.iterrows())
        out.append(("statistical",
                    f"For {pairs}, the warming speeds look different on paper, but the gap is small enough to be "
                    "chance — so we do not claim one is faster."))
    out.append(("caution",
                "All three places are warming; this only asks whether they warm at the SAME speed."))
    return out


# ------------------------------------------------------------------------------
# Forecast vs newly released observations
# ------------------------------------------------------------------------------
def verification_story(ver: Dict[str, Any], units: str, location: str, is_precipitation: bool) -> Story:
    if not ver.get("available"):
        return [("caution", "No newer NASA observations are available yet to check this forecast.")]
    out: Story = [(
        "observed",
        f"This forecast was built only from data up to the end of the previous year. NASA has since released the "
        f"real values for {ver['period']}, so we can check it like a weather forecast after the fact.",
    )]
    if is_precipitation:
        out.append(("statistical",
                    f"For rain the forecast missed by {ver['mae']:.2f} {units} per month on average, versus "
                    f"{ver['naive_mae']:.2f} for simply repeating last year — no real skill, as the backtest warned."))
        return out
    better = ver["mae"] < ver["naive_mae"]
    out.append(("statistical",
                f"Month by month it missed by {ver['mae']:.1f} {units} on average — "
                + (f"better than simply repeating last year ({ver['naive_mae']:.1f} {units})."
                   if better else f"no better than simply repeating last year ({ver['naive_mae']:.1f} {units}).")))
    out.append(("statistical",
                f"{ver['hit_95']:.0f}% of the real months landed inside the 95% band "
                f"(and {ver['hit_80']:.0f}% inside the 80% band)."))
    pa, oa = ver["predicted_mean_anomaly"], ver["observed_mean_anomaly"]
    if pa > 0 and oa > 0:
        if oa > pa + 0.1:
            tail = "Reality turned out even warmer than we predicted."
        elif oa < pa - 0.1:
            tail = "Reality was a little less warm than predicted, but still clearly above normal."
        else:
            tail = "Almost exactly what we predicted."
    else:
        tail = ""
    out.append(("observed",
                f"We expected {location} to run {pa:+.1f} {units} versus the 1981–2010 normal; it actually ran "
                f"{oa:+.1f} {units}. {tail}".strip()))
    return out
