# Changes: From → Into

**Branch**: `feature/rainfall-analysis`  
**Base**: `main` @ `b4b8db4` (already merged into the branch, no conflicts)  
**Scope**: rainfall analysis, temperature-extremes analysis, full UI redesign (dashboard, dark theme, responsive)  
**Status**: 36/36 tests passing · `scripts/reproducible_scientific_audit.py` passes · every page checked in the browser at 375 px, 768 px and 1366 px

> This file lists every change as **From** (how it was on `main`) → **Into** (how it is on this branch), so you can review and merge with confidence.
> All the original statistical code (`mann_kendall_test`, `hamed_rao_modified_mann_kendall`, `compute_sens_slope`, …) is **unchanged**. Everything new is additive, except the UI layout and two corrected variable descriptions.

---

## 0. TL;DR

| # | Area | From | Into |
|---|---|---|---|
| 1 | Rainfall | Only selectable in generic pages; treated like temperature (mm/day averages) | Dedicated analysis + page: totals in mm, wet season, wet months, Pettitt jump test |
| 2 | `T2M_MAX` / `T2M_MIN` | Described as "monthly mean of daily max/min" (**wrong**) | Correctly described as the **absolute monthly extreme** (hottest/coldest value in the month) |
| 3 | Temperature extremes | No dedicated analysis | New module + page with 5 extreme indices |
| 4 | Mean vs extremes | Not compared | Warming rate of T2M, T2M_MAX, T2M_MIN compared per region |
| 5 | Layout | Sidebar with 9 pages | Top navbar with 7 pages, tabs inside pages |
| 6 | Home | Long text page | Dashboard hero with live KPI cards and key findings |
| 7 | Theme | Default Streamlit + white charts (clashed in dark mode) | Consistent dark "space" theme for app **and** charts |
| 8 | Mobile | Desktop only | Responsive: phone, tablet, laptop |
| 9 | Tests | 20 | 36 |

---

## 1. Pages: From → Into (where did my page go?)

| From (old sidebar page) | Into (new navbar page → tab) |
|---|---|
| 🏠 Home & Overview | 🏠 **Home** → tab *About the mission* (old content kept as-is) |
| 📂 Data Explorer | 📂 **Data** |
| 📈 Interactive Time Series | 🌡️ **Temperature** → tab *Time series* |
| 🔬 Trend Analysis Engine | 🌡️ **Temperature** → tab *Trend statistics* |
| ⚠️ Anomaly Detection | 🌡️ **Temperature** → tab *Anomalies* |
| 🗺️ Regional Comparison | 🗺️ **Regions** (+ new "Mean vs Hottest vs Coldest" section at the bottom) |
| 📚 Methodology & Sources | 📚 **Method** |
| — | 🔥 **Extremes** (new) → tabs *Overview · Charts · Statistics · Regions* |
| — | 🌧️ **Rainfall** (new) → tabs *Overview · Charts · Statistics · Regions* |
| — | 🏠 **Home** → tab *Highlights* (new dashboard) |

---

## 2. Data source controls

| From | Into |
|---|---|
| Sidebar: "Source Mode" select → "Select Egyptian Region" → **Load Region Data** button | Top bar: **region dropdown** (Greater Cairo / Alexandria / Aswan) that switches **instantly**, no button |
| Upload CSV and Live NASA POWER API in the sidebar | Same functionality, moved into the **⚙️ popover** in the top bar (+ a "Back to bundled regions" button) |
| "Active Dataset Context" block in the sidebar | **Context strip** under the navbar (region · coordinates · elevation · years · source) |
| Top banner `nasa-header` on every page | Removed (replaced by the navbar + page headers) |

---

## 3. Rainfall (`PRECTOTCORR`)

| From | Into |
|---|---|
| Analysed like temperature: monthly/annual **mean of mm/day** | Converted to **totals**: mm/day × days in month → mm/month → mm/year (`monthly_rainfall_totals`, `annual_rainfall_totals`) |
| Calendar year only | + **Wet season Oct → Mar** (labelled by end year, incomplete first season dropped) |
| — | + **Number of wet months** (≥ 1 mm) per year |
| — | + Monthly climatology ("when does it rain?") and **% of 1981–2010 normal** |
| Trend = Mann-Kendall only | Trend significant **only if Mann-Kendall AND Hamed-Rao agree**; Sen's slope in mm/decade and % of mean/decade |
| No check for zero-inflated data | **Reliability flags** when > 50 % of values are zero or tied, or totals are tiny (Aswan) |
| No break detection | **Pettitt (1979) change-point test** (`pettitt_change_point_test`) — detects a sudden jump that Mann-Kendall cannot see |

**Results (1981–2023)**

| Region | Mean | Gradual trend (MK + HR) | Abrupt shift (Pettitt) |
|---|---|---|---|
| Greater Cairo | 62 mm/yr | **No** (MK p = 0.77) | **Yes, from 2015** (p = 0.0067): 40 → 146 mm/yr |
| Alexandria | 108 mm/yr | **No** (MK p = 0.08, HR p = 0.24) | **Yes, from 2015** (p ≈ 0.006) |
| Aswan | 1.6 mm/yr | **No** (low reliability) | **No** |

> ⚠️ Pettitt says **when**, not **why**. The 2015 jump may be real (e.g. storms of late 2016 and March 2020) or an artefact of changing MERRA-2 inputs. The app states this explicitly. Recommended next step: confirm with an independent dataset (NASA GPM IMERG).

---

## 4. `T2M_MAX` / `T2M_MIN` meaning (correction)

| From | Into |
|---|---|
| `src/data_loader.py`: *"MERRA-2 monthly mean of daily maximum temperature"* | *"maximum 2-meter temperature reached within the month (absolute monthly extreme, NOT a mean of daily maxima)"* — same for `T2M_MIN` |
| README: "daily maximum/minimum temperatures" | "monthly extreme maximum/minimum temperatures (hottest/coldest value reached within each month)" |

**Proof**: in the NASA POWER CSV the `ANN` value of `T2M_MAX` equals the **maximum** of the 12 monthly values (e.g. Cairo 2000: Jan = 24.7 °C, ANN = 44.87 °C = July). A mean of daily maxima for Cairo in January would be ~19–20 °C. Covered by `test_nasa_power_t2m_max_is_monthly_extreme_not_mean`.

👉 **When presenting**: say *"the hottest value reached in the month"*, not *"average daily maximum"*.

---

## 5. Temperature extremes (new)

| From | Into |
|---|---|
| `T2M_MAX` / `T2M_MIN` only selectable in generic pages | New module `src/extremes_analysis.py` + 🔥 **Extremes** page |

Five yearly indices per region, each with Mann-Kendall + Hamed-Rao, slope/decade with 95 % CI and Pettitt:

1. **Hottest value of the year** (max of `T2M_MAX`)
2. **Coldest value of the year** (min of `T2M_MIN`)
3. **Unusually hot months**: `T2M_MAX` above the 1981–2010 **90th percentile of that calendar month**
4. **Unusually cold months**: `T2M_MIN` below the 1981–2010 **10th percentile of that calendar month**
5. **Monthly extreme range** (`T2M_MAX − T2M_MIN`)

Plus a per-calendar-month breakdown with Benjamini-Hochberg FDR, decade counts, record years and a regional table.

> Method note: for indices 3–4 (small integer counts with many ties) Sen's median slope collapses to 0, so the **magnitude** is reported with OLS; **significance** still comes from Mann-Kendall + Hamed-Rao.

**Results, Greater Cairo**: hottest value of the year **+0.33 °C/decade** (significant) · unusually hot months **↑ significant** (0.4/yr in 1981–90 → 1.3/yr in 2011–23, ×3.3) · unusually cold months **↓ significant** · coldest value of the year: not significant · extreme range: not significant.  
All three regions: hottest value ↑ and hot months ↑ significant; Aswan has the fastest rise in hot months (+0.81 months/yr per decade).

---

## 6. Mean vs hottest vs coldest (new, Regions page)

| From | Into |
|---|---|
| Regional page compared only one variable at a time | New section comparing warming of `T2M`, `T2M_MAX`, `T2M_MIN` side by side with 95 % CI error bars (`compare_temperature_extremes` in `src/trend_analysis.py`) |

| Region | T2M | T2M_MAX | T2M_MIN | (°C/decade, all significant) |
|---|---|---|---|---|
| Greater Cairo | +0.29 | +0.34 | +0.37 | |
| Alexandria | +0.26 | +0.44 | +0.39 | |
| Aswan | +0.43 | +0.50 | +0.43 | |

Confidence intervals overlap, so the app does **not** claim extremes warm faster than the mean.

---

## 7. Home page

| From | Into |
|---|---|
| Long text: mission, 4 questions, Cairo benchmark | **Hero**: "Is ⟨region⟩ heating up?" + live YES/NO answer |
| Hard-coded benchmark numbers | **4 KPI cards computed live** from the analysis for the selected region: net warming (+1.23 °C), unusually-hot-month multiplier (×3.3), rainfall jump (2015), significance (p < 0.001) |
| — | Annual temperature chart with Sen's slope, **"Three findings"** with buttons to Temperature / Extremes / Rainfall, and the 4 challenge questions as cards |
| — | Old Home content preserved in the *About the mission* tab |

---

## 8. Visual design

| From | Into |
|---|---|
| Default Streamlit theme (followed OS light/dark) | `.streamlit/config.toml`: fixed **dark theme** (navy background, NASA red primary), minimal toolbar |
| White Plotly charts on a dark page | One dark chart style in `_apply_scientific_theme` (card background, light text, subtle grid, dark tooltips); brighter palette for contrast |
| White `q-box` / `stat-card` | Dark cards with accent borders; `st.metric` styled the same |
| Sidebar navigation | Sticky glass **top navbar** with pill tabs; sidebar hidden |
| Plain section text for the 4 questions | `render_four_questions()` card grid |
| Metric values cut off with "…" | Values wrap |

---

## 9. Responsive

| Width | Layout |
|---|---|
| ≥ 1180 px (laptop) | Navbar in one row, 4 KPI cards per row |
| 960–1180 px | Smaller pills, brand tagline hidden |
| 640–960 px (tablet) | Navbar on two rows (brand · region · ⚙️ / scrollable pills), KPI cards 2 × 2, navbar not sticky |
| < 640 px (phone) | Compact hero/headers, KPI cards 2 × 2, tabs scroll sideways, chart titles smaller, Plotly toolbar hidden |
| < 360 px | KPI cards one per row |

---

## 10. Code structure (for developers)

| From | Into |
|---|---|
| `app.py`: one long `if page == … elif …` chain | Each original page is a function: `page_about`, `page_data_explorer`, `page_time_series`, `page_trend_engine`, `page_regional`, `page_anomalies`, `page_methodology` (**bodies unchanged**) |
| — | New page functions: `page_home`, `page_temperature`, `page_rainfall`, `page_extremes` |
| Page chosen by sidebar radio | `NAV_PAGES` dict + top-bar radio (`key="nav"`) → router at the bottom of `app.py` |
| Analysis recomputed on every rerun | `cached_rainfall`, `cached_extremes`, `cached_temperature_summary`, … (`st.cache_data`, per region) |
| — | Helpers: `kpi_card`, `page_header`, `render_four_questions`, `go_to` |

**To add a new page**: write `def page_xxx() -> None:` and add `"🔤 Name": page_xxx` to `NAV_PAGES`.

### Files

| Status | File | What |
|---|---|---|
| **new** | `src/rainfall_analysis.py` | Rainfall totals, wet season, wet months, climatology, % of normal, trend tests with reliability flags, **Pettitt test**, regional table |
| **new** | `src/extremes_analysis.py` | 5 temperature-extreme indices, percentile thresholds, trends, decade counts, regional table |
| **new** | `tests/test_rainfall_analysis.py` | 11 tests (unit conversion, leap years, wet season, Pettitt on synthetic + real data, mean-vs-extremes comparison, …) |
| **new** | `tests/test_extremes_analysis.py` | 5 tests (incl. proof that `T2M_MAX` is a monthly extreme) |
| **new** | `.streamlit/config.toml` | Dark theme, minimal toolbar |
| **new** | `CHANGES_FROM_INTO.md` | This file |
| changed | `app.py` | Layout, navbar, new pages, CSS (analysis calls unchanged) |
| changed | `src/visualization.py` | Dark chart theme + 7 new chart functions (rainfall, extremes, comparison) |
| changed | `src/trend_analysis.py` | **Added** `compare_temperature_extremes` (nothing existing modified) |
| changed | `src/data_loader.py` | `T2M_MAX` / `T2M_MIN` descriptions corrected (text only) |
| changed | `README.md` | Same description fix (1 line) |
| changed | `.gitignore` | `.streamlit/*` ignored except `config.toml` (secrets stay ignored) |

---

## 11. How to review / run

```bash
git fetch origin
git checkout feature/rainfall-analysis
pip install -r requirements.txt      # no new dependencies
pytest -q                            # expect 36 passed
streamlit run app.py                 # http://localhost:8501
```

---

## 12. Known notes / not done

- **README** still describes the old 7 sidebar pages; it does not yet mention the Rainfall / Extremes pages or the findings above.
- Streamlit prints deprecation warnings for `use_container_width` (Streamlit wants `width="stretch"`); it still works in the pinned version, but should be replaced before upgrading Streamlit.
- `scripts/reproducible_scientific_audit.py` crashes on the Windows console because of the `✓` character; run it with `PYTHONIOENCODING=utf-8` (it passes).
- The rainfall jump in 2015 is **not yet validated** against an independent dataset (GPM IMERG).
- Not started: rainfall anomaly detection (e.g. SPI), forecasting.
