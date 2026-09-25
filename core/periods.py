"""Inference periods: what one is, and what dates it spans.

A leaf module on purpose -- no third-party imports. Everything downstream needs
to know what dates a period spans, and asking that question should not load the
geospatial stack.

`date_span` is the single source of every calendar date in the pipeline. Emitted
inference commands, SAM2 date arguments and detection filenames all derive from
it, so the quarter boundaries below are defined once and nowhere else.
"""
# >> FILE OVERVIEW
# >> Defines a 'period' (a year like 2024, or a quarter like Q125 = Jan-Mar 2025) and
# >> turns it into the start/end dates used for EE requests and output filenames.
# >> Pipeline position: config layer used by scripts/pipeline.py, persistence.py and
# >> sam2_persistence.py. Upstream file (Amazon Mining Watch), byte-identical.
# >> Gotcha: end dates here are the last calendar day (03-31, 12-31), but EE treats
# >> the end date as EXCLUSIVE, so the last day of every period is never imaged.
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, Tuple

# >> Quarter number -> (start MM-DD, end MM-DD). The end is the last day of the
# >> quarter; as an EE end date (exclusive) that day is dropped.
QUARTER_SPANS = {
    1: ("01-01", "03-31"),
    2: ("04-01", "06-30"),
    3: ("07-01", "09-30"),
    4: ("10-01", "12-31"),
}
# >> Regex for quarter tags: 'Q' + quarter 1-4 + 2-digit year, e.g. Q125 = Q1 2025.
QUARTER_TAG_RE = re.compile(r"^Q([1-4])(\d{2})$")


# --------------------------------------------------------------------------
# vocabulary
# --------------------------------------------------------------------------

# >> One period: a whole year (quarter=None) or one quarter of a year.
# >> frozen=True -> immutable and hashable; order=True -> sortable with < and sorted().
# >> Sorting compares sort_key first: (year, end month, 1 if annual else 0), so
# >> Q1..Q4 2025 sort before the annual 2025 period (a year sorts after its quarters).
# >> Used by: scripts/pipeline.py, persistence.py, sam2_persistence.py.
@dataclass(frozen=True, order=True)
class Period:
    """One inference period: a calendar year, or a quarter within one.

    Ordering is chronological by end date, with the annual period sorting after
    the quarters it contains -- an annual mosaic is only assembled once its year
    is complete.
    """
    # >> Computed in __post_init__, not passed by callers; listed first so it drives
    # >> order.
    sort_key: Tuple[int, int, int] = field(init=False, repr=False)
    year: int
    quarter: Optional[int] = None

    # >> Fill sort_key after creation (object.__setattr__ because the class is frozen).
    def __post_init__(self):
        end_month = 12 if self.quarter is None else self.quarter * 3
        object.__setattr__(
            self, "sort_key",
            (self.year, end_month, 1 if self.quarter is None else 0))

    # >> True for a whole-year period.
    @property
    def is_annual(self) -> bool:
        return self.quarter is None

    # >> Short label: '2024' for a year, 'Q125' for Q1 2025 (year % 100 = last 2 digits).
    # >> Gotcha: years 2000-2009 would give a 1-digit year (Q15) that parse() rejects.
    @property
    def tag(self) -> str:
        return str(self.year) if self.is_annual else f"Q{self.quarter}{self.year % 100}"

    # >> Date fragment for filenames and EE requests, e.g. '2024-01-01_2024-12-31' or
    # >> '2025-01-01_2025-03-31'. Callers split on '_' to get start and end.
    # >> Gotcha: used as an EE end date this is exclusive, so 12-31 / 03-31 are skipped.
    @property
    def date_span(self) -> str:
        """The ``{start}_{end}`` fragment used in detection filenames."""
        if self.is_annual:
            return f"{self.year}-01-01_{self.year}-12-31"
        start, end = QUARTER_SPANS[self.quarter]
        return f"{self.year}-{start}_{self.year}-{end}"

    # >> Turn a text tag into a Period: '2024' -> year 2024; 'Q125' -> Q1 of 2025.
    # >> Raises ValueError for anything else. Assumes 21st-century years (2000 + yy).
    @classmethod
    def parse(cls, tag: str) -> "Period":
        tag = str(tag).strip()
        m = QUARTER_TAG_RE.match(tag.upper())
        if m:
            quarter, yy = int(m.group(1)), int(m.group(2))
            return cls(year=2000 + yy, quarter=quarter)
        if re.fullmatch(r"\d{4}", tag):
            return cls(year=int(tag))
        raise ValueError(
            f"Unrecognised period {tag!r}; expected a year (2024) or a quarter tag (Q125)")


# >> Integer code for a period: 2024 -> 2024, Q125 -> 20251 (year*10 + quarter).
# >> Used as the pixel value of 'onset' rasters (uint16 max 65535, so it fits).
# >> Gotcha: every annual code (4 digits) is smaller than every quarter code (5
# >> digits), so 2026 < 20251: codes are NOT chronological when years and quarters mix.
# >> Note: sam2_persistence.py defines its own copy of this function.
def encode_period(tag: str) -> int:
    """Sortable uint16 code for a period tag: ``2024`` -> 2024, ``Q125`` -> 20251.

    Chronological under plain comparison (2024 < 20251 < 20252 < 20261), so any
    cumulative is ``0 < onset <= code``. Years and quarters share one raster,
    which is what makes the year-boundary supersede structural: quarters are only
    emitted for years the rule cannot yet resolve, so recomputing after the next
    annual lands replaces them with the confirmed year automatically.
    """
    period = Period.parse(tag)
    return period.year if period.is_annual else period.year * 10 + period.quarter
