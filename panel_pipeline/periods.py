from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date



FREQUENCIES = ("monthly", "quarterly")
_MONTH_TAG = re.compile(r"^(\d{4})-(\d{2})$")     # 2018-01
_QUARTER_TAG = re.compile(r"^(\d{4})Q([1-4])$")   # 2018Q1


@dataclass(frozen=True, order=True)
class Period:
    year: int
    first_month: int
    n_months: int

    def __post_init__(self):
        if self.n_months not in (1, 3):
            raise ValueError(f"n_months must be 1 or 3, got {self.n_months}")
        if not 1 <= self.first_month <= 12:
            raise ValueError(f"first_month must be 1..12, got {self.first_month}")
        if self.n_months == 3 and self.first_month not in (1, 4, 7, 10):
            raise ValueError(f"a quarter starts in month 1, 4, 7 or 10, "
                             f"got {self.first_month}")
        
    @property
    def frequency(self) -> str:
        return "monthly" if self.n_months == 1 else "quarterly"

    @property
    def tag(self) -> str:
        if self.n_months == 1:
            return f"{self.year}-{self.first_month:02d}"
        return f"{self.year}Q{(self.first_month - 1) // 3 + 1}"

    @property
    def month_index(self) -> int:
        return self.year * 12 + self.first_month - 1

    @property
    def start(self) -> date:
        return date(self.year, self.first_month, 1)

    @property # I am doing it a bit more "complex" here so that years seamlessly roll over
    def end(self) -> date:
        nxt = self.month_index + self.n_months
        return date(nxt // 12, nxt % 12 + 1, 1) # also keeping the AMW logic of including the first day of the next month as the "end" eventhough its not scored

    @property
    def date_span(self) -> str:
        return f"{self.start.isoformat()}_{self.end.isoformat()}"

    def months(self) -> list[Period]:
        return [Period(self.year, self.first_month + i, 1)
                for i in range(self.n_months)]

    def quarter(self) -> Period:
        return Period(self.year, (self.first_month - 1) // 3 * 3 + 1, 3)

    @classmethod
    def parse(cls, tag: str) -> Period:
        tag = str(tag).strip()
        m = _MONTH_TAG.match(tag)
        if m:
            return cls(int(m.group(1)), int(m.group(2)), 1)
        q = _QUARTER_TAG.match(tag)
        if q:
            return cls(int(q.group(1)), (int(q.group(2)) - 1) * 3 + 1, 3)
        raise ValueError(f"unrecognised period {tag!r}; "
                         f"expected a month (2018-01) or a quarter (2018Q1)")


def period_range(first_month: str, last_month: str,
                 frequency: str) -> list[Period]:
    if frequency not in FREQUENCIES:
        raise ValueError(f"frequency must be one of {FREQUENCIES}, got {frequency!r}")
    first, last = Period.parse(first_month), Period.parse(last_month)
    if first.n_months != 1 or last.n_months != 1:
        raise ValueError("first_month and last_month must be months, e.g. 2018-01")
    if last < first:
        raise ValueError(f"last_month {last.tag} is before first_month {first.tag}")

    step = 1 if frequency == "monthly" else 3
    if step == 3 and (first.first_month - 1) % 3 != 0:
        raise ValueError(f"{first.tag} does not start a quarter")
    if step == 3 and last.first_month % 3 != 0:
        raise ValueError(f"{last.tag} does not end a quarter; "
                         f"the last quarter would be incomplete")

    out = []
    i = first.month_index
    while i <= last.month_index:
        out.append(Period(i // 12, i % 12 + 1, step))
        i += step
    return out