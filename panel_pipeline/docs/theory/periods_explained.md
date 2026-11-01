# How `periods.py` works, and how it fits into the pipeline

## 1. It's a toolbox, not a program

You're right: nothing happens when you run `periods.py` on its own. That's on
purpose. It is a **module**, a file that *defines* tools for other files to use.
It's like a sheet of formulas: on its own it computes nothing, but every script
that needs dates reaches for it.

Other files **import** it ("borrow its tools") and use them:

```python
# inside config.py (next block), or any other script
from periods import Period, period_range

months = period_range("2018-01", "2026-09", "monthly")   # 105 Period objects
first = months[0]
print(first.tag)          # 2018-01
print(first.date_span)    # 2018-01-01_2018-02-01
```

Where it gets used later:

```
config.yaml ──► config.py ──► period_range(...) ──► list of Periods
                                                         │
          ┌──────────────────────────────┬───────────────┴──────────────┐
          ▼                              ▼                              ▼
   detect.py                      folder names                   panel columns
   p.start, p.end ──► Earth       <run>/monthly/2018-01/...      period = "2018-01",
   Engine dates                                                  quarter = "2018Q1"
```

One definition of "what is 2018Q1 and which dates does it cover", used in
every place that needs it. If dates were typed in several places, sooner or
later two of them would disagree.

**Try it yourself.** From the `panel_pipeline` folder, in a terminal:

```
python -c "from periods import period_range; print([p.tag for p in period_range('2018-01','2018-06','quarterly')])"
```

It should print `['2018Q1', '2018Q2']`.

## 2. What a class is

A **class** is a blueprint for a kind of thing. An **object** (or *instance*) is
one thing made from the blueprint.

- `Period` is the blueprint: "a period has a year, a first month and a length,
  and here is how to get its dates".
- `Period(2018, 3, 1)` is one object made from it: March 2018.
- `Period(2025, 7, 3)` is another: the third quarter of 2025.

A class bundles **data** (what each period knows) with **behaviour** (what you
can ask it). Without a class you would carry three loose numbers around and
re-do the date arithmetic wherever you needed it.

## 3. The three fields: `year`, `first_month`, `n_months`

These are the only facts each period *stores*. Everything else is worked out
from them.

| Period | `year` | `first_month` | `n_months` |
|---|---|---|---|
| January 2018 | 2018 | 1 | 1 |
| March 2018 | 2018 | 3 | 1 |
| 2018Q1 | 2018 | 1 | 3 |
| 2025Q3 | 2025 | 7 | 3 |

Why "first month + number of months" rather than "month or quarter number"?
Because then a month and a quarter are the same kind of thing ("n months
starting at m"). One date formula works for both, which means one place to get
right instead of two.

## 4. `@dataclass`: what it does

A line starting with `@` (a *decorator*) changes the thing defined right after
it. `@dataclass` looks at the three fields and writes the boring code for you.
Without it, you would have to write all of this by hand:

```python
class Period:
    def __init__(self, year, first_month, n_months):   # how to create one
        self.year = year
        self.first_month = first_month
        self.n_months = n_months
    def __repr__(self):                                  # how it prints
        return f"Period(year={self.year}, ...)"
    def __eq__(self, other):                             # when two are equal
        return (self.year, self.first_month, self.n_months) == (...)
    def __lt__(self, other):                             # how to sort
        ...
```

- `frozen=True`: once made, a period can't be changed (`p.year = 2019` gives an
  error). A period that changed its dates halfway through a run would be a
  nasty bug. Frozen objects can also be used as dictionary keys.
- `order=True`: periods can be sorted and compared (`<`), field by field.
- `__post_init__`: a hook the dataclass calls right after creating a period. We
  use it to refuse impossible ones, like month 13.

## 5. Field vs property vs method vs classmethod

This is the part that looks complicated. All four are ways to get something
from a period:

| Kind | Example | Brackets? | What it is |
|---|---|---|---|
| **field** | `p.year` → `2018` | no | A stored fact |
| **property** | `p.tag` → `"2018-03"` | no | Computed from the fields each time you ask, but it *looks* like a stored fact |
| **method** | `p.months()` → list of months | yes | An action you ask the period to do (it could take inputs) |
| **classmethod** | `Period.parse("2018Q1")` | yes | Asked of the **blueprint**, not of one period. Here: "make me a period from this text" |

- **`self`** in the code means "this particular period". In `p.tag`, Python
  passes `p` in as `self`.
- **Why properties instead of storing `tag`, `start`, `end` as fields?** If they
  were stored, they could disagree with `year` and `first_month` (someone builds
  a period with the wrong tag). Computed, they are always right.
- **Two ways to create a period:**
  - `Period(2018, 1, 3)`, the normal way, from numbers.
  - `Period.parse("2018Q1")`, from text. That's what we need when the period
    comes from `config.yaml` or the command line, which are always text.

## 6. Type hints (`: int`, `-> str`, `list[Period]`)

`year: int` and `-> str` are **labels** for humans and for your editor: "this is
a whole number", "this returns text". Python itself doesn't enforce them.
`list[Period]` means "a list of Periods".

**Your red squiggles.** I reproduced them. Without the line
`from __future__ import annotations` at the top, the editor reports
`"Period" is not defined` on `months`, `quarter` and `parse`. Python also
crashes when it loads the file (`NameError`). The reason: inside the class, the
class doesn't exist yet. Python is still building it, so the label
`list[Period]` points at nothing. The `__future__` line tells Python to treat
labels as text and not look them up while building. With it, the editor shows
no errors and the file loads.

**Fix:** keep `from __future__ import annotations` as the first code line of
the file (only the docstring may come before it), and use `list[Period]` with no
other import. If squiggles remain, check that your editor uses the project's
Python 3.11 environment (VS Code: "Python: Select Interpreter").

## 7. Why so complicated?

Honest answer: you could do it with a list of strings and a few date
calculations sprinkled through the code. We use a class because:

1. The **dates exist in one place only**, and are used for Earth Engine, file
   names and panel columns alike.
2. **Mistakes are caught at the start** (a quarter starting in February, a
   partial last quarter), not three days into a run.
3. **Months and quarters are linked** (`months()`, `quarter()`). The panel
   needs that link.
4. **It's their design.** Their `core/periods.py` is also a frozen dataclass
   `Period` with `parse`, `tag` and `date_span`. We extended it to months and
   kept the structure.
