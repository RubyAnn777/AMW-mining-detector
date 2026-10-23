# How a script actually runs

This covers what happens when you type `python panel_pipeline/plan.py --config x.yaml`: who reads which part of that line, why plan.py looks like "only definitions", and where the `--` words are defined. Line numbers are for the files on `claude/project-thread-ls8p9r`.

## 1. Three programs take part, one after the other

When you press Enter on

```
python panel_pipeline/plan.py --config "my file.yaml"
```

three things happen, in this order.

1. **The shell** (the Terminal program, zsh on your Mac) splits the line into words at the spaces. Quotes keep a space inside one word. It doesn't know what any word means. It finds the program named by the first word, `python` (the one from your active environment), and starts it, handing over the other words.
2. **Python** takes the first word it was given as the file to run. It puts the remaining words in a list called `sys.argv`, and it puts the script's folder (`panel_pipeline/`) first on its import search path. That is why plan.py can write `from config import ...` and Python finds `panel_pipeline/config.py`.
3. **The script** runs, top to bottom. It can read `sys.argv` whenever it likes.

A two-line script shows step 2:

```python
import sys
print(sys.argv)
```

```
$ python show_argv.py --config "my file.yaml" --limit 1
['show_argv.py', '--config', 'my file.yaml', '--limit', '1']
```

The `--` has no special meaning to the shell or to Python. It is only a naming convention. `--config` is just a word in a list, and the quotes are gone because the shell used them to keep `my file.yaml` as one word.

## 2. Python runs every line, but `def` doesn't run the function

Python executes a file from the first line to the last. What looks like "only definitions" is also being executed. Executing a definition just doesn't do the work inside it:

| Line | What executing it does |
|---|---|
| `import pandas as pd` | Loads pandas, then puts it under the name `pd`. |
| `from config import load_config` | Runs config.py's top level once, then copies the name `load_config` into this file. |
| `CHIP_PX = 48` | Stores a value. |
| `def plan(cfg): ...` | Builds a function and stores it under the name `plan`. **The body is not run.** It is a recipe, written down for later. |
| `class Region: ...` | Builds a class (a blueprint for objects) and stores it under `Region`. |
| `plan(cfg)` | Runs the recipe now. This is a *call*. |

So most of plan.py is writing recipes into a recipe book. The cooking happens only where a function is *called*, and the first call happens at the bottom.

## 3. The bottom block is the program

The last block of plan.py (lines 156–177) is the part that does the work:

```python
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Plan a run: folder, settings copy, tiles, chips.")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG),
                    help="path to a config.yaml (default: the one next to this file)")
    args = ap.parse_args()
    try:
        cfg = load_config(args.config)
    except (ValueError, OSError, yaml.YAMLError) as err:
        raise SystemExit(f"CONFIG ERROR: {err}")
    print(describe(cfg) + "\n", flush=True)

    run_dir, record = plan(cfg)
    ...
    print(f"\nplanned  ->  {run_dir}")
```

**`__name__`** is a variable Python sets in every file before running it:
- in the file you ran (`python panel_pipeline/plan.py`), it is `"__main__"`;
- in a file that was imported (`from config import ...`), it is the file's name, `"config"`.

So `if __name__ == "__main__":` means "only if this file was started directly, not imported".

The whole run of `python panel_pipeline/plan.py`, in order:

1. **Lines 1–12**, the docstring. It is stored as the file's description.
2. **Lines 13–30**, the imports. Each of our own modules has its top level run once:
   - config.py's sets `REPO` and `DEFAULT_CONFIG` (lines 15–17), then builds its classes and functions. Its own `if __name__ == "__main__":` block (line 288) is skipped, because there `__name__` is `"config"`.
   - grid.py adds AMW's `core/` folder to the import search path (line 18), so AMW's code can be imported later.
3. **Lines 35, 69 and 119**: `plan_tiles`, `write_chips` and `plan` are stored, not run.
4. **Line 156**: `__name__` is `"__main__"`, so the block runs:
   - argparse reads your `--` words (section 5);
   - `load_config` checks config.yaml;
   - `describe` prints the summary;
   - `plan(cfg)` does the work. It calls `plan_tiles`, which calls grid.py's functions, then `write_chips`.
5. **The end of the file**: Python stops, with exit code 0 ("all fine").

The program flow is in that bottom block. When you read a new script, start there, then follow each call up to its definition.

**`raise SystemExit("...")`**, used throughout our scripts, stops the program at that point. It prints the message and exits with code 1 ("something went wrong"). A cluster scheduler reads that code to know a job failed.

## 4. Why write scripts this way: for and against

**For:**
- **One file can be both a tool and a library.** `python panel_pipeline/lib/config.py` checks your config and prints it, while plan.py and detect.py import `load_config` from the same file. Without the `if` line, every import of config.py would also run that check-and-print.
- **Tests can use the pieces.** My tests import detect.py's functions and call `score_tile` on fake imagery without running the whole command.
- **Named steps.** The bottom block reads like a table of contents: check config, describe, plan. Each step can be read, fixed and tested on its own.

**Against:**
- **You read in jumps.** The story starts at the bottom and calls upward. A notebook-style script that runs straight down is easier to read once.
- **The block's variables are global.** `cfg` and `args` in that block are visible everywhere in the file, which can hide mistakes. The common fix is to write `def main(): ...` and end the file with `if __name__ == "__main__": main()`. That costs one more jump.

AMW uses the same plain-block form (`core/inference_pipeline.py:99`), so our scripts follow it.

## 5. The `--` words: argparse reads them

Since Python only hands the words over as a list, each script has to interpret them itself. Our scripts, and AMW's, use **argparse**, which comes with Python. The `--` options are defined in the script, in the lines that call `add_argument`. That's the answer to "where is this defined".

plan.py, line by line:

```python
ap = argparse.ArgumentParser(description="Plan a run: folder, settings copy, tiles, chips.")
```
This makes an empty parser, an object that will read `sys.argv`. `description` is shown by `--help`.

```python
ap.add_argument("--config", default=str(DEFAULT_CONFIG),
                help="path to a config.yaml (default: the one next to this file)")
```
This declares one option, `--config`.
- **`default`**: what it is when you leave it out, here `panel_pipeline/config.yaml`.
- **`help`**: its line in `--help`.

```python
args = ap.parse_args()
```
This reads `sys.argv[1:]` and matches each word against the declared options. Three outcomes:
- **A match**: the result is an object with one attribute per option, so `args.config` is `"x.yaml"` or the default.
- **An unknown word or a bad value**: it prints the problem and stops with exit code 2.
- **`--help` or `-h`**: it prints the help and stops. `--help` is added automatically.

You can see both in real output:

```
$ python panel_pipeline/plan.py --help
usage: plan.py [-h] [--config CONFIG]

Plan a run: folder, settings copy, tiles, chips.

options:
  -h, --help       show this help message and exit
  --config CONFIG  path to a config.yaml (default: the one next to this file)

$ python panel_pipeline/plan.py --confg x.yaml
usage: plan.py [-h] [--config CONFIG]
plan.py: error: unrecognized arguments: --confg x.yaml
```

detect.py (lines 306–311) uses more of the options `add_argument` takes:

```python
ap.add_argument("--run", required=True, help="the run folder made by plan.py")
ap.add_argument("--frequency", required=True, choices=("monthly", "quarterly"))
ap.add_argument("--subregion", required=True, type=int)
ap.add_argument("--from", dest="first", help="first period, e.g. 2018-01 or 2018Q1")
ap.add_argument("--to", dest="last", help="last period")
ap.add_argument("--limit", type=int, help="score only the first LIMIT tiles (a test)")
```

- **`required=True`**: leaving it out is an error. Options without it are optional, and are `None` when left out unless they have a `default`.
- **`choices`**: any other value is an error, so `--frequency montly` fails at once.
- **`type=int`**: the word `"6"` becomes the number `6`, and `--subregion six` is an error. Everything in `sys.argv` is text, so this conversion matters.
- **`dest="first"`**: the attribute is called `args.first`. `args.from` would be a syntax error, because `from` is a Python keyword.

You can write `--limit 1` or `--limit=1`, and options can come in any order.

**Named options versus positional ones.** argparse also allows options without dashes, matched by position, as in `python detect.py tests_output/pilot monthly 6`.
- For positional: it is shorter to type.
- Against positional: you have to remember the order, and the command doesn't say what `6` is.

AMW names every option (`--model`, `--start_date`, ...), and so do we. A command in a cluster job file then explains itself.

## 6. Try it yourself

Save this as `greet.py` anywhere outside the repository:

```python
"""Say hello: a tiny script to see how argparse works."""
import argparse


def greet(name: str, times: int) -> None:
    for _ in range(times):
        print(f"hello, {name}")


print("this line runs on import too; __name__ is", __name__)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Say hello.")
    ap.add_argument("--name", required=True, help="who to greet")
    ap.add_argument("--times", type=int, default=1, help="how often (default: 1)")
    args = ap.parse_args()
    print(args)
    greet(args.name, args.times)
```

This is what I got for each command:

```
$ python greet.py --name Ruby --times 2
this line runs on import too; __name__ is __main__
Namespace(name='Ruby', times=2)
hello, Ruby
hello, Ruby

$ python greet.py --times two --name Ruby
this line runs on import too; __name__ is __main__
usage: greet.py [-h] --name NAME [--times TIMES]
greet.py: error: argument --times: invalid int value: 'two'

$ python greet.py
this line runs on import too; __name__ is __main__
usage: greet.py [-h] --name NAME [--times TIMES]
greet.py: error: the following arguments are required: --name

$ python -c "import greet; greet.greet('from another file', 1)"
this line runs on import too; __name__ is greet
hello, from another file
```

The last command imports greet.py instead of running it:
- the `print` outside any function still ran, because every top-level line runs on import;
- `__name__` was `greet`, so the bottom block was skipped;
- `greet` was called by hand.

That is how plan.py uses config.py.

## 7. The login check, piece by piece

The Earth Engine check in the thread is a `python -c` command:

```
python -c "import sys; sys.path += ['panel_pipeline', 'core']; import ee, gee, config; gee.EE_PROJECT = config.load_config().ee_project; gee._ensure_earth_engine_initialized(); print('Earth Engine works:', ee.ImageCollection('COPERNICUS/S2_HARMONIZED').filterBounds(ee.Geometry.Point(-69.9, -12.9)).filterDate('2024-07-01', '2024-08-01').size().getInfo(), 'images of La Pampa in July 2024')"
```

- **`-c`** is one of Python's own options, read by Python before any script. It means "run this text as the program instead of a file". Semicolons separate statements on one line.
- **`sys.path += [...]`** adds our folder and AMW's `core/` to the import search path. A file run directly gets its own folder added automatically, but `-c` text has no file, so Python adds only the folder you are in. That's also why the command must be run from the repository folder: the two added paths are relative to it. From anywhere else you get `No module named 'gee'`.
- **`gee.EE_PROJECT = config.load_config().ee_project`** reads `earth_engine.project` from config.yaml and gives it to AMW's code, exactly as detect.py does.
- **`gee._ensure_earth_engine_initialized()`** is AMW's own login (`core/gee.py:47`): same endpoint, same credentials.
- **The last part** asks Earth Engine to count the Sentinel-2 images (the collection AMW composites) that cover a point in La Pampa in July 2024. `.getInfo()` is the moment the request is actually sent. Before it, the chain only describes the request. Only the count comes back, no imagery.
