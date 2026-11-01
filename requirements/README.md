# Which file do I install?

| Where | Command |
|---|---|
| **Mac (laptop)** | `uv pip sync requirements/panel-macos.txt` |
| **Goethe (cluster) / any Linux** | `uv pip sync requirements/panel-linux.txt` |

Run it with the project's venv active (setup: `panel_pipeline/README.md`, A1 for the
Mac, B2 for Goethe). `sync` installs exactly the list and removes anything else.

## The files

| File | What it is | Edit by hand? |
|---|---|---|
| `panel-macos.txt` | every package with one exact version, for macOS | no, generated |
| `panel-linux.txt` | the same for Linux; differs only by `tensorflow-macos` and `appnope` | no, generated |
| `panel.in` | the short list of packages our code imports directly | **yes** |
| `constraints-amw-2026-09-18.txt` | exact versions to reuse from the Sept 2026 Mac venv; read only when regenerating | no |

To add or change a package: edit `panel.in`, then regenerate both lock files and commit them:

```bash
uv pip compile requirements/panel.in -c requirements/constraints-amw-2026-09-18.txt \
    --python-version 3.11 --python-platform x86_64-manylinux_2_28 -o requirements/panel-linux.txt
uv pip compile requirements/panel.in -c requirements/constraints-amw-2026-09-18.txt \
    --python-version 3.11 --python-platform aarch64-apple-darwin -o requirements/panel-macos.txt
```

`requirements-lock_2026-09-18.txt` in the repository root is a snapshot of the Mac venv
from Sept 2026, including SAM2 and torch. Nothing installs from it. It is kept as the
source of the constraints file and as the record of a working SAM2 setup, until SAM2
gets its own list here.
