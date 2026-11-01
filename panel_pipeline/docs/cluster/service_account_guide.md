# Earth Engine service account: full guide

**What it is.** A service account is a Google account for a program instead of a person.
It belongs to the Cloud project `deforestation-and-mining`, gets only the permissions
you grant it, and logs in with a **key file** (a JSON file with a private key) instead
of a password and browser. Our jobs on Goethe use it to call Earth Engine, so the run
doesn't depend on your personal login and nothing needs a browser.

**Why not just your own login.** It works too (`earthengine authenticate --auth_mode=notebook`
on the login node), but it is tied to your Google account and can be revoked by a
password change or a security check. For a run that lasts days, the robot account is
the standard choice.

**The one rule.** The key file is a password. Anyone holding it can spend the project's
Earth Engine quota. It never goes into git, chat, email, or a shared folder.

Source: Google's own page, [Service Accounts](https://developers.google.com/earth-engine/service_account),
and [Control access to resources](https://developers.google.com/earth-engine/cloud/access-control).

---

## Part A. In the Google Cloud console (browser, once)

You need to be an Owner of the project, or someone who is must do steps A2–A4.

**A1. Check the project is ready for Earth Engine.**
Open <https://console.cloud.google.com>, choose the project `deforestation-and-mining`
in the project picker at the top. Go to *APIs & Services → Enabled APIs*. "Google Earth
Engine API" must be in the list. It should be, since your login test already worked
with this project.

**A2. Create the service account.**
*IAM & Admin → Service Accounts → + Create service account.*
- Name: `amw-panel-runner` (any name; this one says what it's for).
- The console makes the email from it:
  `amw-panel-runner@deforestation-and-mining.iam.gserviceaccount.com`. Write it down.
- Click *Create and continue*.

**A3. Give it permissions (the "roles").**
On the next screen, add two roles:
- **Earth Engine Resource Viewer** (`roles/earthengine.viewer`): lets it read
  Earth Engine data and run computations like our `computePixels`. This is the only
  Earth Engine permission we need; we never write assets or export.
- **Service Usage Consumer** (`roles/serviceusage.serviceUsageConsumer`): lets it
  bill its requests to the project. Google says this is needed "depending on your
  project configuration", so add it to avoid a confusing error later.

Click *Done*. Don't give it Owner or Editor: if the key ever leaks, the damage is limited
to reading Earth Engine.

**A4. Create the key file.**
Click the new account → *Keys* tab → *Add key → Create new key → JSON → Create*.
Your browser downloads a file like `deforestation-and-mining-1a2b3c4d.json`.
- This is the only copy Google gives you. If you lose it, delete that key in the same tab
  and create a new one.
- **If the button is greyed out or you get "key creation is disabled by policy"**: the
  Google organisation the project belongs to blocks key files
  (`iam.disableServiceAccountKeyCreation`). Only an organisation admin can make an
  exception for this project. Tell me if that happens; the fallback is your own login.

---

## Part B. Put the key on Goethe (terminal, once)

On your Mac, from the folder the key was downloaded to:

```bash
# 1. a private folder in your backed-up home on Goethe
ssh <user>@goethe.hhlr-gu.de 'mkdir -p ~/.secrets && chmod 700 ~/.secrets'

# 2. copy the key into it with a fixed name
scp deforestation-and-mining-*.json <user>@goethe.hhlr-gu.de:~/.secrets/ee-amw-panel.json

# 3. only you may read it
ssh <user>@goethe.hhlr-gu.de 'chmod 600 ~/.secrets/ee-amw-panel.json'

# 4. delete the downloaded copy on the Mac (or move it to your password manager)
rm deforestation-and-mining-*.json
```

What the commands mean:
- `ssh host 'command'` runs one command on Goethe and comes back.
- `scp source host:destination` copies a file over the same SSH connection.
- `chmod 700` on a folder = only you can open it; `chmod 600` on a file = only you can
  read and write it. Other users on the cluster share the file system, so this matters.
- `~` is your home folder, `/home/<group>/<user>` on Goethe. Home is backed up nightly,
  so the key survives a disk failure. Compute nodes see `/home` too, so every job can
  read it.

---

## Part C. Tell the code to use it

### How AMW's code already handles it
`core/gee.py:69-91`, the function `_ensure_earth_engine_initialized`:
1. reads the environment variable `GOOGLE_APPLICATION_CREDENTIALS`;
2. if it holds a path, checks the file exists (else it stops with an error), builds
   credentials from it with the Earth Engine scope, and passes them to `ee.Initialize`;
3. if it's empty, `ee.Initialize` falls back to your personal login.

An **environment variable** is a named text value that a program inherits from the
shell that started it. `export NAME=value` sets it; Python reads it with
`os.environ.get("NAME")`.

### Where we set it, now: `cluster/goethe_env.sh` (no code change)
The cluster scripts export `GOOGLE_APPLICATION_CREDENTIALS=~/.secrets/ee-amw-panel.json`
in every shell and job (`cluster/goethe_env.sh`, sourced by `cluster/detect_tasks.sbatch`).
AMW's `gee.py` reads it and logs in as the service account; your Mac, without the
variable, keeps using your own login. `_status.json` does not record which login was
used; the config option below would. **Update 23 Oct:** this is what the step-by-step
setup uses, so nothing has to be typed into config.py first.

### Later option: in `config.yaml` (following your settings-in-config rule)
You asked for settings in config sections rather than environment variables
(handoff §2.12). So the path goes in the existing `earth_engine:` section:

```yaml
earth_engine:
  project: "deforestation-and-mining"
  credentials: "~/.secrets/ee-amw-panel.json"   # null on your Mac = use your own login
```

and detect.py copies it into the environment just before logging in, because that is
the only door AMW's `gee.py` has. Two small changes, which you type as usual:

`config.py`: allow the new key in the section and store it (expanded and checked):
```python
ee = _section(raw, "earth_engine", {"project", "credentials"})
...
ee_credentials=(Path(ee["credentials"]).expanduser() if ee["credentials"] else None),
```
plus a check that the file exists when it isn't `None`, and the new field
`ee_credentials: Path | None` on `Config`.

`detect.py`, next to `gee.EE_PROJECT = cfg.ee_project`:
```python
if cfg.ee_credentials is not None:
    # ADDITION: service-account login. gee.py:69 only reads it from the environment.
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(cfg.ee_credentials)
```

**For:** one place for every setting; the run's frozen config copy records which login
the run used (the *path*, never the key itself).
**Against:** the config differs between your Mac (`null`) and Goethe (the path). That's
already true for `output_root`, so no new problem.
**Alternative:** `export GOOGLE_APPLICATION_CREDENTIALS=~/.secrets/ee-amw-panel.json`
in the job script, no code change. Simpler, but invisible in the run's config.

I'll write both changes against the latest detect.py once you're ready for them.

---

## Part D. Test it on Goethe (no imagery, no cost)

On the login node (it's a tiny request, allowed there), with the venv active:

```bash
cd ~/AMW-mining-detector
GOOGLE_APPLICATION_CREDENTIALS=~/.secrets/ee-amw-panel.json python - <<'EOF'
import sys; sys.path.append("core")
import gee, ee
gee.EE_PROJECT = "deforestation-and-mining"
gee._ensure_earth_engine_initialized()
n = (ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
       .filterBounds(ee.Geometry.Point(-69.9, -12.9))
       .filterDate("2024-07-01", "2024-08-01").size().getInfo())
print("login works, La Pampa images in July 2024:", n)
EOF
```

`NAME=value python ...` sets the variable for that one command only. `<<'EOF' ... EOF`
feeds the lines in between to Python as a script.

| What it prints | Meaning | Fix |
|---|---|---|
| `login works, ... 12` (some number) | done | – |
| `FileNotFoundError: GOOGLE_APPLICATION_CREDENTIALS=...` | wrong path | check `ls -l ~/.secrets/` |
| `403 ... Permission denied` / `caller does not have permission` | role missing | redo A3 (role changes can take a few minutes) |
| `... serviceusage.services.use ...` | Service Usage Consumer missing | add it (A3) |
| `Project ... not registered to use Earth Engine` | project/API issue | A1 |
| `invalid_grant` | the node's clock is off | rare on clusters; tell me |

Then run the same test once inside a job on a compute node
(`salloc --partition=test --nodes=1 --time=0:10:00`, then `srun python ...`), because
that's where detect.py will run.

---

## Part E. Housekeeping

- **`.gitignore`**: add `.secrets/` and `deforestation-and-mining-*.json` (the name Google
  gives key files), in case a key ever lands in the repo folder. Git would otherwise
  commit it like any other file.
- **Rotation**: if the key may have leaked (pushed, pasted, laptop lost), delete it under
  *Keys* in the console at once and create a new one. Deleting a key stops it working
  within minutes.
- **Billing/quota**: all requests count against `deforestation-and-mining` whether they come
  from you or the robot account. Nothing about the Earth Engine limits changes.
- **Your Mac** keeps using your own login (`credentials: null`). Same code, same results.
