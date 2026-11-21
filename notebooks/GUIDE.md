# Looking at the mining panel on Goethe

The results stay on the Goethe cluster. You open them in your browser through VS Code.
You don't install git, Python or anything else: the notebooks run from Ruby's copy of the
code and Ruby's Python environment on Goethe.

## Before you start: a Goethe account

You need your own login on Goethe, in the **goldmine** group. The group is what lets you
read Ruby's results; without it the notebooks won't start.

- **No account yet:** ask Ruby. The group leader requests it from the HPC team at the
  university's computing centre (CSC); it can take a few days.
- **Account just created:** log in once as in step 1 below. If the HPC team gave you a
  starting password, Goethe may ask you to change it on that first login.
- **Check the group:** in the terminal of step 1, type `groups`. The list must include
  `goldmine`. If it doesn't, tell Ruby before going further.
- **From outside the university:** if VS Code can't reach `goethe.hhlr-gu.de`, connect to
  the university's VPN first and try again.

## Once: set up VS Code

1. Install VS Code: https://code.visualstudio.com
2. In VS Code, click the Extensions icon in the left bar, search for **Remote - SSH**
   (by Microsoft), and click **Install**.

## Each time

### 1. Connect to Goethe

1. In VS Code press **F1** (Mac: **Cmd-Shift-P**), type `Remote-SSH: Connect to Host`, and
   press Enter.
2. Type your login and press Enter:
   ```
   <your user>@goethe.hhlr-gu.de
   ```
3. If VS Code asks for the platform, choose **Linux**. Type your Goethe password.
   A new window opens. "SSH: goethe.hhlr-gu.de" in its bottom-left corner means it is
   connected. The first time takes a minute. After that, Goethe is in the list when you
   connect.
4. Open a terminal in that window: menu **Terminal → New Terminal**. Everything you type
   there runs on Goethe.

### 2. Start the notebooks

In the terminal:

```bash
sbatch /home/goldmine/grambauer/AMW-mining-detector/cluster/view.sbatch
```

This asks the cluster for a machine for two hours. Check that the job has started:

```bash
squeue -u $USER
```

When the `ST` column shows `R`, print the job's instructions:

```bash
cat $(ls -t ~/marimo-view-*.out | head -1)
```

They show a machine and port, for example `node17-001:20123`, and an address that starts
with `http://localhost:`.

### 3. Forward the port

1. Click the **Ports** tab next to **Terminal** at the bottom of VS Code.
2. If a row from last time is still there, right-click it and choose **Stop Forwarding
   Port**. It points to a machine that is gone, and it would hold the port.
3. Click **Forward a Port** and type the machine and port from the instructions, for
   example `node17-001:20123`. Press Enter.

### 4. Open the notebooks

Copy the `http://localhost:...?access_token=...` address from the instructions into your
normal browser (Chrome, Safari or Firefox), not VS Code's preview. You see the list of
notebooks. Click one.

### 5. Stop

When you are done:

1. Close the browser tab.
2. In the **Ports** tab, right-click the row and choose **Stop Forwarding Port**.
3. In the terminal, stop the job. This also stops the notebooks on the cluster:
   ```bash
   scancel --name=amw-view
   ```
4. Close the VS Code window.

If you skip step 3, the job stops by itself after two hours.

## Reading the notebooks

The settings are at the top of each notebook (subregions, detection rule, frequency or
quarter). Everything below updates when you change them. Each notebook starts with a
fold-out list of the variables it uses and what they mean, and every figure has a caption
saying what it shows: what one mark is, what the axes and colours are, and which chips and
periods it covers. Every table has a download button for a CSV file, and every chart has a
menu (`...`) to save it as an image. You can't change the data from these pages.

- **1 Clouds**: how cloudy each period and place was, maps over time, and whether
  detections depend on it.
- **2 Persistence**: whether mines stay detected from period to period, the gaps when they
  don't, and AMW's yearly rule.
- **3 Distance**: how spread out the scores are, near misses, and distance to known mines.
- **4 Frequencies**: months, quarters and AMW's quarters and years against each other,
  down to single chips.
- **Panel Overview**: the panel itself (contents, an example row, variables, checks,
  counts, a map, chips through time).
- **AMW Comparison**: our quarters against AMW's published detections. Move the threshold
  slider to see whether the disagreements are borderline scores or confident ones.
- **AMW History**: whether scoring AMW's known mining tiles first would have caught our
  detections.

## Another run

The notebooks open the newest run. To open another one, list the runs and add its folder
to the `sbatch` line:

```bash
ls /work/goldmine/grambauer/runs/
sbatch /home/goldmine/grambauer/AMW-mining-detector/cluster/view.sbatch /work/goldmine/grambauer/runs/<name>
```

## What you don't need

No git, no Python, no environment to activate, no `cd`. The job uses Ruby's copy of the
code and Ruby's Python environment on Goethe, and Ruby keeps both up to date.

## If something fails

- **`sbatch` says something about an account or is refused:** send Ruby the message.
- **`cat` finds no file, or the file says `VIEW ERROR`:** send Ruby what it says. The most
  common cause is a missing `goldmine` group (see "Before you start").
- **The browser can't open the address:** check the Ports tab forwards exactly the machine
  and port from the instructions, and that the job still shows `R` in `squeue -u $USER`.

## More

How the cluster setup works: `README_cluster.md`, section 10. To open it in VS Code, use
**File → Open File...** and enter
`/home/goldmine/grambauer/AMW-mining-detector/README_cluster.md`.
