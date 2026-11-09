# Looking at the mining panel on Goethe

The results stay on the Goethe cluster. You open them in your browser through VS Code.
You only need your Goethe login.

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
2. Click **Forward a Port** and type the machine and port from the instructions, for
   example `node17-001:20123`. Press Enter.

### 4. Open the notebooks

Copy the `http://localhost:...?access_token=...` address from the instructions into your
normal browser (Chrome, Safari or Firefox), not VS Code's preview. You see two notebooks.
Click one. The first time it opens, it can take a minute or two to read the run.

### 5. Stop

When you are done, close the browser tab and run this in the terminal:

```bash
scancel --name=amw-view
```

Then close the VS Code window. If you skip this, the job stops by itself after two hours.

## Reading the notebooks

The settings are at the top of each notebook: the quarter and the detection rule in Panel
Overview, the quarter and the detection threshold in AMW Comparison. Everything below
updates when you change them. Every table has a download button for a CSV file, and every
chart has a menu (`...`) to save it as an image. You can't change the data from these pages.

**Panel Overview**, the panel itself:

1. What is in the panel: periods, chips, and how many chips were scored.
2. The panel as a table: example rows, what each variable means, and how each variable
   is distributed.
3. Checks: what must always hold in the data, and whether it does.
4. How many chips each detection rule flags, per month and quarter.
5. How well each period was seen: how many chips had (almost) no cloud-free image.
6. Whether detections depend on how well a chip was seen.
7. Whether the months agree with their quarter.
8. A map of the detections. Scroll to zoom, drag to move, click a chip to follow it.
9. The chips you clicked, or typed IDs of, through time.
10. AMW's distances against distances on the Earth.

**AMW Comparison**, our quarters against AMW's published detections:

- At the top: how many chips we detect, how many AMW detects, and how much the two overlap.
- Below: agreement per quarter and subregion, the two scores side by side, image quality,
  why the scores differ (the last day of the quarter), a map of where they differ, border
  chips, and our months against AMW's quarter.

Move the threshold slider to see whether the disagreements are borderline scores or
confident ones.

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

## More

How the cluster setup works: `README_cluster.md`, section 10. To open it in VS Code, use
**File → Open File...** and enter
`/home/goldmine/grambauer/AMW-mining-detector/README_cluster.md`.
