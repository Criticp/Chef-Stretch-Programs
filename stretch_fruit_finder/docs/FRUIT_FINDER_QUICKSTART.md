# Fruit Finder Quickstart

A new user's guide to running the Stretch RE2 Fruit Finder GUI.

## TL;DR — running the GUI

After everything is installed, **double-click "Chef Fruit Finder" on Stretch's desktop.** A terminal window opens, the GUI window opens above it, and you're ready to go.

If you don't see the desktop icon, see [First-time setup](#first-time-setup) below.

---

## What the GUI does

The Level-4 GUI (`bringup/test_04_gui.py`) gives you:

- **Live camera preview** with bounding boxes and a center crosshair.
- **Set target** dropdown — pick a fruit (apple, banana, etc.) or "(any food)".
- **Start search** — pans the head across pan + multiple tilt rows looking for the chosen fruit.
- **Stop** — interrupts whatever is running and returns to IDLE.
- **Hover above target** — only enabled while TRACKING. Stows the arm, then visually-servos the lift and arm so the closed gripper hovers above the fruit.
- **Stow arm** — returns arm to the known starting pose.
- **Gripper points inward** checkbox — flips the wrist 180° between outward (along arm) and inward (back toward robot). Persists across Stow / Hover.

State machine: `IDLE → SEARCHING → TRACKING → HOVERING → HOVER_HOLD → TRACKING (auto-resume)`.

Keybindings (focus the camera area first, then):

| Keys | Action |
|------|--------|
| WASD / arrows | Drive the base |
| I / K | Lift up / down |
| J / L | Arm retract / extend |
| U / O | Wrist yaw |
| Y / H | Wrist pitch |
| N / M | Wrist roll |
| `[` / `]` | Gripper close / open |
| X | Stow arm |

---

## First-time setup

These steps only need to be done once per user account on Stretch. After that, the desktop icon does everything.

### 1. Have the repo somewhere on Stretch

Typical location: `~/chef_ai/Chef-Stretch-Programs/`. If it's already cloned, just `git pull` to get the latest.

```bash
cd ~/chef_ai/Chef-Stretch-Programs
git pull
```

### 2. Install Python dependencies (once per fresh setup)

```bash
cd ~/chef_ai/Chef-Stretch-Programs/stretch_fruit_finder
pip install -r requirements.txt
```

If pip complains, you may need `python3 -m pip install --user -r requirements.txt`.

### 3. Install the desktop launcher

```bash
cd ~/chef_ai/Chef-Stretch-Programs/stretch_fruit_finder
chmod +x scripts/run_gui.sh scripts/install_launcher.sh
./scripts/install_launcher.sh
```

Output should look like:
```
Installed launcher:
  /home/<you>/.local/share/applications/chef-fruit-finder.desktop
  /home/<you>/Desktop/chef-fruit-finder.desktop
```

You should now see a **Chef Fruit Finder** icon on the GNOME desktop and in the Activities grid. Double-click it to launch.

If the desktop icon shows as "Untrusted Application Launcher" the first time, right-click it → **Allow Launching**. The installer also tries to mark it trusted automatically via `gio set ... metadata::trusted true`, but that requires the GNOME desktop session to be running — it's a no-op when run over SSH.

### 4. (Optional) Test from terminal first

If anything goes wrong, run the launcher directly so you can see the Python error in the terminal:

```bash
~/chef_ai/Chef-Stretch-Programs/stretch_fruit_finder/scripts/run_gui.sh
```

Same effect as double-clicking the icon. The icon launches it under `Terminal=true`, so the terminal window with logs opens automatically — that's where to look first if the GUI doesn't appear.

---

## Re-running after a `git pull`

The launcher reads the current code each time it runs, so you don't need to re-install anything after a normal `git pull`. The desktop entry hardcodes the path to `run_gui.sh`, and that script always runs the latest checkout.

You only need to re-run `./scripts/install_launcher.sh` if:

- You moved the repo to a new location.
- You installed it under a different user account.
- The launcher template (`scripts/chef-fruit-finder.desktop.in`) changed and you want the new metadata.

---

## Common issues

### Desktop icon does nothing / "Untrusted launcher"

Right-click the icon → **Allow Launching**. Or in a terminal:
```bash
gio set ~/Desktop/chef-fruit-finder.desktop metadata::trusted true
```

### Window opens but camera shows "Waiting for camera..." forever

The RealSense isn't being detected. Plug-cycle the USB or check:
```bash
rs-enumerate-devices | head
```
If that doesn't list a D435i, the camera isn't reachable at the OS level — fix that before bothering with the GUI.

### "Module not found" Python errors in the terminal

Re-run `pip install -r requirements.txt` from `stretch_fruit_finder/`. The launcher uses the system `python3`, so packages must be installed into that interpreter (or via `--user`).

### Gamepad/keyboard inputs ignored

Click on the **camera image** in the GUI window to give it keyboard focus. Tk only routes key events to the focused widget.

### Hover button is greyed out

It only enables during TRACKING. You need to first set a target, click Start search, and wait for the head to lock onto the fruit before Hover becomes clickable.

---

## What to tell the user verbally

> "There's a Chef Fruit Finder icon on the desktop. Double-click it — a terminal opens, then a GUI window with a live camera. Pick a fruit from the dropdown, click Set target, click Start search, and the head will sweep until it locks on. Once it's tracking, click Hover above target and the arm will move so the gripper hovers right above the fruit. Stop button or closing the window ends everything."
