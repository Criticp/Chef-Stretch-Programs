# Stretch AI — Complete First-Time Setup Guide
## Stretch RE2 (Gen 2, No D405 Wrist Camera)

This guide covers every step to get stretch_ai running on a Stretch RE2 with a GPU computer over WiFi. It was written from a real setup session and includes all the fixes we encountered.

---

## Hardware

- **Robot:** Stretch RE2 (Gen 2) — no D405 wrist camera
- **GPU Computer:** Ubuntu 22.04 (or WSL2 on Windows) with NVIDIA GPU
- **Network:** Both machines on the same WiFi network

---

## Part 1: Robot Setup (Docker)

All commands in this section run **on the robot**.

### 1.1 Unplug USB Dongle
Before starting Docker setup, unplug the USB dongle on the robot.

### 1.2 Install Docker

```bash
sudo apt-get update
sudo apt-get install docker.io
sudo usermod -aG docker $USER
newgrp docker
```

Verify:

```bash
docker run hello-world
```

You should see "Hello from Docker!"

### 1.3 Clone the Repo

```bash
git clone https://github.com/hello-robot/stretch_ai.git
```

### 1.4 Home the Robot

```bash
stretch_robot_home.py
```

### 1.5 Start the ROS2 Bridge Server

```bash
cd stretch_ai
./scripts/run_stretch_ai_ros2_bridge_server.sh --no-d405
```

**First run will download a 10+ GB Docker image.** Use ethernet if possible to speed this up. You only download it once.

When the server is running correctly you should see:
- Two beeps from the robot
- Lidar spinning
- Joint state output scrolling in the terminal

### 1.6 Open Firewall Ports (if firewall is active)

```bash
sudo ufw status
# If active:
sudo ufw allow 4401:4404/tcp
```

### 1.7 Get the Robot's IP Address

```bash
hostname -I
```

Note the first IP address (e.g., `192.168.20.203`). You'll need this for the GPU computer.

### 1.8 Verify Head Camera (D435i)

The head camera on top of the mast MUST be detected for the server to publish observations. If the server shows `[WARN] No RealSense devices were found!`, the head camera is not connected.

To check:

```bash
rs-enumerate-devices
```

Or:

```bash
lsusb | grep -i intel
```

If the camera isn't detected, try rebooting the robot:

```bash
sudo reboot
```

Then re-home and restart the server.

---

## Part 2: GPU Computer Setup (Mamba + Virtual Env)

All commands in this section run **on the GPU computer**.

### 2.1 Install Mamba

```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```

**Important:** When the installer asks "Do you wish the installer to initialize Miniforge3?" say **yes**.

Close and reopen your terminal. You should see `(base)` at the start of your prompt.

If `mamba` isn't found after install:

```bash
~/miniforge3/bin/mamba init
```

Then restart your terminal.

### 2.2 Install Git LFS and Clone the Repo

```bash
sudo apt-get install git-lfs
git lfs install
git clone https://github.com/hello-robot/stretch_ai.git --recursive
```

### 2.3 Check Your CUDA Version

```bash
nvidia-smi
```

Note the CUDA Version in the top-right corner (e.g., 12.7).

### 2.4 Run the Install Script

```bash
cd stretch_ai
./install.sh --cuda=$CUDA_VERSION --no-version
```

Replace `$CUDA_VERSION` with your actual version (e.g., `12.7`).

This creates a `stretch_ai` mamba environment.

### 2.5 Activate the Environment

```bash
mamba activate stretch_ai
```

### 2.6 Install the Stretch Package (if not automatically installed)

If you get `ModuleNotFoundError: No module named 'stretch'`:

```bash
cd ~/chef_ai/stretch_ai/src
pip install -e .
```

### 2.7 Fix PyTorch CUDA Mismatch (if needed)

Check what PyTorch was compiled with:

```bash
python -c "import torch; print(torch.version.cuda); print(torch.__version__)"
```

If the CUDA version is higher than what `nvidia-smi` shows (e.g., PyTorch has CUDA 13.0 but your driver only supports 12.7), reinstall PyTorch with a compatible CUDA version:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --force-reinstall
```

Verify:

```bash
python -c "import torch; print(torch.version.cuda); print(torch.cuda.is_available())"
```

Should show `12.4` and `True`.

### 2.8 Fix NumPy Compatibility (if needed)

If you see `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x`:

```bash
pip install "numpy<2"
```

### 2.9 Fix Audio for WSL (if using WSL)

If you get ALSA errors when using `--voice`:

```bash
export PULSE_SERVER=unix:/mnt/wslg/PulseServer
sudo apt install libasound2-plugins
```

Also ensure microphone access is enabled in **Windows Settings → Privacy & Security → Microphone → Let desktop apps access your microphone**.

---

## Part 3: Network Configuration

Run **on the GPU computer**.

### 3.1 Set the Robot IP

```bash
cd stretch_ai
./scripts/set_robot_ip.sh 192.168.20.203
```

Replace with your robot's actual IP. This saves to `~/.stretch/robot_ip.txt` and only needs to be done once per IP.

### 3.2 Open Firewall Ports

```bash
sudo ufw allow 4401:4404/tcp
```

### 3.3 Verify Connectivity

```bash
ping 192.168.20.203
```

Should get responses. Ctrl+C to stop.

Test the specific ports:

```bash
nc -zv 192.168.20.203 4401
nc -zv 192.168.20.203 4402
nc -zv 192.168.20.203 4403
nc -zv 192.168.20.203 4404
```

All four should say "succeeded."

---

## Part 4: Verify Installation

With the server running on the robot, run **on the GPU computer**:

```bash
mamba activate stretch_ai
python -m stretch.app.view_images
```

You should see:
- The robot's gripper opens
- The arm moves
- Camera video appears on your GPU computer

Press `q` to quit.

---

## Part 5: LLM Setup

### 5.1 Chat with Qwen (Local, Free)

```bash
python -m stretch.app.chat --llm qwen25
```

First run downloads ~6 GB of model weights. Once loaded, you'll see a `You:` prompt. Type a message and press Enter.

### 5.2 Chat with Voice

```bash
python -m stretch.app.chat --llm qwen25 --prompt pickup --voice
```

Press Enter to start recording, speak your command.

### 5.3 OpenAI Alternative (No GPU Needed for LLM)

```bash
export OPENAI_API_KEY="sk-your-key-here"
python -m stretch.app.chat --llm openai
```

---

## Part 6: What Works on RE2 Without D405

### Works

| App | Command |
|---|---|
| View camera images | `python -m stretch.app.view_images` |
| Keyboard teleop | `python -m stretch.app.keyboard_teleop` |
| Print joint states | `python -m stretch.app.print_joint_states` |
| LLM chat (text) | `python -m stretch.app.chat --llm qwen25` |
| LLM chat (voice) | `python -m stretch.app.chat --llm qwen25 --prompt pickup --voice` |
| Autonomous mapping | `python -m stretch.app.mapping` |
| Read saved maps | `python -m stretch.app.read_map -i <file>.pkl` |

### Does NOT Work (Requires D405 Wrist Camera)

| Feature | Reason |
|---|---|
| `ai_pickup` (pick and place) | Grasp planning needs wrist camera depth |
| `grasp_object` | Same — needs wrist RGBD |
| Learning from Demonstration | Data collection requires wrist camera |
| Dex Teleop | Requires D405 + Dex Teleop Kit |

**To unlock pick-and-place:** Purchase the Stretch 2 Upgrade Kit from Hello Robot, which adds the D405 wrist camera.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `No RealSense devices were found` | Head camera not detected. Reboot robot with `sudo reboot`, re-home, restart server. |
| `Timeout waiting for observations` | Server not publishing. Check robot terminal for errors. Verify `--no-d405` flag was used. |
| `ModuleNotFoundError: No module named 'stretch'` | Run `pip install -e .` from `stretch_ai/src/` directory. |
| CUDA driver too old | Reinstall PyTorch: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --force-reinstall` |
| NumPy version conflict | Run `pip install "numpy<2"` |
| ALSA audio errors (WSL) | Set `export PULSE_SERVER=unix:/mnt/wslg/PulseServer` and install `libasound2-plugins` |
| Ping works but ports fail | Run `sudo ufw allow 4401:4404/tcp` on both machines |
| Docker image download slow | Plug ethernet into the robot for faster download |
