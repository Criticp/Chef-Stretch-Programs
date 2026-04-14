# Stretch AI — Daily Quick-Start Guide
## Steps to Run Each Session

Everything is already installed. Follow these steps each time you want to use the robot.

---

## Step 1: Start the Robot

Go to the **robot's terminal** (or SSH in from your GPU computer).

### 1a. Home the robot

```bash
stretch_robot_home.py
```

### 1b. Start the server

```bash
cd stretch_ai
./scripts/run_stretch_ai_ros2_bridge_server.sh --no-d405
```

**Wait for:**
- Two beeps
- Lidar spinning
- Joint state output scrolling

Leave this terminal running.

---

## Step 2: Set Up the GPU Computer

Open a terminal on your **GPU computer**.

### 2a. Activate the environment

```bash
mamba activate stretch_ai
```

### 2b. (WSL only) Enable audio for voice commands

```bash
export PULSE_SERVER=unix:/mnt/wslg/PulseServer
```

---

## Step 3: Verify Connection

Quick check that everything is connected:

```bash
python -m stretch.app.view_images
```

You should see camera video. Press `q` to quit.

If it times out, check:
- Is the server running on the robot?
- Can you `ping 192.168.20.203`?
- Did the robot's IP change? If so: `./scripts/set_robot_ip.sh NEW.IP.HERE`

---

## Step 4: Run What You Need

### Autonomous Mapping (robot explores on its own)
```bash
python -m stretch.app.mapping
```

### Keyboard Teleop (drive the robot manually)
```bash
python -m stretch.app.keyboard_teleop
```

### LLM Chat — Text Input
```bash
python -m stretch.app.chat --llm qwen25
```

### LLM Chat — Voice Input
```bash
python -m stretch.app.chat --llm qwen25 --prompt pickup --voice
```
Press Enter to record, speak your command.

### LLM Chat — OpenAI Instead of Local Qwen
```bash
export OPENAI_API_KEY="sk-your-key-here"
python -m stretch.app.chat --llm openai
```

---

## Step 5: Shut Down

1. **GPU computer:** Ctrl+C to stop the app
2. **Robot:** Ctrl+C to stop the server

---

## If the Robot Reboots

Just repeat from Step 1 (home + start server). The GPU computer side doesn't need any changes unless the robot's IP changed.

---

## Quick Reference

| I want to... | Command (GPU computer) |
|---|---|
| See camera feed | `python -m stretch.app.view_images` |
| Drive manually | `python -m stretch.app.keyboard_teleop` |
| Auto-explore & map | `python -m stretch.app.mapping` |
| Chat with LLM (text) | `python -m stretch.app.chat --llm qwen25` |
| Chat with LLM (voice) | `python -m stretch.app.chat --llm qwen25 --prompt pickup --voice` |
| Check joint states | `python -m stretch.app.print_joint_states` |
| Update robot IP | `./scripts/set_robot_ip.sh NEW.IP.HERE` |
| Update Docker image | `./scripts/run_stretch_ai_ros2_bridge_server.sh --update` (on robot) |
| Update stretch_ai code | `git pull -ff origin main` (on GPU computer) |
