# gr00t_wbc

Software stack for loco-manipulation experiments across multiple humanoid platforms, with primary support for the Unitree G1. This repository provides whole-body control policies, a teleoperation stack, and a data exporter. 

---

## System Installation

### Prerequisites
- Ubuntu 22.04
- NVIDIA GPU with a recent driver
- Docker and NVIDIA Container Toolkit (required for GPU access inside the container)

### Repository Setup
Install Git and Git LFS:
```bash
sudo apt update
sudo apt install git git-lfs
git lfs install
```

Clone the repository:
```bash
mkdir -p ~/Projects
cd ~/Projects
git clone https://github.com/NVlabs/gr00t_wbc.git
cd gr00t_wbc
```

### Docker Environment
We provide a Docker image with all dependencies pre-installed.

Install a fresh image and start a container:
```bash
./docker/run_docker.sh --install --root
```
This pulls the latest `gr00t_wbc` image from `docker.io/nvgear`.

Start or re-enter a container:
```bash
./docker/run_docker.sh --root
```

Use `--root` to run as the `root` user. To run as a normal user, build the image locally:
```bash
./docker/run_docker.sh --build
```
---

## Running the Control Stack

Once inside the container, the control policies can be launched directly.

- Simulation:
  ```bash
  python gr00t_wbc/control/main/teleop/run_g1_control_loop.py
  ```
- Real robot: Ensure the host machine network is configured per the [G1 SDK Development Guide](https://support.unitree.com/home/en/G1_developer) and set a static IP at `192.168.123.222`, subnet mask `255.255.255.0`:
  ```bash
  python gr00t_wbc/control/main/teleop/run_g1_control_loop.py --interface real
  ```

Keyboard shortcuts (terminal window):
- `]`: Activate policy
- `o`: Deactivate policy
- `9`: Release / Hold the robot
- `w` / `s`: Move forward / backward
- `a` / `d`: Strafe left / right
- `q` / `e`: Rotate left / right
- `z`: Zero navigation commands
- `1` / `2`: Raise / lower the base height
- `backspace` (viewer): Reset the robot in the visualizer

---

## Running the Teleoperation Stack

The teleoperation policy primarily uses Pico controllers for coordinated hand and body control. It also supports other teleoperation devices, including LeapMotion and HTC Vive with Nintendo Switch Joy-Con controllers.

Keep `run_g1_control_loop.py` running, and in another terminal run:

```bash
python gr00t_wbc/control/main/teleop/run_teleop_policy_loop.py --hand_control_device=pico --body_control_device=pico
```

### Pico Setup and Controls
Configure the teleop app on your Pico headset by following the [XR Robotics guidelines](https://github.com/XR-Robotics). 

The necessary PC software is pre-installed in the Docker container. Only the [XRoboToolkit-PC-Service](https://github.com/XR-Robotics/XRoboToolkit-PC-Service) component is needed.

Prerequisites: Connect the Pico to the same network as the host computer.

Controller bindings:
- `menu + left trigger`: Toggle lower-body policy
- `menu + right trigger`: Toggle upper-body policy
- `Left stick`: X/Y translation
- `Right stick`: Yaw rotation
- `L/R triggers`: Control hand grippers

Pico unit test:
```bash
python gr00t_wbc/control/teleop/streamers/pico_streamer.py
```

---

## Running the Data Collection Stack

Run the full stack (control loop, teleop policy, and camera forwarder) via the deployment helper:
```bash
python scripts/deploy_g1.py \
    --interface sim \
    --camera_host localhost \
    --sim_in_single_process \
    --simulator robocasa \
    --image-publish \
    --enable-offscreen \
    --env_name PnPBottle \
    --hand_control_device=pico \
    --body_control_device=pico
```

The `tmux` session `g1_deployment` is created with panes for:
- `control_data_teleop`: Main control loop, data collection, and teleoperation policy
- `camera`: Camera forwarder
- `camera_viewer`: Optional live camera feed

Operations in the `controller` window (`control_data_teleop` pane, left):
- `]`: Activate policy
- `o`: Deactivate policy
- `k`: Reset the simulation and policies
- `` ` ``: Terminate the tmux session
- `ctrl + d`: Exit the shell in the pane

Operations in the `data exporter` window (`control_data_teleop` pane, right top):
- Enter the task prompt

Operations on Pico controllers:
- `A`: Start/Stop recording
- `B`: Discard trajectory

---

## Running G1 Without Dexterous Hands (Stub Configuration)

If your G1 robot has rubber stubs instead of Dex3/Inspire hands, use these configurations:

### Network Setup (Real Robot)

Configure the ethernet interface connected to the robot:
```bash
sudo ip link set <interface> up
sudo ip addr flush dev <interface>
sudo ip addr add 192.168.123.222/24 dev <interface>
```

### Control Loop Commands

**Simulation mode (no hands):**
```bash
python gr00t_wbc/control/main/teleop/run_g1_control_loop.py --interface sim --no-with-hands
```

**Real robot (no hands):**
```bash
python gr00t_wbc/control/main/teleop/run_g1_control_loop.py --interface real --no-with-hands
```

### Motion Playback

To play recorded motions on the upper body while the RL policy controls legs:

```bash
python gr00t_wbc/control/main/teleop/publish_upper_body_from_results.py \
  --results resources/poses/results.pkl \
  --loop \
  --teleop-frequency 30 \
  --hand-mode zero \
  --speed 0.5 \
  --initial-pose-seconds 5.0 \
  --upper-body-only
```

Key flags:
- `--no-with-hands`: Disables hand SDK communication (prevents DDS errors)
- `--upper-body-only`: Only publishes arm poses, lets RL policy control legs
- `--speed 0.5`: Slows motion playback to prevent safety violations
- `--initial-pose-seconds 5.0`: Allows robot to slowly reach initial pose

### Troubleshooting

| Error | Cause | Solution |
|-------|-------|----------|
| `ddsi_udp_conn_write to udp/127.0.0.1:7410 failed` | Hand SDK trying to connect to non-existent hands | Use `--no-with-hands` flag |
| `real: does not match an available interface` | No network interface has `192.168.123.x` IP | Configure ethernet interface or use `--interface sim` |
| `Joint safety bounds exceeded` | Motion causing joints to move too fast | Reduce `--speed` and increase `--initial-pose-seconds` |

---

## Hand Configurations

| Hand Type | DOF (per hand) | Total Joints | Flag |
|-----------|---------------|--------------|------|
| Rubber Stubs | 0 | 29 (body only) | `--no-with-hands` |
| Unitree Dex3 | 7 | 43 | `--with-hands --hand_type dex3` (default) |
| Inspire RH56 | 6 | 41 | `--with-hands --hand_type inspire` |

Notes:
- Inspire hand DDS topics are supported in the hand wrappers.
- MuJoCo simulation for `g1 + inspire` is not fully wired yet in this repo (scene + full robot model mapping), so use Inspire with `--interface real` for now.
