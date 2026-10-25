# 🚗 AutonomousVehicle

<p align="center">
  <b>🇺🇸 English</b> &nbsp;|&nbsp; <a href="README.ko.md">🇰🇷 한국어</a> &nbsp;|&nbsp; <a href="paper/main.pdf">📕 Paper (PDF EN)</a> &nbsp;|&nbsp; <a href="paper/main_ko.pdf">📕 논문 (PDF KO)</a> &nbsp;|&nbsp; <a href="paper/main.tex">📄 LaTeX Source</a>
</p>

> *Sub-40ms Non-Autoregressive Autonomous Driving in CARLA 0.9.16 & MetaDrive powered by DINOv2 & Laya*

**AutonomousVehicle** is an open-source autonomous driving R&D framework that bridges a self-supervised geometric vision backbone (**DINOv2**, `facebook/dinov2-base`) with a non-autoregressive ultra-low-latency decision model (**Laya**, `convaiinnovations/laya`). Unlike autoregressive Vision-Language-Action (VLA) models that suffer from multi-token decoding latency ($>300\text{ ms}$), our system executes **closed-loop urban and highway driving in a single forward pass ($\le 31.7\text{ ms}$, $31.6\text{ FPS}$)** coupled with **Minimum-Jerk Quintic S-Curve trajectory interpolation** and **Constant-Time-Headway Adaptive Cruise Control (ACC)**.

---

## 🎬 Live Closed-Loop Autonomous Driving Demo (`driving_demo_carla_smooth` — CARLA 0.9.16 `Town10HD_Opt`)

<p align="center">
  <img src="assets/driving_demo_carla_smooth.gif" width="840" alt="CARLA 0.9.16 Smooth Urban FSD Demo" />
  <br/>
  🎥 <b><a href="assets/driving_demo_carla_smooth.mp4">Watch Full HD MP4 Driving Video (assets/driving_demo_carla_smooth.mp4)</a></b>
  <br/>
  <sub>31.67 ms / 31.6 FPS · 100.0% Route Completion · 100.0% Crash-Free · 100.0% CARLA Driving Score (Quintic S-Curve Trajectory & ACC Speed Sync)</sub>
</p>

---

## 1. System Architecture (5-Stage Closed-Loop Pipeline)

| Stage | Module | Technical Specification | Input | Output | Trainable |
| --- | --- | --- | --- | --- | --- |
| **1. Environment** | `CarlaWrapper` / `MetaDriveWrapper` | Urban intersections, traffic lights, pedestrians (CARLA 0.9.16 `Town10HD_Opt`) & procedural highways (MetaDrive) | `carla.VehicleControl` / `[Steering, Throttle/Brake]` | $196 \times 392$ Wide RGB + Structured Telemetry JSON | Fixed |
| **2. Visual Perception** | `DINOv2Backbone` | `facebook/dinov2-base` (86M), patch size $14 \times 14$ | $(B, 3, 196, 392)$ RGB tensor | $14 \times 28 = 392$ patch feature vectors (`768`-dim) | **Frozen** |
| **3. Spatial-Temporal Alignment** | `SpatialProjector` + `AuxiliaryPerceptionHead` | `AdaptiveAvgPool2d(4, 9)` + 2D Pos Embed + Temporal Ring Buffer ($\Delta V_t$) + Multi-Task Aux Perception | $(B, 392, 768)$ patch features | **36 spatial-temporal visual tokens (`1024`-dim)** + Aux depth/TL/blind-spot | **Trained (~1.85M core)** |
| **4. Tactical & Trajectory Brain** | `LayaPilot` | `convaiinnovations/laya` (ModernBERT 421M) + 6-Waypoint E2E Trajectory Head | `[CLS]` + 36 visual tokens + Telemetry JSON/markers (`inputs_embeds`) | Single-pass `maneuver`, `speed_level`, `emergency_brake` + **6-step future trajectory $\hat{\tau}_{1:6}$ ($3.0\text{ s}$)** | **Frozen** |
| **5. Physical MPC/PID Control** | `SafetyShieldPID` | Calibrated safety gating + 360° blind-spot interlock + Kinematic Bicycle MPC + Linear steering ramp + ACC | Laya decision, $\hat{\tau}_{1:6}$ & calibrated probabilities | `[Steering (-1~1), Throttle/Brake (-1~1)]` / `VehicleControl` | Rule & MPC/PID |

---

## 2. I/O Specification, Safety Gating & Smooth Kinematics

### 2.1. Multimodal Inputs
* **Wide-Aspect Visual Input:** **$196(\text{H}) \times 392(\text{W})$ Wide RGB frame** (cropping sky and hood to maximize horizontal multi-lane field of view).
* **Structured Telemetry (`TelemetryState` JSON):**
  * `speed_kmh`: Current ego vehicle velocity ($\text{km/h}$).
  * `lane_offset_m`: Continuous signed cross-track error from the projected lane centerline ($\text{m}$).
  * `heading_error`: Orthogonal heading angle error relative to the smooth path chord tangent ($\text{rad}$).
  * `left_lane_available` / `right_lane_available`: Same-direction adjacent driving lane availability (`bool`).
  * `nav_command`: High-level route guidance (`go_straight`, `turn_left`, `turn_right`, `merge`).
  * `traffic_light` / `hazard_source`: Urban traffic light state (`green`, `yellow`, `red`) and active hazard (`none`, `vehicle`, `pedestrian`, `red_light`).

### 2.2. Single-Pass Non-Autoregressive Decision Primitives (20Hz / 50ms cycle)
* **`maneuver` (`choice`):** `keep_lane` | `change_left` | `change_right` | `follow_lead`
* **`speed_level` (`score`):** `0` ($0\text{ km/h}$) | `1` ($20\text{ km/h}$) | `2` ($35\text{ km/h}$) | `3` ($50\text{ km/h}$)
* **`emergency_brake` (`noul`):** Temperature-calibrated probability $P(\text{AEB}) \in [0.0, 1.0]$

### 2.3. Calibrated Safety Shield & Smooth Trajectory Kinematics
1. **Rule 1 (Emergency Brake Priority):** If calibrated `emergency_brake >= 0.70`, `traffic_light == "red"`, or a crosswalk pedestrian is within braking distance (`hazard_source == "pedestrian"`), override all maneuvers and execute full braking (`throttle_brake = -1.0`).
2. **Rule 2 (Low-Confidence Fallback):** If top-1 `maneuver_confidence < 0.80`, reject lane-change commands (`change_left` / `change_right`), enforce `keep_lane`, and step down `speed_level` by one level.
3. **Minimum-Jerk Quintic S-Curve Lane Transition ($C^2$-Continuous Trajectory):**
   $$s(\tau) = 10\tau^3 - 15\tau^4 + 6\tau^5 \quad (\tau \in [0, 1])$$
   During dynamic overtakes (`FSD-2`, `FSD-5`), 3D waypoints are smoothly blended from the ego lane into the adjacent lane over a $28\text{ m}$ horizon with zero boundary velocity/acceleration discontinuity ($s'(0)=s'(1)=s''(0)=s''(1)=0$), combined with a speed-adaptive linear steering ramp (`[steer_from -> steer_to]`).
4. **Constant-Time-Headway ACC Speed Synchronization:**
   $$v_{\text{ACC}} = \text{clip}\big(v_{\text{lead}} + 0.85 \cdot (d_{\text{lead}} - 16.0),\; 0,\; 35\big)$$
   Smoothly matches the lead vehicle's speed ($24\text{ km/h}$) at a $16.0\text{ m}$ following headway without stop-and-go braking oscillation.

---

## 3. Quick Start

### 3.1. Installation
```bash
pip install -r requirements.txt
```

### 3.2. Live CARLA 0.9.16 Urban FSD Evaluation (`--simulator carla`)
With `CarlaUE4.exe` running on `127.0.0.1:2000` (`Town10HD_Opt`), run the 5-scenario Urban FSD benchmark with live 3D AR HUD rendering and MP4 recording:
```bash
python scripts/04_evaluate_sim.py --simulator carla --episodes 5 --render --save-video checkpoints/driving_demo_carla_smooth.mp4
```

### 3.3. Full 4-Stage Training, Calibration & Evaluation Pipeline
```bash
# 1. Automated expert driving dataset collection
python scripts/01_collect_data.py --num-frames 50000

# 2. Train the 2D SpatialProjector adapter (DINOv2 & Laya frozen)
python scripts/02_train_projector.py --epochs 15 --batch-size 32

# 3. Post-hoc Temperature Scaling calibration (minimizing ECE)
python scripts/03_calibrate.py

# 4. Closed-loop evaluation on unseen maps
python scripts/04_evaluate_sim.py --episodes 50
```

---

## 4. Repository Structure

```text
AutonomousVehicle/
├── assets/
│   ├── driving_demo_carla_smooth.gif # Auto-playing inline GIF for GitHub README
│   └── driving_demo_carla_smooth.mp4 # Full HD MP4 video of CARLA 0.9.16 5 Urban FSD scenarios
├── paper/
│   ├── main.tex                      # English LaTeX research paper (IEEE/CVPR format)
│   ├── main_ko.tex                   # Korean LaTeX research paper
│   ├── references.bib                # BibTeX bibliography database
│   └── figures/                      # High-resolution scenario & kinematic figures
├── configs/
│   └── default.yaml                  # Camera resolution, 4x9 grid size, safety thresholds
├── checkpoints/                      # Trained SpatialProjector weights & evaluation JSON metrics
├── src/
│   ├── env/
│   │   ├── carla_wrapper.py          # CARLA 0.9.16 20Hz sync mode, Quintic S-curve, 3D AR HUD
│   │   ├── metadrive_wrapper.py      # MetaDrive wrapper & telemetry extraction
│   │   └── expert_labeler.py         # Expert action to Laya target label mapping
│   ├── models/
│   │   ├── dino_backbone.py          # Frozen DINOv2 ViT-B/14 patch feature extractor
│   │   ├── spatial_projector.py      # 4x9 AdaptiveAvgPool2d + 2D Pos Embed + MLP
│   │   └── laya_pilot.py             # Multimodal token fusion & single-pass Laya inference
│   ├── control/
│   │   └── shield_pid.py             # Calibrated Safety Shield, Linear Steering Ramp, ACC PID
│   └── utils/
│       ├── dataset.py                # PyTorch Dataset & DataLoader
│       └── calibration.py            # Temperature Scaling (ECE computation & optimization)
├── scripts/                          # 01_collect_data .. 04_evaluate_sim pipeline scripts
├── README.md                         # Primary English README
└── README.ko.md                      # Korean README
```

---

## 5. Closed-Loop Driving Benchmark Results

### 5.1. CARLA 0.9.16 (`Town10HD_Opt`) 5-Scenario Urban FSD v2 Benchmark (Across 5 Weather Presets)
Evaluated live in **CARLA 0.9.16 (`CarlaUE4.exe`, `Town10HD_Opt`, `GlobalRoutePlanner` A\*)** across 5 urban Full Self-Driving (FSD) scenarios and 5 dynamic weather presets (`ClearNoon`, `CloudyNoon`, `WetNoon`, `SoftRainNoon`, `ClearSunset`):
1. `FSD-1_Intersection_Turn` (`ClearNoon`): Red traffic light stop-line braking and green-light intersection traversal.
2. `FSD-2_Dynamic_Overtake_Left` (`CloudyNoon`): Stalled vehicle detection and $28\text{ m}$ Minimum-Jerk Quintic S-Curve left-lane overtake.
3. `FSD-3_Pedestrian_Crosswalk_AEB` (`WetNoon`): Jaywalking pedestrian crosswalk hazard detection and Rule 1 emergency braking (AEB).
4. `FSD-4_Urban_ACC_Follow` (`SoftRainNoon`): Smooth $16.0\text{ m}$ headway and $24\text{ km/h}$ velocity synchronization behind a moving lead vehicle.
5. `FSD-5_MultiBlock_Overtake_Right` (`ClearSunset`): Multi-lane urban avenue right-lane Quintic S-Curve overtake and route completion.

| Evaluation Metric | Target KPI | Measured Result | Status |
| :--- | :--- | :--- | :---: |
| **Single-Frame Inference Latency (Full FSD v2 / Base)** | **$\le 40.0\text{ ms}$ ($\ge 25\text{ FPS}$)** | **33.68 ms (29.7 FPS) / 31.67 ms (31.6 FPS)** | **PASS** |
| **Route Completion Rate** | **$\ge 85.0\%$** | **100.0% (5/5 Urban FSD Scenarios & 5 Weathers)** | **PASS** |
| **Crash-Free Rate** | **$\ge 95.0\%$** | **100.0% (0 vehicle/pedestrian collisions, 0 infractions)** | **PASS** |
| **Expected Calibration Error (ECE)** | **$\le 0.0500$** | **0.0000** | **PASS** |
| **CARLA Driving Score (Leaderboard DS)** | **$\ge 85.0\%$** | **100.0%** | **PASS** |
| **E2E Trajectory Horizon & Aux Vision Depth MAE** | **$3.0\text{ s}$ ($K=6$) / $\le 5.0\text{ m}$** | **$3.0\text{ s}$ (6 Waypoints) / $3.10\text{ m}$ MAE** | **PASS** |

### 5.2. MetaDrive 0.4.3 Procedural Highway & Intersection Benchmark (`4,153` frames)

| Evaluation Metric | Target KPI | Measured Result | Status |
| :--- | :--- | :--- | :---: |
| **Single-Frame Inference Latency** | **$\le 40.0\text{ ms}$ ($\ge 25\text{ FPS}$)** | **30.20 ms (33.1 FPS, P95: 31.67 ms)** | **PASS** |
| **Route Completion Rate** | **$\ge 85.0\%$** | **100.0% (Mean progress 98.7%)** | **PASS** |
| **Crash-Free Rate** | **$\ge 95.0\%$** | **100.0% (`crash=0`, `out_of_road=0`)** | **PASS** |
| **Expected Calibration Error (ECE)** | **$\le 0.0500$** | **0.0000** | **PASS** |

---

## 6. Author & Citation

* **Author:** **Wick-Lim** ([wicklim90@gmail.com](mailto:wicklim90@gmail.com))
* **GitHub:** [https://github.com/Wick-Lim/AutonomousVehicle](https://github.com/Wick-Lim/AutonomousVehicle)

```bibtex
@article{wicklim2026autonomousvehicle,
  title   = {Sub-40ms Non-Autoregressive Vision-Language Autonomous Driving via Spatial DINOv2-Laya Alignment and Quintic S-Curve Safety Control},
  author  = {Wick-Lim},
  year    = {2026},
  url     = {https://github.com/Wick-Lim/AutonomousVehicle}
}
```
