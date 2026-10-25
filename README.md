# 🚗 AutonomousVehicle

<p align="center">
  <b>🇺🇸 English</b> &nbsp;|&nbsp; <a href="README.ko.md">🇰🇷 한국어</a> &nbsp;|&nbsp; <a href="paper/main.pdf">📕 Paper (PDF EN)</a> &nbsp;|&nbsp; <a href="paper/main_ko.pdf">📕 논문 (PDF KO)</a> &nbsp;|&nbsp; <a href="paper/main.tex">📄 LaTeX Source</a>
</p>

> *Sub-40ms Non-Autoregressive Autonomous Driving in CARLA 0.9.16 & MetaDrive powered by DINOv2 & Laya*

**AutonomousVehicle** is an open-source hybrid closed-loop autonomous driving R&D framework that couples a frozen self-supervised vision backbone (**DINOv2**, `facebook/dinov2-base`, $86.58\text{M}$) and a frozen non-autoregressive decision transformer (**Laya**, `convaiinnovations/laya`, $395.21\text{M}$) via a trainable $4 \times 9$ spatial-temporal adapter ($4.23\text{M}$ parameters) with HD-map routing, a rule-based safety shield, and kinematic path tracking. Unlike autoregressive Vision-Language-Action (VLA) models ($>300\text{ ms/frame}$), our stack executes **one synchronized forward+control pass in $32.85\text{ ms}$ ($30.4\text{ FPS}$; $31.67\text{ ms}$ base without aux/trajectory heads)** coupled with **14-node ($28\text{ m}$) Spatial Quintic Spline transitions**, **5-step ($0.25\text{ s}$) OLS lead-speed estimation**, **decoupled pedestrian depth estimation**, and **Constant-Time-Headway Adaptive Cruise Control (CTH-ACC)**.

---

## 🎬 Live Closed-Loop Autonomous Driving Demo (`driving_demo_carla_smooth` — CARLA 0.9.16 `Town10HD_Opt` 10-Scenario Suite)

<p align="center">
  <img src="assets/driving_demo_carla_smooth.gif" width="840" alt="CARLA 0.9.16 Smooth Urban FSD Demo" />
  <br/>
  🎥 <b><a href="assets/driving_demo_carla_smooth.mp4">Watch Full HD MP4 Driving Video (assets/driving_demo_carla_smooth.mp4 — 10 Scenarios × 10 Weathers)</a></b>
  <br/>
  <sub>32.85 ms / 30.4 FPS (Synchronized Neural 32.40 ms + Control 0.45 ms) · 31.67 ms (Base) · 10/10 Paired & 47/50 (94.0%) Cross-Weather Completion in Zero-Actor-Query <code>vision_estimated</code> Mode · 0.094 m Vis / 0.085 m GT Cross-Track Error</sub>
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
4. **True Constant-Time-Headway (CTH) ACC Speed Synchronization:**
   $$d^*(v_{\text{ego,kmh}}) = d_0 + \tau_h \left(\frac{v_{\text{ego,kmh}}}{3.6}\right), \quad v_{\text{ACC}} = \text{clip}\big(v_{\text{lead}} + k_g \cdot (d_{\text{lead}} - d^*(v_{\text{ego,kmh}})),\; 0,\; \min(35.0, v_{\text{lead}} + 6.0)\big)$$
   With $d_0 = 8.0\text{ m}$, $\tau_h = 1.2\text{ s}$, and gain $k_g = 0.85\text{ (km/h)/m} \equiv 0.236\text{ s}^{-1}$, the desired headway evaluates to $d^*(24\text{ km/h}) = 16.0\text{ m}$ and smoothly synchronizes with the lead vehicle without longitudinal hunting.

---

## 3. Quick Start

### 3.1. Installation
```bash
pip install -r requirements.txt
```

### 3.2. Live CARLA 0.9.16 Urban FSD Evaluation (`--simulator carla`)
With `CarlaUE4.exe` running on `127.0.0.1:2000` (`Town10HD_Opt`), run the 10-scenario, 10-weather Urban FSD benchmark with live 3D AR HUD rendering and MP4 recording:
```bash
# Run CARLA 0.9.16 evaluation in zero-actor-query vision_estimated mode (or --perception-mode gt)
python scripts/04_evaluate_sim.py --simulator carla --perception-mode vision_estimated --episodes 10 --render --save-video checkpoints/driving_demo_carla_smooth.mp4

# Run automated verification unit tests (5-step OLS range-rate, decoupled ped head, ΔV_t/Δn_t, AEB event grouping)
python tests/test_v8_verification.py
```

### 3.3. Full 4-Stage Training, Calibration & Evaluation Pipeline
```bash
# 1. Automated expert driving dataset collection
python scripts/01_collect_data.py --num-frames 50000

# 2. Train the 4.23M SpatialProjector + AuxPerceptionHead + Readout/Trajectory heads (DINOv2 & Laya frozen)
python scripts/02_train_projector.py --epochs 15 --batch-size 32

# 3. Post-hoc Temperature Scaling calibration on episode-block validation split
python scripts/03_calibrate.py

# 4. Closed-loop evaluation on 50 unseen maps / cross-weather episodes
python scripts/04_evaluate_sim.py --perception-mode vision_estimated --episodes 50
```

---

## 4. Repository Structure

```text
AutonomousVehicle/
├── assets/
│   ├── driving_demo_carla_smooth.gif # Auto-playing inline GIF for GitHub README
│   └── driving_demo_carla_smooth.mp4 # Full HD MP4 video of CARLA 0.9.16 Urban FSD scenarios
├── paper/
│   ├── main.tex                      # English LaTeX research paper (IEEE/CVPR format)
│   ├── main_ko.tex                   # Korean LaTeX research paper
│   ├── references.bib                # BibTeX bibliography database (42 references)
│   └── figures/                      # High-resolution scenario & kinematic figures
├── configs/
│   └── default.yaml                  # Camera resolution, 4x9 grid size, safety thresholds
├── checkpoints/                      # Trained SpatialProjector weights & evaluation JSON metrics
├── src/
│   ├── env/
│   │   ├── carla_wrapper.py          # CARLA 0.9.16 20Hz sync mode, 14-node Quintic spline, 3D AR HUD
│   │   ├── metadrive_wrapper.py      # MetaDrive wrapper & telemetry extraction
│   │   └── expert_labeler.py         # Expert action to Laya target label mapping
│   ├── models/
│   │   ├── dino_backbone.py          # Frozen DINOv2 ViT-B/14 patch feature extractor (86.58M)
│   │   ├── spatial_projector.py      # 4x9 AdaptiveAvgPool2d + ΔV_t/Δn_t + Decoupled AuxPerceptionHead
│   │   └── laya_pilot.py             # Frozen Laya (395.21M) + discrete & 6-WP trajectory heads
│   ├── control/
│   │   └── shield_pid.py             # Safety Shield, Bicycle Curvature FF, Linear Steer Ramp, CTH-ACC
│   └── utils/
│       ├── dataset.py                # Episode-block PyTorch Dataset & DataLoader
│       └── calibration.py            # Temperature Scaling (ECE computation & L-BFGS optimization)
├── tests/
│   └── test_v8_verification.py       # Unit tests for 5-step OLS, ped_dist_head, ΔV_t/Δn_t, and AEB events
├── scripts/                          # 01_collect_data .. 04_evaluate_sim pipeline scripts
├── README.md                         # Primary English README
└── README.ko.md                      # Korean README
```

---

## 5. Closed-Loop Driving Benchmark Results (`vision_estimated` vs. `gt` State Separated)

### 5.1. CARLA 0.9.16 (`Town10HD_Opt` & Unseen `Town03`) Closed-Loop Benchmark
Evaluated live in **CARLA 0.9.16 (`CarlaUE4.exe`, 20 Hz synchronous, `GlobalRoutePlanner` A\*)** separating zero-actor-query `vision_estimated` perception from simulator `gt` state:

| Evaluation Metric | Target KPI | `vision_estimated` Mode (No Sim Actor Query) | `gt` State Hybrid Mode | Status |
| :--- | :--- | :--- | :--- | :---: |
| **Single-Frame Latency (`32.40 ms` neural + `0.45 ms` ctrl; `+6.40 ms` IPC)** | **$\le 40.0\text{ ms}$** | **32.85 ms (30.4 FPS, P95: 33.96 ms)** | **32.85 ms (31.67 ms base w/o aux+traj)** | **PASS** |
| **`Town10HD_Opt`: 10 Paired Eps (`680 m`) / 50 Cross-Weather (`3,400 m`)** | **$\ge 85.0\%$** | **100.0% (10/10) / 94.0% (47/50, 3-seed $46.7\pm 0.6$)** | **100.0% (10/10) / 98.0% (49/50)** | **PASS** |
| **`Town10HD_Opt`: 10 Multi-Block Long Routes (`200 m/ep`, `2,000 m`)** | **$\ge 85.0\%$** | **90.0% (9/10, $R_c=95.4\%$)** | **100.0% (10/10, $R_c=100.0\%$)** | **PASS** |
| **`Town03` Unseen Map (`10 × 200 m` Roundabout/Grade, `2,000 m`)** | **$\ge 80.0\%$** | **80.0% (8/10, $R_c=91.2\%$, $\overline{|e_y|}=0.112\text{ m}$)** | **90.0% (9/10, $R_c=96.5\%$)** | **PASS** |
| **Stress Test: 10% Frame Drop + 25% Patch Occlusion (50-Ep)** | **$\ge 80.0\%$** | **88.0% (44/50, $\Delta V_t/\Delta n_t$ + latch active)** | **94.0% (47/50)** | **PASS** |
| **Teacher $\text{ECE}_{\text{mean}}$ / Physical $\text{ECE}_{\text{phys}}$ / $\text{HCER}_{0.90,\text{AEB}}$ / Physical $\text{FNR}_{\text{AEB}}$** | **$\le 0.0500$** | **0.0142 / 0.0318 / 0.45% (1.12% all) / 1/48 (2.1%)** | **0.0000 / 0.0110 / 0.10% / 0/48** | **PASS** |
| **Lead Depth MAE / Co-Pres Ped MAE / Speed Jitter RMS / $\overline{|e_y|}$** | **$\le 5.0\text{ m}$ / $\le 0.25\text{ m}$** | **3.23 m / 2.64 m / 1.84 km/h / 0.094 m** | **0.0 m / 0.0 m / 0.0 km/h / 0.085 m** | **PASS** |

### 5.2. MetaDrive 0.4.3 Procedural Highway & Intersection Benchmark ($\Delta t = 0.05\text{ s}$, 50 Unseen Maps, `20,765` steps / `9,520 m`)

| Evaluation Metric | Target KPI | `vision_estimated` Mode | `gt` State Hybrid Mode | Status |
| :--- | :--- | :--- | :--- | :---: |
| **Single-Frame Latency ($\bar{v}=33.0\text{ km/h}$, `415.3` steps / `190.4 m` per ep)** | **$\le 40.0\text{ ms}$** | **30.20 ms (33.1 FPS, P95: 31.67 ms)** | **30.20 ms (33.1 FPS)** | **PASS** |
| **Arrival Rate: 10-Ep (`4,153` steps, `1,904 m`) / 50-Ep (`20,765` steps, `9,520 m`)** | **$\ge 85.0\%$** | **100.0% (10/10) / 96.0% (48/50, CI $[86.3, 99.5]\%$)** | **100.0% (10/10) / 100.0% (50/50; 98.7% arc)** | **PASS** |
| **Unseen Closed-Loop Test $\text{ECE}_{\text{mean}}$ / NLL / Brier ($M=15$ Bins)** | **$\le 0.0500$** | **0.0184 / 0.096 / 0.031 (Noisy Sensor)** | **0.0000 / 0.041 / 0.012 (GT Teacher)** | **PASS** |

---

## 6. Author & Citation

* **Author:** **Wick-Lim** ([wicklim90@gmail.com](mailto:wicklim90@gmail.com))
* **GitHub:** [https://github.com/Wick-Lim/AutonomousVehicle](https://github.com/Wick-Lim/AutonomousVehicle)

```bibtex
@article{wicklim2026autonomousvehicle,
  title   = {Real-Time Closed-Loop Driving with Frozen DINOv2-Laya Spatial Pooling and Quintic Path Tracking},
  author  = {Wick-Lim},
  year    = {2026},
  url     = {https://github.com/Wick-Lim/AutonomousVehicle}
}
```
