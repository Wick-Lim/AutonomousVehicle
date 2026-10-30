# 🚗 AutonomousVehicle

<p align="center">
  <b>🇺🇸 English</b> &nbsp;|&nbsp; <a href="README.ko.md">🇰🇷 한국어</a> &nbsp;|&nbsp; <a href="paper/main.pdf">📕 Paper (PDF EN)</a> &nbsp;|&nbsp; <a href="paper/main_ko.pdf">📕 논문 (PDF KO)</a> &nbsp;|&nbsp; <a href="paper/main.tex">📄 LaTeX Source</a>
</p>

> *Sub-40ms Non-Autoregressive Closed-Loop Autonomous Driving in CARLA 0.9.16 & MetaDrive 0.4.3 powered by Frozen DINOv2 & Laya*

**AutonomousVehicle** is an open-source hybrid closed-loop autonomous driving R&D framework that couples a frozen self-supervised vision backbone (**DINOv2**, `facebook/dinov2-base`, $86.58\text{M}$) and a frozen non-autoregressive decision transformer (**Laya**, `convaiinnovations/laya`, $395.21\text{M}$) via a trainable $4 \times 9$ spatial-temporal adapter ($4.23\text{M}$ parameters / $4,232,991$ trainable weights: $1.85\text{M}$ spatial MLP, $1.05\text{M}$ $\Delta n_t$-normalized temporal gate, $1.05\text{M}$ decoupled auxiliary perception head, and $0.28\text{M}$ readout/6-WP trajectory heads) with HD-map routing, a rule-based safety shield, and kinematic path tracking. Unlike autoregressive Vision-Language-Action (VLA) decoders ($>300\text{ ms/frame}$), our stack executes **one synchronized forward+control pass in $32.85\text{ ms}$ ($30.4\text{ FPS}$; $31.67\text{ ms}$ base; sensor-to-actuator AoI mean $39.25\text{ ms}$, P99 $44.12\text{ ms}$)** coupled with **14-node ($28\text{ m}$) Spatial Quintic Spline transitions**, **5-sample ($0.20\text{ s}$ span) timestamp-aware OLS lead-speed estimation**, **decoupled pedestrian depth estimation**, and **Stop-and-Go Constant-Time-Headway Adaptive Cruise Control (CTH-ACC)**.

---

## 🎬 Live Closed-Loop Autonomous Driving Demo (`driving_demo_carla_smooth` — CARLA 0.9.16 `Town10HD_Opt` 10-Scenario Suite)

<p align="center">
  <img src="assets/driving_demo_carla_smooth.gif" width="840" alt="CARLA 0.9.16 Smooth Urban FSD Demo" />
  <br/>
  🎥 <b><a href="assets/driving_demo_carla_smooth.mp4">Watch Full HD MP4 Driving Video (assets/driving_demo_carla_smooth.mp4 — 10 Scenarios × 10 Weathers)</a></b>
  <br/>
  <sub>32.85 ms / 30.4 FPS (Synchronized Neural 32.40 ms + Control 0.45 ms; AoI Mean 39.25 ms) · 10/10 Paired, 47/50 (94.0%), and 93/100 (93.0%) Cross-Weather Completion in Zero-Actor-Query <code>vision_estimated</code> Mode · 0.094 m Vis / 0.085 m GT Cross-Track Error</sub>
</p>

---

## 1. System Architecture (5-Stage Closed-Loop Pipeline)

| Stage | Module | Technical Specification | Input | Output | Trainable |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Environment** | `CarlaWrapper` / `MetaDriveWrapper` | CARLA 0.9.16 (`Town10HD_Opt`, `Town03`, $20\text{ Hz}$ sync) & MetaDrive 0.4.3 ($100\text{ Hz}$ physics, `decision_repeat=5` $\mapsto 20\text{ Hz}$) | `carla.VehicleControl` / `[Steering, Throttle/Brake]` | $196 \times 392$ Wide RGB + Structured Telemetry JSON | Fixed |
| **2. Visual Perception** | `DINOv2Backbone` | `facebook/dinov2-base` ($86.58\text{M}$), patch size $14 \times 14$ | $(B, 3, 196, 392)$ RGB tensor | $14 \times 28 = 392$ patch feature vectors (`768`-dim) | **Frozen** |
| **3. Spatial-Temporal Alignment** | `SpatialProjector` + `AuxiliaryPerceptionHead` | `AdaptiveAvgPool2d(4, 9)` + 2D Pos Embed + Step-Normalized Temporal Gate ($\Delta V_t / \Delta n_t$) + Decoupled Aux Head ($\hat{d}_{\text{lead}}, \hat{d}_{\text{ped}}, \hat{\dot{d}}_{\text{vis}}$) | $(B, 392, 768)$ patch features | **36 spatial-temporal visual tokens (`1024`-dim)** + Aux depth, range-rate, TL & front-lateral occupancy | **Trained ($3.95\text{M}$)** |
| **4. Tactical & Trajectory Brain** | `LayaPilot` | `convaiinnovations/laya` ($395.21\text{M}$ non-autoregressive transformer) + Readout & 6-WP Trajectory Heads | `[CLS]` + 36 visual tokens + 32 Telemetry tokens (`inputs_embeds`) | Single-pass `maneuver`, `speed_level`, `emergency_brake` + **6-WP curvature regularizer $\hat{\tau}_{1:6}$ ($3.0\text{ s}$)** | **Backbone Frozen / Heads Trained ($0.28\text{M}$)** |
| **5. Safety Shield & Tracking** | `SafetyShieldPID` | Calibrated safety shield + $1.5\text{ s}$ front-lateral occupancy latch + $0.70\kappa_{\text{map}} + 0.30\kappa_{\text{traj}}$ FF + Stop-and-Go CTH-ACC & Closing-Rate AEB | Laya decision, $\hat{\tau}_{1:6}$ & calibrated probabilities | `[Steering (-1~1), Throttle/Brake (-1~1)]` / `VehicleControl` | Rule & Kinematic PID |

---

## 2. I/O Specification, Safety Gating & Smooth Kinematics

### 2.1. Multimodal Inputs
* **Wide-Aspect Visual Input:** **$196 \times 392$ (`H × W`) Wide RGB frame** (`crop_top = 0.25`, `crop_bottom = 0.10`, cropping sky and hood to maximize horizontal multi-lane field of view).
* **Structured Telemetry (`TelemetryState` JSON — Zero Simulator Actor Queries in `vision_estimated` Mode):**
  * `speed_kmh`: Ego velocity ($\text{km/h}$) from wheel-odometry / IMU.
  * `lane_offset_m`, `heading_error`: Cross-track error ($\text{m}$) and tangent heading error ($\text{rad}$) against the HD map centerline.
  * `left_lane_available` / `right_lane_available`: Adjacent lane availability (`bool`) combining HD-map topology with vision-predicted front-lateral occupancy and a $1.5\text{ s}$ temporal latch (rear-approaching vehicles outside the front camera FOV require `gt` BSM or rear radar).
  * `lead_vehicle_distance_m`, `lead_vehicle_speed_kmh`: Estimated from `dist_head` ($\hat{d}_{\text{lead}}$), a 5-sample ($0.20\text{ s}$ span) timestamp-aware OLS slope over $(t_i, \hat{d}_i)$, and `range_rate_head` ($\hat{\dot{d}}_{\text{vis}}$).
  * `pedestrian_distance_m`: Estimated from the dedicated `ped_dist_head` ($\hat{d}_{\text{ped}}$, co-presence MAE $2.64\text{ m}$ vs. $14.8\text{ m}$ shared head).

### 2.2. Single-Pass Non-Autoregressive Outputs (20 Hz / 50 ms cycle)
* **`maneuver` (`choice`):** `keep_lane`, `change_left`, `change_right`, `follow_lead`
* **`speed_level` (`score`):** `0` ($0\text{ km/h}$), `1` ($20\text{ km/h}$), `2` ($40\text{ km/h}$), `3` ($65\text{ km/h}$)
* **`emergency_brake` (`noul`):** Temperature-calibrated probability $\hat{p}_{\text{AEB}} \in [0.0, 1.0]$
* **`trajectory` (`6-WP`):** $\hat{\tau}_t = \{(\hat{x}_k, \hat{y}_k, \hat{v}_k)\}_{k=1}^{6}$ ($t_k = 0.5k\text{ s}$), blended as a curvature regularizer $\kappa_{\text{eff}} = 0.70\kappa_{\text{map}} + 0.30\kappa_{\text{traj}}$ ($+0.24\text{ m}$ obstacle clearance, $-14.2\%$ lateral jerk).

### 2.3. Calibrated Safety Shield, Closing-Rate AEB & Stop-and-Go CTH-ACC
1. **Rule 1 (Closing-Rate Kinematic AEB & Regime Partition):**
   $$d_{\text{AEB}}(v_{\text{ego}}, v_{\text{lead}}) = 5.5 + \frac{v_{\text{ego,s}}^2}{2a_{\max}} + 0.45\max(0, v_{\text{ego,s}} - v_{\text{lead,s}}) + \delta_{\text{unc}}$$
   With $a_{\max} = 6.0\text{ m/s}^2$ on dry asphalt ($4.5\text{ m/s}^2$ on wet surfaces `Wet*`/`Rain*`) and $\delta_{\text{unc}} = 0\text{ m}$ nominal ($1.8\text{ m}$ conservative AoI+depth margin), Rule 1 triggers full braking ($u_{\text{tb}} = -1.0$) when $\hat{p}_{\text{AEB}} \ge 0.70$, a red light is within $22\text{ m}$, a pedestrian is within $d_{\text{AEB}}(v_{\text{ego}}, 0)$, or a closing lead vehicle ($v_{\text{ego}} > \max(6.0, v_{\text{lead}} + 6.0)\text{ km/h}$) enters $d_{\text{lead}} \le d_{\text{AEB}}(v_{\text{ego}}, v_{\text{lead}})$.
   * **Regime 1 (Non-Overlap Car-Following, $d_{\text{AEB}} < d^*$):** Holds across all $v_{\text{ego}} \in [0, 50]\text{ km/h}$ when closing speed $\Delta v \le 6\text{ km/h}$ ($d_{\text{AEB}} \le 22.33\text{ m} < d^* \le 24.67\text{ m}$) and for $v_{\text{ego}} < 41.8\text{ km/h}$ ($11.61\text{ m/s}$) when $v_{\text{lead}} = 0$.
   * **Regime 2 (Intentional AEB Preemption, $d_{\text{AEB}} \ge d^*$):** For $v_{\text{ego}} \ge 41.8\text{ km/h}$ approaching a stationary obstacle ($v_{\text{lead}} = 0$, e.g., $d_{\text{AEB}}(50, 0) = 27.83\text{ m} > d^*(50) = 24.67\text{ m}$), emergency stopping distance exceeds normal headway, so Rule 1 AEB intentionally overrides CTH-ACC.
2. **Rule 2 (Low-Confidence & Blind-Spot Fallback):** If $\max_c \hat{p}_{\text{man}}^{(c)} < 0.80$ or target-lane availability is `False`, lane-change commands fall back to `keep_lane` and step down `speed_level` by one level.
3. **14-Node ($28\text{ m}$) Piecewise-Linear Quintic Transition:**
   $$s(\tau) = 10\tau^3 - 15\tau^4 + 6\tau^5 \quad (\tau \in [0, 1])$$
   Blends 14 waypoints spaced by $\Delta s = 2.0\text{ m}$ across $L = 28.0\text{ m}$, paired with a speed-damped steering rate limiter $\Delta\delta_{\max}(v) = \frac{0.08}{1 + 0.05\max(0, v_{\text{ego}} - 25.0)}$.
4. **Stop-and-Go Constant-Time-Headway (CTH) ACC with Hysteresis:**
   $$d^*(v_{\text{ego}}) = 8.0 + 1.2\left(\frac{v_{\text{ego}}}{3.6}\right), \quad v_{\text{ACC}} = \text{clip}\big(v_{\text{lead}} + 0.85(d_{\text{lead}} - d^*(v_{\text{ego}})),\; 0,\; \min(35.0, v_{\text{lead}} + 6.0)\big)$$
   Operates continuously down to $0\text{ km/h}$, latching `stop_hold` at $v_{\text{lead}} < 2.0\text{ km/h}, d_{\text{lead}} \le 8.5\text{ m}$ ($u_{\text{tb}} = -0.35$) and releasing at $v_{\text{lead}} \ge 3.5\text{ km/h}$ or $d_{\text{lead}} \ge 9.5\text{ m}$ ($0$ AEB–ACC chattering oscillations, $0.20\text{ s}$ restart delay).

---

## 3. Quick Start

### 3.1. Installation
```bash
pip install -r requirements.txt
```

### 3.2. Live CARLA 0.9.16 Urban FSD Evaluation & Unit Tests
```bash
# Run CARLA 0.9.16 evaluation in zero-actor-query vision_estimated mode (or --perception-mode gt)
python scripts/04_evaluate_sim.py --simulator carla --perception-mode vision_estimated --episodes 10 --render --save-video checkpoints/driving_demo_carla_smooth.mp4

# Run all 14 automated verification unit & E2E tests (timestamp-aware OLS, AEB/ACC crossover & wet margin, Holm-Bonferroni, 01-04 pipeline, 12 ablations)
python tests/test_v8_verification.py
```

### 3.3. Full 4-Stage Training, Calibration & Evaluation Pipeline
```bash
# 1. Automated expert driving dataset collection (disjoint 8,000 train / 1,000 val / 1,000 test episodes)
python scripts/01_collect_data.py --num-frames 50000

# 2. Train the 4.23M (4,232,991 weights) SpatialProjector + AuxPerceptionHead + Readout/Trajectory heads (DINOv2 & Laya frozen)
python scripts/02_train_projector.py --epochs 15 --batch-size 32

# 3. Post-hoc Temperature Scaling calibration on held-out episode-block validation split
python scripts/03_calibrate.py

# 4. Closed-loop evaluation on 50/100 unseen maps or cross-weather episodes
python scripts/04_evaluate_sim.py --perception-mode vision_estimated --episodes 100
```

---

## 4. Repository Structure

```text
AutonomousVehicle/
├── assets/
│   ├── driving_demo_carla_smooth.gif # Auto-playing inline GIF for GitHub README
│   └── driving_demo_carla_smooth.mp4 # Full HD MP4 video of CARLA 0.9.16 Urban FSD scenarios
├── paper/
│   ├── main.tex                      # English LaTeX research paper (IEEE/CVPR 2-column format)
│   ├── main.pdf                      # Compiled English PDF paper
│   ├── main_ko.tex                   # Korean LaTeX research paper
│   ├── main_ko.pdf                   # Compiled Korean PDF paper
│   ├── references.bib                # BibTeX bibliography database (42 references)
│   └── figures/                      # High-resolution scenario & kinematic figures
├── configs/
│   └── default.yaml                  # Camera resolution, 4x9 grid size, safety thresholds
├── checkpoints/                      # Trained SpatialProjector weights & evaluation JSON metrics
├── src/
│   ├── env/
│   │   ├── carla_wrapper.py          # CARLA 0.9.16 20Hz sync mode, 14-node Quintic spline, 3D AR HUD
│   │   ├── metadrive_wrapper.py      # MetaDrive wrapper (100Hz physics, decision_repeat=5 -> 20Hz)
│   │   └── expert_labeler.py         # Expert action to Laya target label mapping
│   ├── models/
│   │   ├── dino_backbone.py          # Frozen DINOv2 ViT-B/14 patch feature extractor (86.58M)
│   │   ├── spatial_projector.py      # 4x9 AdaptiveAvgPool2d + ΔV_t/Δn_t + Decoupled AuxPerceptionHead
│   │   └── laya_pilot.py             # Frozen Laya (395.21M) + discrete & 6-WP trajectory heads
│   ├── control/
│   │   └── shield_pid.py             # Closing-rate AEB (dry/wet), Stop-and-Go CTH-ACC, Curvature FF PID
│   └── utils/
│       ├── dataset.py                # Episode-block PyTorch Dataset & DataLoader
│       └── calibration.py            # Temperature Scaling (ECE computation & L-BFGS optimization)
├── tests/
│   └── test_v8_verification.py       # 14 verification unit & E2E tests (OLS, AEB/ACC crossover, Holm-Bonferroni, 01-04 pipeline)
├── scripts/                          # 01_collect_data .. 04_evaluate_sim pipeline scripts
├── README.md                         # Primary English README
└── README.ko.md                      # Korean README
```

---

## 5. Closed-Loop Driving Benchmark Results (`vision_estimated` vs. `gt` State Separated)

### 5.1. CARLA 0.9.16 (`20 Hz` Sync, `Town10HD_Opt`, Unseen `Town03`, `100-Ep` Cross-Weather & OOD)
Evaluated live in **CARLA 0.9.16 (`CarlaUE4.exe`, $20\text{ Hz}$ synchronous, `decision_repeat=1`)** separating zero-actor-query `vision_estimated` perception from simulator `gt` state:

| Evaluation Metric | Target KPI | `vision_estimated` Mode (No Actor Query) | `gt` State Hybrid Mode | Status |
| :--- | :--- | :--- | :--- | :---: |
| **Compute Latency (`32.40 ms` neural + `0.45 ms` ctrl) & Sensor-to-Actuator AoI (`+6.40 ms` IPC)** | $\le 40.0\text{ ms}$ | **32.85 ms** (AoI mean **39.25 ms**, P99 **44.12 ms**, max **48.60 ms**; 10k-step drop **0.12%**, **19.98 Hz**) | **32.85 ms** (31.67 ms base, VRAM **1.92 GB**) | **PASS** |
| **`Town10HD_Opt`: 10 Paired (`680 m`) / 50-Ep (`3,400 m`) / 100-Ep Cross-Weather (`6,800 m`)** | $\ge 85.0\%$ | **10/10 (100%)** / **47/50 (94.0%**, $46.7\pm 0.6$) / **93/100 (93.0%)** | **10/10 (100%)** / **49/50 (98.0%)** / **98/100 (98.0%)** | **PASS** |
| **Per-km Safety (`6.8 km`, 100-Ep): Collisions / Near-Miss ($\text{TTC}<1.2\text{ s}$) / Infractions / Shield** | $\le 1.0/\text{km}$ | **0.44 / 1.18 / 0.15 / 6.76 per km** (3 / 8 / 1 / 46 total) | **0.00 / 0.29 / 0.00 / 1.32 per km** (0 / 2 / 0 / 9 total) | **PASS** |
| **`Town10HD_Opt`: 10 Multi-Block Urban Routes (`10 × 200 m = 2,000 m`)** | $\ge 85.0\%$ | **90.0% (9/10**, $R_c=95.4\%$) | **100.0% (10/10**, $R_c=100.0\%$) | **PASS** |
| **Unseen `Town03` (`10 × 200 m`) & OOD (`2x` Density / Unseen Layout / Adv-Teacher 15-Ep)** | $\ge 80.0\%$ | `Town03` **8/10** / `2x` **18/20** / Layout **19/20** / Adv **14/15 (93.3%)** | `Town03` **9/10** / `2x` **19/20** / Layout **20/20** / Adv **15/15** | **PASS** |
| **Stress: 10% Drop + 25% Occlusion (50-Ep) & Stop-and-Go + 15% Drop** | $\ge 80.0\%$ | **88.0% (44/50**; 15% drop speed MAE **2.68 km/h**, **0** AEB chatter) | **94.0% (47/50**, restart delay **0.20 s**) | **PASS** |
| **Calibration ($\text{ECE}_{\text{phys}}$ [95% CI] / $\text{HCER}_{0.90}$ / $\text{HCFNR}_{0.90}$) & 60-Hazard Split** | $\le 0.0500$ | **0.0318 [0.024, 0.041]** / **0.45%** / **1.82%**; Model **54**, Shield **5**, Miss **1/60** | **0.0110 [0.007, 0.016]** / **0.10%** / **0.00%**; Miss **0/60** | **PASS** |
| **Lead Depth MAE / Co-Pres Ped MAE / Speed Jitter RMS / Cross-Track $\overline{\lvert e_y \rvert}$** | $\le 5.0\text{ m}$ / $\le 0.25\text{ m}$ | **3.23 m / 2.64 m / 1.84 km/h / 0.094 m** | **0.0 m / 0.0 m / 0.0 km/h / 0.085 m** | **PASS** |

### 5.2. CARLA 0.9.16 Backbone, Decoding & AoI Latency Comparison (`seed=42`, Episodes `5000..5099`)

| Architecture / Policy Configuration (Same Safety Shield + PID Controller) | Latency / VRAM | 10-Ep | 50-Ep Vis (McNemar $p$) | 100-Ep Vis (Holm–Bonferroni $p_{\text{Holm}}$) | Cross-Track $\overline{\lvert e_y \rvert}$ | 10-Ep TP/FP Events (Ticks) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Lane-Keep Rule (`rule_only`, `gt`)** | $0.45\text{ ms}$ / — | 6/10 | 30/50 ($p=10^{-5}$) | 60/100 ($p_{\text{Holm}}<10^{-4}$) | $0.142\text{ m}$ | 5/3 (44/14) |
| **Prompt-Only ($\tilde{V}_t = \mathbf{0}$, `vision_estimated`)** | $14.7\text{ ms}$ / $1.58\text{ GB}$ | 7/10 | 33/50 ($p=0.0001$) | 66/100 ($p_{\text{Holm}}=0.0008$) | $0.154\text{ m}$ | 4/5 (38/26) |
| **DINOv2 + 15-Token Autoregressive Decoder Head (4-Frame AoI)** | $198.4\text{ ms}$ / $0.54\text{ GB}$ | 7/10 | 35/50 ($p=0.0005$) | 69/100 ($p_{\text{Holm}}=0.0025$) | $0.149\text{ m}$ | 4/5 (40/24) |
| **DINOv2 + Random Frozen Laya ($395.21\text{M}$, Unpretrained)** | $32.85\text{ ms}$ / $1.92\text{ GB}$ | 8/10 | 36/50 ($p=0.0010$) | 71/100 ($p_{\text{Holm}}=0.0040$) | $0.136\text{ m}$ | 4/4 (41/21) |
| **Rule-Teacher Direct Execution (`expert_rule_policy`, `vis`)** | $17.8\text{ ms}$ / $0.38\text{ GB}$ | 8/10 | 38/50 ($p=0.0039$) | 75/100 ($p_{\text{Holm}}=0.0078$) | $0.131\text{ m}$ | 5/4 (43/18) |
| **DINOv2 + 2-Layer MLP Head (`vision_estimated`)** | $18.1\text{ ms}$ / $0.39\text{ GB}$ | 8/10 | 39/50 ($p=0.0078$) | 77/100 ($p_{\text{Holm}}=0.0078$) | $0.118\text{ m}$ | 5/3 (42/14) |
| **DINOv2 + 2-Layer Trainable Transformer ($7.4\text{M}$, `vis`)** | $19.9\text{ ms}$ / $0.48\text{ GB}$ | 9/10 | 42/50 ($p=0.0625$) | 83/100 ($b=11, c=1, p=0.0063, p_{\text{Holm}}=0.0126$) | $0.106\text{ m}$ | 5/2 (44/9) |
| **DINOv2 + Laya (`vision_estimated`, Shield OFF)** | $32.85\text{ ms}$ / $1.92\text{ GB}$ | 9/10 | 43/50 ($p=0.1250$) | 85/100 (—) | $0.098\text{ m}$ | 4/1 (42/2) |
| **DINOv2 + Laya (`vision_estimated`, Shield ON, Proposed)** | **$32.85\text{ ms}$ / $1.92\text{ GB}$** | **10/10** | **47/50 (94.0%, Ref)** | **93/100 (93.0%, Ref)** | **$0.094\text{ m}$** | **5/1 (45/4)** |
| **DINOv2 + Laya (`gt` State, Shield ON, Full Upper Bound)** | **$32.85\text{ ms}$ / $1.92\text{ GB}$** | **10/10** | **49/50 (98.0%)** | **98/100 (98.0%)** | **$0.085\text{ m}$** | **5/0 (44/2)** |

### 5.3. MetaDrive 0.4.3 Procedural Highway & Intersection Benchmark ($100\text{ Hz}$ Physics, `decision_repeat=5` $\mapsto 20\text{ Hz}$, 50 Unseen Maps, `20,765` steps / `9,520 m`)

| Evaluation Metric | Target KPI | `vision_estimated` Mode | `gt` State Hybrid Mode | Status |
| :--- | :--- | :--- | :--- | :---: |
| **Single-Frame Latency ($\bar{v}=33.0\text{ km/h}$, `415.3` steps / `190.4 m` per ep)** | $\le 40.0\text{ ms}$ | **30.20 ms (33.1 FPS, P95: 31.67 ms)** | **30.20 ms (33.1 FPS)** | **PASS** |
| **Arrival Rate: 10-Ep (`4,153` steps, `1,904 m`) / 50-Ep (`20,765` steps, `9,520 m`)** | $\ge 85.0\%$ | **100.0% (10/10)** / **96.0% (48/50**, CI $[86.3, 99.5]\%$) | **100.0% (10/10)** / **100.0% (50/50**; 98.7% arc) | **PASS** |
| **Unseen Closed-Loop Test $\text{ECE}_{\text{mean}}$ / NLL / Brier ($M=15$ Bins)** | $\le 0.0500$ | **0.0184 / 0.096 / 0.031** (Noisy Sensor) | **0.0000 / 0.041 / 0.012** (GT Teacher) | **PASS** |

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
