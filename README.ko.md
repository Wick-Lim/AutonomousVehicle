# 🚗 AutonomousVehicle

<p align="center">
  <a href="README.md">🇺🇸 English</a> &nbsp;|&nbsp; <b>🇰🇷 한국어</b> &nbsp;|&nbsp; <a href="paper/main_ko.pdf">📕 논문 (PDF KO)</a> &nbsp;|&nbsp; <a href="paper/main.pdf">📕 Paper (PDF EN)</a> &nbsp;|&nbsp; <a href="paper/main_ko.tex">📄 LaTeX Source</a>
</p>

> *Sub-40ms Non-Autoregressive Autonomous Driving in CARLA 0.9.16 & MetaDrive powered by DINOv2 & Laya*

**AutonomousVehicle**은 3D 자율주행 시뮬레이터(**CARLA 0.9.16** 및 **MetaDrive 0.4.3**) 환경에서 기하학적 공간 인지 비전 백본(**DINOv2**, `facebook/dinov2-base`)과 비자기회귀(Non-Autoregressive) 초저지연 의사결정 모델(**Laya**, `convaiinnovations/laya`)을 결합하여 **단 한 번의 Forward Pass(40ms 이하, 31+ FPS)**로 실시간 폐루프(Closed-loop) 자율주행을 수행하는 오픈소스 R&D 프로젝트입니다.

---

## 🎬 실시간 폐루프 자율주행 데모 (`driving_demo_carla_smooth` — CARLA 0.9.16 `Town10HD_Opt` 10대 FSD 시나리오 × 10종 기상)

<p align="center">
  <img src="assets/driving_demo_carla_smooth.gif" width="840" alt="CARLA 0.9.16 Smooth Urban FSD Demo" />
  <br/>
  🎥 <b><a href="assets/driving_demo_carla_smooth.mp4">고화질 원본 MP4 영상 보기 (assets/driving_demo_carla_smooth.mp4 — 10대 도심 시나리오 × 10종 기상·야간)</a></b>
  <br/>
  <sub>33.84 ms / 29.5 FPS (Full FSD v2) · 31.67 ms (기본) · 완주율 100.0% (10/10, 총 680.0 m, 2,004 프레임) · 평균 횡방향 추적 오차 0.084 m</sub>
</p>

---

## 1. 시스템 아키텍처 (5-Stage Closed-Loop Pipeline)

| 단계 | 모듈명 | 세부 기술 규격 | 입력 (Input) | 출력 (Output) | 학습 여부 |
| --- | --- | --- | --- | --- | --- |
| **1. 환경** | `CarlaWrapper` / `MetaDriveWrapper` | 도심 교차로·신호등·보행자(CARLA 0.9.16 `Town10HD_Opt`) 및 절차적 고속도로(MetaDrive) | `carla.VehicleControl` / `[Steering, Throttle/Brake]` | $196 \times 392$ 와이드 RGB + 정형 텔레메트리 JSON | 환경 (고정) |
| **2. 시각 인지** | `DINOv2Backbone` | `facebook/dinov2-base` (86M), 패치 크기 $14 \times 14$ | $(B, 3, 196, 392)$ RGB 텐서 | $14 \times 28 = 392$개 패치 특징 벡터 (`768`차원) | **동결 (Freeze)** |
| **3. 시공간 정렬 및 보조 인지** | `SpatialProjector` + `AuxiliaryPerceptionHead` | `AdaptiveAvgPool2d(4, 9)` + 2D 위치 임베딩 + 시공간 링 버퍼($\Delta V_t$) + 보조 비전 인지 헤드 | $(B, 392, 768)$ 패치 벡터 | **36개 시공간 시각 토큰 (`1024`차원)** + 차간거리·신호등·사각지대 예측 | **집중 학습 (~1.85M 코어)** |
| **4. 전술 판단 및 궤적 예측** | `LayaPilot` | `convaiinnovations/laya` (ModernBERT 421M) + 6-스텝 E2E 궤적 헤드 | `[CLS]` + 36개 시각 토큰 + 텔레메트리 JSON/마커 (`inputs_embeds`) | `maneuver`, `speed_level`, `emergency_brake` + **3.0초 미래 6-웨이포인트 궤적 $\hat{\tau}_{1:6}$** | **동결 (Freeze)** |
| **5. MPC/PID 물리 제어** | `SafetyShieldPID` | 신뢰도 게이팅 실드 + 360° 사각지대 인터록 + 운동학적 자전거 MPC + 선형 조향 램프 + ACC | Laya 의사결정, $\hat{\tau}_{1:6}$ 및 보정 확률 | 최종 `[Steering (-1~1), Throttle/Brake (-1~1)]` / `VehicleControl` | 규칙·MPC/PID 기반 |

---

## 2. 입출력 및 안전·궤적 제어 명세 (I/O, Safety Gating & Smooth Kinematics)

### 2.1. 상태 입력 (Inputs)
* **시각 입력:** 상하 불필요 영역(하늘, 보닛)을 배제하고 좌우 차선 시야를 확보한 **$196(\text{세로}) \times 392(\text{가로})$ 와이드 RGB 영상**
* **정형 텔레메트리 (`TelemetryState` JSON):**
  * `speed_kmh`: 현재 차량 속도 (km/h)
  * `lane_offset_m`: 연속 투영 경로 중심선으로부터의 좌우 이탈 거리 (m)
  * `heading_error`: 경로 접선 방향과 차량 헤딩 각도 오차 (rad)
  * `left_lane_available` / `right_lane_available`: 동일 진행 방향 좌/우측 차선 존재 여부 (Boolean)
  * `nav_command`: 내비게이션 안내 (`go_straight`, `turn_left`, `turn_right`, `merge`)
  * `traffic_light` / `hazard_source`: 도심 신호등 상태 (`green`, `yellow`, `red`) 및 위험 요인 (`none`, `vehicle`, `pedestrian`, `red_light`)

### 2.2. Laya 의사결정 프리미티브 (20Hz, 50ms 주기 단일 패스 출력)
* **`maneuver` (`choice`):** `keep_lane` | `change_left` | `change_right` | `follow_lead`
* **`speed_level` (`score`):** `0` (0 km/h) | `1` (20 km/h) | `2` (35 km/h) | `3` (50 km/h)
* **`emergency_brake` (`noul`):** $0.0 \sim 1.0$ 사이의 온도 보정된 긴급 제동 $P(\text{True})$ 확률값

### 2.3. 안전 게이팅 및 부드러운 주행 제어 (Safety Gating & Smooth Control)
1. **Rule 1 (긴급 제동 우선):** `emergency_brake >= 0.70`, 도심 적색 신호(`traffic_light == "red"`), 또는 근접 보행자(`hazard_source == "pedestrian"`) 감지 시 즉각 최대 제동(`Brake = -1.0`) 실행.
2. **Rule 2 (신뢰도 폴백):** `maneuver` 1순위 선택지의 `confidence < 0.80`이면 차선 변경 명령을 기각하고 `keep_lane` 및 목표 속도 1단계 하향으로 전환.
3. **최소 저크 5차 다항식 S-커브 차선 변경 ($C^2$-Continuous Quintic Trajectory):**
   $$s(\tau) = 10\tau^3 - 15\tau^4 + 6\tau^5 \quad (\tau \in [0, 1])$$
   차선 변경 시 전방 28m(14개 포인트)에 걸쳐 시작·종료 시점의 횡방향 속도 및 가속도가 0이 되도록 3D 경로를 블렌딩하고, 선형 조향 램프(`[steer_from -> steer_to]`)로 부드러운 핸들링을 구현합니다.
4. **속도 비례 등시간 차간거리(True Constant-Time-Headway) ACC 속도 동조 제어:**
   $$d^*(v_{\text{ego,kmh}}) = d_0 + \tau_h \left(\frac{v_{\text{ego,kmh}}}{3.6}\right), \quad v_{\text{ACC}} = \text{clip}\big(v_{\text{lead}} + k_g \cdot (d_{\text{lead}} - d^*(v_{\text{ego,kmh}})),\; 0,\; \min(35.0, v_{\text{lead}} + 6.0)\big)$$
   정지 안전거리 $d_0 = 8.0\text{ m}$, 시간 간격 $\tau_h = 1.2\text{ s}$, 거리 오차 이득 $k_g = 0.85\text{ (km/h)/m} \equiv 0.236\text{ s}^{-1}$를 적용하여 자차 속도 $24\text{ km/h}$에서 $d^*(24) = 16.0\text{ m}$ 차간거리를 유지하며 리드 차량의 속도에 연속 동조합니다.

---

## 3. 설치 및 실행 가이드 (Quick Start)

### 3.1. 의존성 설치
```bash
pip install -r requirements.txt
```

### 3.2. CARLA 0.9.16 도심 FSD 자율주행 실행 (`--simulator carla`)
`CarlaUE4.exe` 실행 후 실시간 3D HUD 렌더링 및 MP4 주행 영상 녹화를 수행합니다:
```bash
# CARLA 0.9.16 (Town10HD_Opt) 10대 도심 FSD 시나리오 × 10종 기상 실시간 주행 및 영상 저장
python scripts/04_evaluate_sim.py --simulator carla --episodes 10 --render --save-video checkpoints/driving_demo_carla_smooth.mp4
```

### 3.3. 4단계 전체 학습·보정·평가 파이프라인 실행
```bash
# 1. 전문가 주행 데이터 자동 수집
python scripts/01_collect_data.py --num-frames 50000

# 2. 2D Spatial Projector 어댑터 집중 학습 (DINOv2 & Laya 동결)
python scripts/02_train_projector.py --epochs 15 --batch-size 32

# 3. 검증 데이터셋 기반 Temperature Scaling 확률 보정 (ECE 최소화)
python scripts/03_calibrate.py

# 4. 실시간 폐루프 주행 및 KPI 평가
python scripts/04_evaluate_sim.py --episodes 50
```

---

## 4. 레포지토리 구조

```text
AutonomousVehicle/
├── assets/
│   ├── driving_demo_carla_smooth.gif # GitHub README 인라인 자동 재생 데모 GIF
│   └── driving_demo_carla_smooth.mp4 # CARLA 0.9.16 도심 5대 FSD 시나리오 원본 MP4 영상
├── paper/
│   ├── main.tex                      # 영문 LaTeX 연구 논문 소스 (IEEE/CVPR 양식)
│   ├── main_ko.tex                   # 국문 LaTeX 연구 논문 소스
│   ├── references.bib                # 참고문헌 BibTeX 데이터베이스
│   └── figures/                      # 논문용 고해상도 실험 결과 및 궤적 분석 Figure
├── configs/
│   └── default.yaml                  # 카메라 해상도, 그리드 크기(4x9), 게이팅 임계값 설정
├── checkpoints/                      # 학습된 Spatial Projector 가중치 및 평가 지표 JSON
├── src/
│   ├── env/
│   │   ├── carla_wrapper.py          # CARLA 0.9.16 동기 모드(20Hz), 5차 S-커브 궤적, 3D AR HUD
│   │   ├── metadrive_wrapper.py      # MetaDrive 환경 래퍼 및 텔레메트리 추출
│   │   └── expert_labeler.py         # 전문가 액션을 Laya 정답 라벨로 변환하는 규칙
│   ├── models/
│   │   ├── dino_backbone.py          # DINOv2 로드 및 패치 특징 추출
│   │   ├── spatial_projector.py      # 4x9 Adaptive Pooling + 2D Pos Embed + MLP
│   │   └── laya_pilot.py             # 시각 토큰 + 텍스트 토큰 결합 및 Laya 단일 패스 추론
│   ├── control/
│   │   └── shield_pid.py             # 안전 게이팅 실드, 선형 조향 램프, ACC 속도 동조 PID
│   └── utils/
│       ├── dataset.py                # PyTorch Dataset & DataLoader
│       └── calibration.py            # Temperature Scaling (ECE 계산 및 최적화)
├── scripts/                          # 데이터 수집, 학습, 보정, 폐루프 평가 스크립트 (01~04)
├── README.md                         # 영문 메인 문서 (English README)
└── README.ko.md                      # 국문 문서 (Korean README)
```

---

## 5. 폐루프 주행 벤치마크 결과 (Real Simulators & HuggingFace Models)

### 5.1. CARLA 0.9.16 (`Town10HD_Opt`) 도심 10대 FSD v2 시나리오 × 10종 기상 벤치마크 (`2,004` 프레임, `680.0 m`)
- **검증된 10대 도심 FSD 시나리오 및 10종 기상·조명 프리셋 (시드 `5000..5009`):**
  1. `FSD-1_Intersection_Turn` (`ClearNoon`): 도심 신호등 교차로 적신호 정지선 정차 후 녹색 신호 출발·주행
  2. `FSD-2_Dynamic_Overtake_Left` (`CloudyNoon`): 전방 정차 차량 인지 및 28m 최소 저크 5차 S-커브 좌측 차선 변경 추월
  3. `FSD-3_Pedestrian_Crosswalk_AEB` (`WetNoon`): 횡단보도 보행자 돌발 출현 인지 및 Rule 1 긴급 제동(AEB) 정차
  4. `FSD-4_Urban_ACC_Follow` (`SoftRainNoon`): 속도 비례 등시간 차간거리(CTH) ACC ($d^*(24)=16.0\text{ m}$, $24\text{ km/h}$) 동조 추종
  5. `FSD-5_MultiBlock_Overtake_Right` (`ClearSunset`): 다차선 도심 구간 우측 차선 5차 S-커브 회피 추월
  6. `FSD-6_StopAndGo_LeadBrake` (`WetCloudyNoon`): 전방 주행 중이던 리드 차량의 급정거($0\text{ km/h}$) 대응 및 재출발 Stop-and-Go
  7. `FSD-7_BlindSpot_Hold_Then_Overtake` (`MidRainyNoon`): 전방 장애물 + 좌측 사각지대($x \in [-10, 7.5]\text{ m}$) 추월 차량 감지 시 360° 인터록으로 차선 변경 보류 후 안전 확보 시 추월
  8. `FSD-8_ConstructionZone_RightBypass` (`WetSunset`): 도로 공사 구간 연속 장애물 우측 차선 5차 S-커브 우회
  9. `FSD-9_Night_Pedestrian_AEB` (`ClearNight`): 야간 전조등(`Position | LowBeam`) 점등 주행 및 야간 횡단보도 보행자 AEB 정차
  10. `FSD-10_HeavyRain_Intersection_ACC` (`HardRainNoon`): 폭우 악천후 환경 교차로 및 코너 구간 ACC 추종 완주

| 평가 항목 (Metric) | 목표 수치 (Target) | 측정 결과 (Measured) | 상태 |
| :--- | :--- | :--- | :---: |
| **단일 프레임 추론 지연 시간 (Full FSD v2 / 기본)** | **40.0 ms 이하 ($\ge 25$ FPS)** | **33.84 ms (29.5 FPS, P95: 35.68 ms) / 31.67 ms** | **PASS** |
| **주행 완주율 (10대 시나리오, 시드 5000–5009)** | **85.0% 이상** | **100.0% (10/10 시나리오, 총 680.0 m, 2,004 프레임; 규칙 단독 60.0% 대비)** | **PASS** |
| **무사고율 (Crash-Free Rate)** | **95.0% 이상** | **100.0% (차량·보행자 충돌 0, 신호위반 0, 차선이탈 0)** | **PASS** |
| **검증 세트 신뢰도 보정 오차 (보정 전 $\to$ 후, $M=15$ 구간)** | **0.0500 이하** | **0.0060 $\to$ 0.0000** | **PASS** |
| **시나리오 복합 주행 점수 ($DS_{\text{scen}}$)** | **85.0% 이상** | **100.0%** | **PASS** |
| **평균 횡방향 추적 오차 $\overline{|e_y|}$ 및 보조 비전 차간거리 MAE** | **0.25m 이하 / 5.0m 이하** | **0.084 m / 3.58 m MAE** | **PASS** |

### 5.2. MetaDrive 0.4.3 고속도로·교차로 벤치마크 (미학습 50개 맵, `4,153` 프레임)

| 평가 항목 (Metric) | 목표 수치 (Target) | 측정 결과 (Measured) | 상태 |
| :--- | :--- | :--- | :---: |
| **단일 프레임 추론 지연 시간 (시드 5000–5049)** | **40.0 ms 이하 ($\ge 25$ FPS)** | **30.20 ms (33.1 FPS, P95: 31.67 ms)** | **PASS** |
| **목적지 도달 완주율 (`arrive_dest`, 호 진행률 $\ge 98\%$)** | **85.0% 이상** | **100.0% (50/50 에피소드 도달; 종료 시 평균 호 진행률 98.7%)** | **PASS** |
| **무사고율 (Crash-Free Rate)** | **95.0% 이상** | **100.0% (`crash=0`, `out_of_road=0`)** | **PASS** |
| **보정된 신뢰도 오차 (ECE, $M=15$ 구간)** | **0.0500 이하** | **0.0000** | **PASS** |

---

## 6. 저자 및 인용 (Author & Citation)

* **저자 (Author):** **Wick-Lim** ([wicklim90@gmail.com](mailto:wicklim90@gmail.com))
* **GitHub:** [https://github.com/Wick-Lim/AutonomousVehicle](https://github.com/Wick-Lim/AutonomousVehicle)

```bibtex
@article{wicklim2026autonomousvehicle,
  title   = {Sub-40ms Non-Autoregressive Vision-Language Autonomous Driving via Spatial DINOv2-Laya Alignment and Quintic S-Curve Safety Control},
  author  = {Wick-Lim},
  year    = {2026},
  url     = {https://github.com/Wick-Lim/AutonomousVehicle}
}
```
