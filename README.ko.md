# 🚗 AutonomousVehicle

<p align="center">
  <a href="README.md">🇺🇸 English</a> &nbsp;|&nbsp; <b>🇰🇷 한국어</b> &nbsp;|&nbsp; <a href="paper/main_ko.pdf">📕 논문 (PDF KO)</a> &nbsp;|&nbsp; <a href="paper/main.pdf">📕 Paper (PDF EN)</a> &nbsp;|&nbsp; <a href="paper/main_ko.tex">📄 LaTeX Source</a>
</p>

> *Sub-40ms Non-Autoregressive Autonomous Driving in CARLA 0.9.16 & MetaDrive powered by DINOv2 & Laya*

**AutonomousVehicle**은 3D 자율주행 시뮬레이터(**CARLA 0.9.16** 및 **MetaDrive 0.4.3**) 환경에서 동결된 자기지도 공간 인지 비전 백본(**DINOv2**, `facebook/dinov2-base`, $86.58\text{M}$)과 동결된 비자기회귀(Non-Autoregressive) 의사결정 트랜스포머(**Laya**, `convaiinnovations/laya`, $395.21\text{M}$)를 $4 \times 9$ 시공간 어댑터($4.23\text{M}$ 파라미터)로 결합하고, HD 맵 라우팅, 규칙 기반 안전 실드, 운동학적 경로 추종 제어기를 통합한 실시간 하이브리드 폐루프 자율주행 R&D 프로젝트입니다. **단 한 번의 동기화된 순전파+제어 패스($32.85\text{ ms}$, $30.4\text{ FPS}$; 기본 $31.67\text{ ms}$)** 안에 **14-웨이포인트($28\text{ m}$) 공간 5차 스플라인 전이**, **5-스텝($0.25\text{ s}$) 최소제곱(OLS) 선행차 속도 추정**, **선행차·보행자 분리형 거리 추정 헤드**, **속도 비례 등시간 차간거리(CTH) ACC**를 실행합니다.

---

## 🎬 실시간 폐루프 자율주행 데모 (`driving_demo_carla_smooth` — CARLA 0.9.16 `Town10HD_Opt` 10대 FSD 시나리오 × 10종 기상)

<p align="center">
  <img src="assets/driving_demo_carla_smooth.gif" width="840" alt="CARLA 0.9.16 Smooth Urban FSD Demo" />
  <br/>
  🎥 <b><a href="assets/driving_demo_carla_smooth.mp4">고화질 원본 MP4 영상 보기 (assets/driving_demo_carla_smooth.mp4 — 10대 도심 시나리오 × 10종 기상·야간)</a></b>
  <br/>
  <sub>32.85 ms / 30.4 FPS (신경망 32.40 ms + 제어기 0.45 ms) · 31.67 ms (기본) · 액터 쿼리 없는 <code>vision_estimated</code> 모드 10/10 쌍 및 47/50 (94.0%) 교차 기상 완주 · 평균 횡방향 오차 0.094 m (Vis) / 0.085 m (GT)</sub>
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
# CARLA 0.9.16 (Town10HD_Opt) 액터 쿼리 없는 vision_estimated 모드 실시간 주행 및 영상 저장 (--perception-mode gt 선택 가능)
python scripts/04_evaluate_sim.py --simulator carla --perception-mode vision_estimated --episodes 10 --render --save-video checkpoints/driving_demo_carla_smooth.mp4

# v8 수식·모듈 검증 단위 테스트 실행 (5-스텝 OLS 상대속도, 보행자 전용 거리 헤드, ΔV_t/Δn_t, AEB 사건 묶음)
python tests/test_v8_verification.py
```

### 3.3. 4단계 전체 학습·보정·평가 파이프라인 실행
```bash
# 1. 전문가 주행 데이터 자동 수집
python scripts/01_collect_data.py --num-frames 50000

# 2. 4.23M SpatialProjector + AuxPerceptionHead + 판독·궤적 헤드 공동 학습 (DINOv2 & Laya 동결)
python scripts/02_train_projector.py --epochs 15 --batch-size 32

# 3. 에피소드 블록 분할 검증 데이터셋 기반 Temperature Scaling 확률 보정
python scripts/03_calibrate.py

# 4. 미학습 50개 맵 / 교차 기상 에피소드 실시간 폐루프 평가
python scripts/04_evaluate_sim.py --perception-mode vision_estimated --episodes 50
```

---

## 4. 레포지토리 구조

```text
AutonomousVehicle/
├── assets/
│   ├── driving_demo_carla_smooth.gif # GitHub README 인라인 자동 재생 데모 GIF
│   └── driving_demo_carla_smooth.mp4 # CARLA 0.9.16 도심 FSD 시나리오 원본 MP4 영상
├── paper/
│   ├── main.tex                      # 영문 LaTeX 연구 논문 소스 (IEEE/CVPR 양식)
│   ├── main_ko.tex                   # 국문 LaTeX 연구 논문 소스
│   ├── references.bib                # 참고문헌 BibTeX 데이터베이스 (42편)
│   └── figures/                      # 논문용 고해상도 실험 결과 및 궤적 분석 Figure
├── configs/
│   └── default.yaml                  # 카메라 해상도, 그리드 크기(4x9), 게이팅 임계값 설정
├── checkpoints/                      # 학습된 Spatial Projector 가중치 및 평가 지표 JSON
├── src/
│   ├── env/
│   │   ├── carla_wrapper.py          # CARLA 0.9.16 동기 모드(20Hz), 14-노드 5차 스플라인, 3D AR HUD
│   │   ├── metadrive_wrapper.py      # MetaDrive 환경 래퍼 및 텔레메트리 추출
│   │   └── expert_labeler.py         # 전문가 액션을 Laya 정답 라벨로 변환하는 규칙
│   ├── models/
│   │   ├── dino_backbone.py          # 동결 DINOv2 ViT-B/14 로드 및 패치 특징 추출 (86.58M)
│   │   ├── spatial_projector.py      # 4x9 Adaptive Pooling + ΔV_t/Δn_t + 분리형 AuxPerceptionHead
│   │   └── laya_pilot.py             # 동결 Laya (395.21M) + 이산 및 6-WP 궤적 헤드
│   ├── control/
│   │   └── shield_pid.py             # 안전 게이팅 실드, 자전거 곡률 피드포워드, 조향 램프, CTH-ACC
│   └── utils/
│       ├── dataset.py                # 에피소드 블록 분할 PyTorch Dataset & DataLoader
│       └── calibration.py            # Temperature Scaling (ECE 계산 및 L-BFGS 최적화)
├── tests/
│   └── test_v8_verification.py       # 5-스텝 OLS, 보행자 전용 거리 헤드, ΔV_t/Δn_t, AEB 사건 단위 테스트
├── scripts/                          # 데이터 수집, 학습, 보정, 폐루프 평가 스크립트 (01~04)
├── README.md                         # 영문 메인 문서 (English README)
└── README.ko.md                      # 국문 문서 (Korean README)
```

---

## 5. 폐루프 주행 벤치마크 결과 (`vision_estimated` vs. `gt` 분리 표기)

### 5.1. CARLA 0.9.16 (`Town10HD_Opt` 및 미학습 `Town03`) 폐루프 벤치마크
시뮬레이터 액터 쿼리를 일절 사용하지 않는 `vision_estimated` 모드와 시뮬레이터 `gt` 상태를 분리 측정한 결과:

| 평가 항목 (Metric) | 목표 기준 | `vision_estimated` 모드 (액터 쿼리 없음) | 시뮬레이터 `gt` 상태 하이브리드 모드 | 상태 |
| :--- | :--- | :--- | :--- | :---: |
| **단일 프레임 지연 (`32.40 ms` 신경망 + `0.45 ms` 제어; `+6.40 ms` IPC 별도)** | **$\le 40.0\text{ ms}$** | **32.85 ms (30.4 FPS, P95: 33.96 ms)** | **32.85 ms (보조+궤적 생략 시 31.67 ms)** | **PASS** |
| **`Town10HD_Opt`: 10개 쌍 (`680 m`) / 50개 교차 기상 (`3,400 m`)** | **$\ge 85.0\%$** | **100.0% (10/10) / 94.0% (47/50, 3-시드 $46.7\pm 0.6$)** | **100.0% (10/10) / 98.0% (49/50)** | **PASS** |
| **`Town10HD_Opt`: 10개 다중 블록 도심 경로 (`200 m/ep`, `2,000 m`)** | **$\ge 85.0\%$** | **90.0% (9/10, $R_c=95.4\%$)** | **100.0% (10/10, $R_c=100.0\%$)** | **PASS** |
| **미학습 `Town03` 맵 (`10 × 200 m` 회전교차로·경사로, `2,000 m`)** | **$\ge 80.0\%$** | **80.0% (8/10, $R_c=91.2\%$, $\overline{|e_y|}=0.112\text{ m}$)** | **90.0% (9/10, $R_c=96.5\%$)** | **PASS** |
| **센서 스트레스 테스트: 10% 프레임 드롭 + 25% 패치 가림 (50-Ep)** | **$\ge 80.0\%$** | **88.0% (44/50, $\Delta V_t/\Delta n_t$ 및 래치 활성)** | **94.0% (47/50)** | **PASS** |
| **교사 $\text{ECE}_{\text{mean}}$ / 물리적 위험 $\text{ECE}_{\text{phys}}$ / $\text{HCER}_{0.90,\text{AEB}}$ / 물리적 위험 $\text{FNR}_{\text{AEB}}$** | **$\le 0.0500$** | **0.0142 / 0.0318 / 0.45% (전체 1.12%) / 1/48건 (2.1%)** | **0.0000 / 0.0110 / 0.10% / 0/48건** | **PASS** |
| **선행차 거리 MAE / 동시출현 보행자 MAE / 속도 떨림 RMS / $\overline{|e_y|}$** | **$\le 5.0\text{ m}$ / $\le 0.25\text{ m}$** | **3.23 m / 2.64 m / 1.84 km/h / 0.094 m** | **0.0 m / 0.0 m / 0.0 km/h / 0.085 m** | **PASS** |

### 5.2. MetaDrive 0.4.3 고속도로·교차로 벤치마크 ($\Delta t = 0.05\text{ s}$, 미학습 50개 맵, `20,765` 스텝 / `9,520 m`)

| 평가 항목 (Metric) | 목표 기준 | `vision_estimated` 모드 | 시뮬레이터 `gt` 상태 하이브리드 모드 | 상태 |
| :--- | :--- | :--- | :--- | :---: |
| **단일 프레임 추론 지연 시간 ($\bar{v}=33.0\text{ km/h}$, 에피소드당 `415.3` 스텝 / `190.4 m`)** | **$\le 40.0\text{ ms}$** | **30.20 ms (33.1 FPS, P95: 31.67 ms)** | **30.20 ms (33.1 FPS)** | **PASS** |
| **목적지 도달 완주율: 10-Ep (`4,153` 스텝, `1,904 m`) / 50-Ep (`20,765` 스텝, `9,520 m`)** | **$\ge 85.0\%$** | **100.0% (10/10) / 96.0% (48/50, CI $[86.3, 99.5]\%$)** | **100.0% (10/10) / 100.0% (50/50; 진행률 98.7%)** | **PASS** |
| **독립 폐루프 테스트 $\text{ECE}_{\text{mean}}$ / NLL / Brier ($M=15$ 구간)** | **$\le 0.0500$** | **0.0184 / 0.096 / 0.031 (센서 잡음)** | **0.0000 / 0.041 / 0.012 (결정론적 GT)** | **PASS** |

---

## 6. 저자 및 인용 (Author & Citation)

* **저자 (Author):** **Wick-Lim** ([wicklim90@gmail.com](mailto:wicklim90@gmail.com))
* **GitHub:** [https://github.com/Wick-Lim/AutonomousVehicle](https://github.com/Wick-Lim/AutonomousVehicle)

```bibtex
@article{wicklim2026autonomousvehicle,
  title   = {Real-Time Closed-Loop Driving with Frozen DINOv2-Laya Spatial Pooling and Quintic Path Tracking},
  author  = {Wick-Lim},
  year    = {2026},
  url     = {https://github.com/Wick-Lim/AutonomousVehicle}
}
```
