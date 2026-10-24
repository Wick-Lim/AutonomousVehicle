# 🚗 AutonomousVehicle

> *Sub-40ms Non-Autoregressive Autonomous Driving in MetaDrive powered by DINOv2 & Laya*

**AutonomousVehicle**은 경량 3D 주행 시뮬레이터(**MetaDrive**) 환경에서 기하학적 공간 인지 비전 백본(**DINOv2**, `facebook/dinov2-base`)과 비자기회귀(Non-Autoregressive) 초저지연 의사결정 모델(**Laya**, `convaiinnovations/laya`)을 결합하여 **단 한 번의 Forward Pass(40ms 이하, 25+ FPS)**로 실시간 폐루프(Closed-loop) 자율주행을 수행하는 오픈소스 R&D 프로젝트입니다.

---

## 1. 시스템 아키텍처 (5-Stage Closed-Loop Pipeline)

| 단계 | 모듈명 | 세부 기술 규격 | 입력 (Input) | 출력 (Output) | 학습 여부 |
| --- | --- | --- | --- | --- | --- |
| **1. 환경** | `MetaDriveWrapper` | 절차적 도로 생성 및 주변 트래픽 시뮬레이션 | `[Steering, Throttle/Brake]` | $196 \times 392$ 와이드 RGB + 정형 텔레메트리 JSON | 환경 (고정) |
| **2. 시각 인지** | `DINOv2Backbone` | `facebook/dinov2-base` (86M), 패치 크기 $14 \times 14$ | $(B, 3, 196, 392)$ RGB 텐서 | $14 \times 28 = 392$개 패치 특징 벡터 (`768`차원) | **동결 (Freeze)** |
| **3. 공간 정렬** | `SpatialProjector` | `AdaptiveAvgPool2d(4, 9)` + 2D 위치 임베딩 + 2-Layer MLP | $(B, 392, 768)$ 패치 벡터 | 가로 해상도가 보존된 **36개 공간 시각 토큰 (`1024`차원)** | **집중 학습 (~1.85M)** |
| **4. 전술 판단** | `LayaPilot` | `convaiinnovations/laya` (ModernBERT 421M) | `[CLS]` + 36개 시각 토큰 + 텔레메트리 JSON/마커 (`inputs_embeds`) | `maneuver`(`choice`), `speed_level`(`score`), `emergency_brake`(`noul`) 및 신뢰도 | **동결 (Freeze)** *(LoRA 옵션)* |
| **5. 물리 제어** | `SafetyShieldPID` | 신뢰도 게이팅 안전 실드 + 횡/종방향 PID 제어기 | Laya 의사결정 및 보정 확률 | 최종 `[Steering (-1~1), Throttle/Brake (-1~1)]` | 규칙 기반 (Rule-based) |

---

## 2. 입출력 및 안전 게이팅 명세 (I/O & Safety Gating)

### 2.1. 상태 입력 (Inputs)
* **시각 입력:** 상하 불필요 영역(하늘, 보닛)을 배제하고 좌우 차선 시야를 확보한 **$196(\text{세로}) \times 392(\text{가로})$ 와이드 RGB 영상**
* **정형 텔레메트리 (`TelemetryState` JSON):**
  * `speed_kmh`: 현재 차량 속도 (km/h)
  * `lane_offset_m`: 차선 중심선으로부터의 좌우 이탈 거리 (m)
  * `heading_error`: 차선 진행 방향과 차량 헤딩 각도 오차 (rad)
  * `left_lane_available` / `right_lane_available`: 좌/우측 차선 존재 여부 (Boolean)
  * `nav_command`: 내비게이션 안내 (`go_straight`, `turn_left`, `turn_right`, `merge`)

### 2.2. Laya 의사결정 프리미티브 (20Hz, 50ms 주기 단일 패스 출력)
* **`maneuver` (`choice`):** `keep_lane` | `change_left` | `change_right` | `follow_lead`
* **`speed_level` (`score`):** `0` (0 km/h) | `1` (20 km/h) | `2` (40 km/h) | `3` (65 km/h)
* **`emergency_brake` (`noul`):** $0.0 \sim 1.0$ 사이의 온도 보정된 긴급 제동 $P(\text{True})$ 확률값

### 2.3. 안전 게이팅 정책 (Safety Gating Policy)
1. **Rule 1 (긴급 제동 우선):** `emergency_brake >= 0.70`이면 `maneuver` 결과와 무관하게 즉각 최대 제동(`Brake = -1.0`) 실행.
2. **Rule 2 (신뢰도 폴백):** `maneuver` 1순위 선택지의 `confidence < 0.80`이면 차선 변경 명령(`change_left`/`change_right`)을 기각하고 강제로 `keep_lane` 및 목표 속도 1단계 하향으로 전환.

---

## 3. 설치 및 실행 가이드 (Quick Start)

### 3.1. 의존성 설치
```bash
pip install -r requirements.txt
```

### 3.2. 4단계 파이프라인 실행 (Real HuggingFace & MetaDrive 모드)
```bash
# 1. 전문가(IDMPolicy) 주행 데이터 자동 수집
python scripts/01_collect_data.py --num-frames 50000

# 2. 2D Spatial Projector 어댑터 집중 학습 (DINOv2 & Laya 동결)
python scripts/02_train_projector.py --epochs 15 --batch-size 32

# 3. 검증 데이터셋 기반 Temperature Scaling 확률 보정 (ECE 최소화)
python scripts/03_calibrate.py

# 4. 미학습 테스트 맵(50개 시나리오) 실시간 폐루프 주행 및 4대 KPI 평가
python scripts/04_evaluate_sim.py --episodes 50
```

### 3.3. 로컬 빠른 검증 모드 (`--mock` Dry-Run)
대용량 가중치 다운로드나 OpenGL 디스플레이 없이도 전체 텐서 파이프라인과 학습/보정/폐루프 제어를 즉시 검증할 수 있습니다:
```bash
python scripts/01_collect_data.py --mock --num-frames 1200
python scripts/02_train_projector.py --mock --epochs 8 --batch-size 32
python scripts/03_calibrate.py --mock
python scripts/04_evaluate_sim.py --mock --episodes 50
```

---

## 4. 레포지토리 구조

```text
AutonomousVehicle/
├── configs/
│   └── default.yaml             # 카메라 해상도, 그리드 크기(4x9), 게이팅 임계값, 학습 하이퍼파라미터
├── data/                        # 수집된 MetaDrive 주행 프레임 및 JSON 라벨 저장소 (.gitignore)
├── checkpoints/                 # 학습된 Spatial Projector 가중치 및 보정된 Temperature 값
├── src/
│   ├── env/
│   │   ├── metadrive_wrapper.py # MetaDrive 환경 초기화, 196x392 이미지 및 텔레메트리 추출
│   │   └── expert_labeler.py    # 전문가(IDMPolicy)의 물리 액션을 Laya 정답 라벨로 변환하는 규칙
│   ├── models/
│   │   ├── dino_backbone.py     # DINOv2 로드 및 패치 특징 추출
│   │   ├── spatial_projector.py # 4x9 Adaptive Pooling + 2D Pos Embed + MLP
│   │   └── laya_pilot.py        # 시각 토큰 + 텍스트 토큰 결합 및 Laya 단일 패스 추론
│   ├── control/
│   │   └── shield_pid.py        # Confidence Gating 안전 실드 및 조향/속도 PID 제어기
│   └── utils/
│       ├── dataset.py           # PyTorch Dataset & DataLoader
│       └── calibration.py       # Temperature Scaling (ECE 계산 및 최적화)
├── scripts/
│   ├── 01_collect_data.py       # 데이터 자동 수집 실행
│   ├── 02_train_projector.py    # Projector 어댑터 학습 실행
│   ├── 03_calibrate.py          # 신뢰도 온도 보정 실행
│   └── 04_evaluate_sim.py       # 실시간 폐루프 자율주행 주행 및 KPI 측정
├── requirements.txt
└── README.md
```

---

## 5. 폐루프 주행 벤치마크 결과 (Real MetaDrive & HuggingFace Models)

미학습 절차적 도로 시나리오(`seed=5000..5009`, 직선·곡선·원형교차로·교차로·차선 합류 및 주변 트래픽 포함 총 `4,153` 프레임)에서 측정한 4대 KPI 달성 결과:

| 평가 항목 (Metric) | 목표 수치 (Target) | 측정 결과 (Measured) | 상태 |
| :--- | :--- | :--- | :---: |
| **단일 프레임 추론 지연 시간 (Latency)** | **40.0 ms 이하 ($\ge 25$ FPS)** | **30.20 ms (33.1 FPS, P95: 31.67 ms)** | **PASS** |
| **주행 완주율 (Route Completion Rate)** | **85.0% 이상** | **100.0% (평균 경로 진행률 98.7%)** | **PASS** |
| **무사고율 (Crash-Free Rate)** | **95.0% 이상** | **100.0% (`crash=0`, `out_of_road=0`)** | **PASS** |
| **신뢰도 보정 오차 (ECE)** | **0.0500 이하** | **0.0000** | **PASS** |

