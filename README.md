# AI 외상 응급도 판별 서비스

외상 부위 사진을 업로드하면 AI가 상처 종류와 응급도를 분류하고, 주변 병원을 지도로 안내하는 Streamlit 웹 애플리케이션입니다.

---

## 파일 구성

| 파일 | 역할 |
|------|------|
| `wound_app.py` | Streamlit 메인 앱. UI, 이미지 전처리, 모델 추론, 지도 렌더링 담당 |
| `prepare_data.py` | Kaggle 데이터셋 2종을 병합해 `dataset/`, `dataset_severity/` 폴더 구조 생성 |
| `train_model.py` | Model A 학습 — 진단명 7종 분류 (MobileNetV2 전이학습) |
| `train_severity_model.py` | Model B 학습 — 응급도 3단계 분류 (자가처치 / 일반병원 / 응급실) |
| `wound_model.tflite` | Model A 경량화 모델 (26MB → 2.6MB, 동적 양자화) |
| `severity_model.tflite` | Model B 경량화 모델 (25MB → 2.5MB, 동적 양자화) |
| `wound_model.keras` | Model A 원본 학습 결과 |
| `severity_model.keras` | Model B 원본 학습 결과 |
| `class_names.json` | Model A 출력 클래스 목록 (7종) |
| `severity_class_names.json` | Model B 출력 클래스 목록 (3종) |
| `requirements.txt` | 배포 의존성 목록 |

---

## 핵심 코드 3선

### 1. 응급도 게이지 — 가중 구간 합산

```python
# ZONE_PCT: 각 심각도 구간의 중심 위치(%)
ZONE_PCT = {"자가처치": 16.5, "일반병원": 49.5, "응급실": 83.0}

bar_pct = round(
    sum(sev_proba[i] * ZONE_PCT.get(sev_classes[i], 49.5)
        for i in range(len(sev_classes))), 1
)
```

단순히 가장 높은 확률 클래스 위치를 쓰는 대신, 각 클래스의 확률을 해당 구간 중심값과 가중합산해 게이지 위치를 계산합니다. 예를 들어 일반병원 60% + 응급실 40%이면 핀이 두 구간 경계 쪽에 표시되어 실제 불확실성을 직관적으로 표현합니다.

---

### 2. 2단계 전이학습 (Phase 1 → Phase 2 Fine-tuning)

```python
# Phase 1: MobileNetV2 가중치 동결, 분류 헤드만 학습
base.trainable = False
model.compile(optimizer=Adam(1e-3), ...)
model.fit(train_ds, epochs=EPOCHS_FROZEN, ...)

# Phase 2: 상위 30%만 해동해 미세조정
base.trainable = True
for layer in base.layers[:int(len(base.layers) * 0.70)]:
    layer.trainable = False
model.compile(optimizer=Adam(1e-5), ...)  # 학습률 100배 낮춤
model.fit(train_ds, epochs=EPOCHS_FINE, ...)
```

ImageNet으로 사전학습된 MobileNetV2 특징 추출기를 활용합니다. Phase 1에서 새로 추가한 분류 헤드를 먼저 안정화시킨 뒤, Phase 2에서 상위 30% 레이어만 해동해 의료 이미지 도메인에 맞게 미세조정합니다. 학습률을 1/100로 낮춰 기존 가중치가 무너지지 않게 합니다.

---

### 3. 클래스 불균형 보정 (Model B)

```python
def compute_class_weights(train_ds, num_classes):
    counts = np.zeros(num_classes, dtype=np.int64)
    for _, y_batch in train_ds:
        for label in y_batch.numpy():
            counts[label] += 1
    total = counts.sum()
    return {i: total / (num_classes * c) for i, c in enumerate(counts) if c > 0}
```

응급실 클래스는 학습 데이터가 172장으로 일반병원(1,848장)의 1/10 수준입니다. 클래스 가중치를 계산해 소수 클래스에 더 높은 페널티를 부여하고, 응급 환자 미감지율(FNR)을 47% → 29%로 낮췄습니다.

---

## 데이터셋 및 전처리

### 사용 데이터셋 (Kaggle)

| 출처 | 내용 | 이미지 수 |
|------|------|----------|
| `yasinpratomo/wound-dataset` | 찰과상, 타박상, 화상, 열상, 부종, 자상 | 약 900장 |
| `ibrahimfateen/wound-classification` | 찰과상, 화상, 열상, 당뇨성 상처, 정맥성 상처 등 | 약 2,700장 |
| **합계** | 중복 클래스 병합 후 | **약 3,600장** |

### 전처리 절차 (`prepare_data.py`)

1. 두 데이터셋의 클래스를 공통 진단명으로 매핑 (예: `Laseration` → `열상`)
2. `Normal` 클래스 등 불필요한 항목 제외
3. 전체를 섞은 뒤 Train 80% / Val 20% 분할
4. Model A용 (`dataset/`) — 진단명 7종 폴더 구조 생성
5. Model B용 (`dataset_severity/`) — 심각도 3단계 폴더 구조 생성

### 진단명 → 심각도 매핑

| 심각도 | 진단명 |
|--------|--------|
| 🟢 자가처치 | 찰과상, 타박상, 부종·염좌 의심 |
| 🟡 일반병원 | 열상, 감염 의심 |
| 🔴 응급실 | 화상, 출혈성 상처 |

---

## AI 학습 방법 및 절차

### 모델 구조

```
입력 (224×224×3)
  └─ MobileNetV2 (ImageNet 사전학습, 특징 추출기)
       └─ GlobalAveragePooling2D
            └─ Dropout(0.3)
                 └─ Dense(128, relu)   ← Model A
                 └─ Dense(64, relu)    ← Model B
                      └─ Dropout(0.2)
                           └─ Dense(N, softmax)  출력
```

### 데이터 증강

```python
RandomFlip("horizontal")
RandomRotation(0.15)
RandomZoom(0.10)
RandomBrightness(0.20)
RandomContrast(0.15)
```

### 학습 설정

| 항목 | Phase 1 | Phase 2 |
|------|---------|---------|
| Optimizer | Adam | Adam |
| Learning Rate | 1e-3 | 1e-5 |
| Epochs | 최대 10 | 최대 10 |
| Early Stopping | patience=3 | patience=3 |
| Batch Size | 16 | 16 |
| Base 레이어 | 전부 동결 | 상위 30% 해동 |

### TFLite 변환 (경량화)

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # 동적 범위 양자화
tflite_model = converter.convert()
```

26MB `.keras` 파일을 2.6MB `.tflite`로 변환해 서버 메모리와 로딩 속도를 크게 줄였습니다.

---

## 외부 API

### 공공데이터포털 — 응급의료기관 목록 조회

- **제공처**: 보건복지부 / 국립중앙의료원
- **서비스**: `ErmctInfoInqireService/getEgytLcinfoInqire`
- **엔드포인트**: `https://apis.data.go.kr/B552657/ErmctInfoInqireService`
- **인증**: 공공데이터포털 발급 API 키 (`serviceKey`)
- **주요 파라미터**:

| 파라미터 | 설명 |
|----------|------|
| `WGS84_LAT` | 기준 위도 |
| `WGS84_LON` | 기준 경도 |
| `numOfRows` | 최대 반환 건수 |

- **응답 필드**: `dutyName`(기관명), `wgs84Lat/Lon`(좌표), `dutyAddr`(주소), `dutyTel1`(전화)
- **거리 계산**: API 응답에 거리 필드가 없어 Haversine 공식으로 직접 계산 후 필터링

### Folium + streamlit-folium

- OpenStreetMap 기반 지도 렌더링
- 현재 위치(파란 핀) + 반경 원 + 병원 핀(응급실은 빨간색, 일반병원은 주황색) 표시

---

## 배포 절차

### 환경

- **플랫폼**: [Streamlit Community Cloud](https://streamlit.io/cloud)
- **저장소**: GitHub (`gangajee/aiprogramming`)
- **Python 런타임**: 3.11 (Streamlit Cloud 기본)

### 의존성 최적화 경위

| 단계 | 패키지 | 문제 |
|------|--------|------|
| 초기 | `tensorflow>=2.15.0` | 설치 용량 500MB+, 타임아웃 발생 |
| 변경 | `ai-edge-litert>=2.1.0` | ~10MB, TFLite 추론 전용 경량 패키지 |

TensorFlow 전체를 제거하고 TFLite 인터프리터만 포함된 `ai-edge-litert`로 교체해 Streamlit Cloud 배포 문제를 해결했습니다.

### 배포 단계

1. `.keras` 모델을 `.tflite`로 변환 (동적 양자화)
2. `requirements.txt`에서 `tensorflow` 제거 → `ai-edge-litert>=2.1.0` 추가
3. `wound_app.py`의 모델 로딩 코드를 `ai_edge_litert.Interpreter`로 교체
4. GitHub에 push → Streamlit Cloud 자동 재배포
5. Streamlit Cloud Secrets에 `API_KEY` 등록

### Secrets 설정

Streamlit Cloud 대시보드 → 앱 Settings → Secrets 탭:

```toml
API_KEY = "공공데이터포털_발급_키"
```

---

## 실행 방법

### 로컬

```bash
pip install -r requirements.txt
streamlit run wound_app.py
```

### 모델 재학습

```bash
# 1. Kaggle에서 데이터셋 다운로드
kaggle datasets download -d yasinpratomo/wound-dataset -p wound_raw/
kaggle datasets download -d ibrahimfateen/wound-classification -p extra_raw/

# 2. 데이터 병합 및 폴더 생성
python3 prepare_data.py

# 3. 모델 학습
python3 train_model.py          # Model A (진단명)
python3 train_severity_model.py # Model B (응급도)
```

---

## 주의사항

본 서비스는 AI 기반 초기 참고 정보를 제공하며 의학적 진단을 대체하지 않습니다. 긴급 상황 시 **119**에 신고하십시오.
