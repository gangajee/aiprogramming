"""
AI 외상 응급도 판별 서비스
한양대학교 ERICA 인공지능프로그래밍 프로젝트

실행:
    python3 wound_app.py       ← VSCode Code Runner / 터미널
    streamlit run wound_app.py ← 직접 실행
"""
import sys
import os
import subprocess

API_KEY       = os.environ.get("API_KEY", "")
API_BASE      = "https://apis.data.go.kr/B552657/ErmctInfoInqireService"
API_BASE_HOSP = "https://apis.data.go.kr/B551182/HsptlAsembySearchService"

CITIES = {
    "서울": (37.5665, 126.9780), "부산": (35.1796, 129.0756),
    "인천": (37.4563, 126.7052), "대구": (35.8714, 128.6014),
    "광주": (35.1595, 126.8526), "대전": (36.3504, 127.3845),
    "울산": (35.5384, 129.3114), "세종": (36.4801, 127.2890),
    "수원": (37.2636, 127.0286), "안산": (37.3219, 126.8309),
    "고양": (37.6584, 126.8320), "성남": (37.4201, 127.1269),
    "안양": (37.3943, 126.9568), "용인": (37.2411, 127.1775),
    "청주": (36.6424, 127.4890), "전주": (35.8242, 127.1480),
    "창원": (35.2279, 128.6811), "포항": (36.0190, 129.3435),
    "제주": (33.4996, 126.5312), "안동": (36.5684, 128.7294),
}

# 심각도 표시 순서 (앞일수록 심각)
SEVERITY_ORDER = [
    "출혈성 상처", "감염 의심", "화상",
    "열상", "타박상", "찰과상", "부종·염좌 의심",
]

# Model A 출력 클래스명 → 화면 표시명
MODEL_TO_DISPLAY = {
    "감염_의심":   "감염 의심",
    "부종_염좌":   "부종·염좌 의심",
    "출혈성_상처": "출혈성 상처",
    "열상": "열상", "찰과상": "찰과상",
    "타박상": "타박상", "화상": "화상",
    "정상": "정상",
}

# 외상 신뢰도 최소 임계값 — 이 미만이면 "외상 아님" 처리
WOUND_CONFIDENCE_THRESHOLD = 0.40

# Model B 클래스 → 게이지 구간 중심값 (자가0-33 / 병원33-66 / 응급66-100)
ZONE_PCT = {"자가처치": 16.5, "일반병원": 49.5, "응급실": 83.0}

SEV_LABEL_TO_LEVEL = {"자가처치": 0, "일반병원": 1, "응급실": 2}
TYPE_TO_LEVEL = {
    "출혈성 상처": 2, "감염 의심": 2, "화상": 2,
    "열상": 1, "타박상": 0, "찰과상": 0, "부종·염좌 의심": 0,
}

LEVELS = {
    0: {"icon": "🟢", "label": "자가 처치 가능",      "css": "level-0"},
    1: {"icon": "🟡", "label": "가까운 병원 내원 권고", "css": "level-1"},
    2: {"icon": "🔴", "label": "즉시 응급실 방문 필요", "css": "level-2"},
}

WOUND_GUIDES = {
    "찰과상": (
        "• 흐르는 물로 5분 이상 세척해 이물질을 제거하세요.\n"
        "• 소독약(베타딘 등)으로 가볍게 소독 후 거즈로 덮으세요.\n"
        "• 딱지가 생기기 전까지 습윤 밴드 사용을 권장합니다.\n"
        "• 발적·부종·고름이 생기면 감염 신호이므로 병원을 방문하세요."
    ),
    "열상": (
        "• 깨끗한 천으로 상처를 강하게 눌러 지혈하세요.\n"
        "• 봉합(실밥)이 필요할 수 있으니 오늘 중 외과를 방문하세요.\n"
        "• 상처 입구를 테이프로 임시 고정하지 마세요.\n"
        "• 6시간 이상 경과한 열상은 봉합이 어려울 수 있으니 서두르세요."
    ),
    "타박상": (
        "• 즉시 냉찜질(얼음팩을 수건으로 감싸기)을 20분간 적용하세요.\n"
        "• 다친 부위를 심장보다 높게 올려 부종을 줄이세요.\n"
        "• 48시간은 안정을 취하고 무리한 움직임을 피하세요.\n"
        "• 통증이 심하거나 골절이 의심되면 병원에서 X-ray를 찍으세요."
    ),
    "화상": (
        "• 즉시 흐르는 찬물로 15~20분간 식히세요 (얼음 직접 사용 금지).\n"
        "• 물집을 터뜨리지 마세요. 감염 위험이 높아집니다.\n"
        "• 옷이 붙어 있으면 억지로 벗기지 말고 그대로 병원에 가세요.\n"
        "• 화상 면적이 손바닥보다 크거나 얼굴·손·관절 부위면 응급실로 이동하세요."
    ),
    "출혈성 상처": (
        "• 깨끗한 천으로 상처를 강하게 압박하고 5~10분간 유지하세요.\n"
        "• 출혈 부위를 심장보다 높이 올리세요.\n"
        "• 압박 후에도 출혈이 멈추지 않으면 즉시 119에 신고하세요.\n"
        "• 혼자 이동하지 말고 보호자나 구급대를 기다리세요."
    ),
    "부종·염좌 의심": (
        "• RICE 원칙: 휴식·냉찜질·압박·거상(높이 올리기).\n"
        "• 냉찜질은 1회 20분, 하루 4~6회 반복하세요.\n"
        "• 무게를 싣거나 해당 관절을 사용하지 마세요.\n"
        "• 2일 후에도 붓기·통증이 지속되면 골절 여부를 확인하러 병원을 방문하세요."
    ),
    "감염 의심": (
        "• 상처를 손으로 짜거나 만지지 마세요.\n"
        "• 흐르는 물로 부드럽게 세척 후 깨끗한 거즈로 덮으세요.\n"
        "• 항생제 연고(후시딘 등)를 바르고 하루 2회 드레싱을 교체하세요.\n"
        "• 발열·붉은 선(림프관염)·고름이 보이면 오늘 중 병원에서 항생제 처방을 받으세요."
    ),
}

BADGE_COLORS = {
    "찰과상":         ("#fef3c7", "#92400e"),
    "열상":           ("#fee2e2", "#991b1b"),
    "타박상":         ("#ede9fe", "#4c1d95"),
    "화상":           ("#ffedd5", "#7c2d12"),
    "출혈성 상처":    ("#fee2e2", "#7f1d1d"),
    "부종·염좌 의심": ("#dbeafe", "#1e3a8a"),
    "감염 의심":      ("#d1fae5", "#064e3b"),
}


def _has_skin(pil_image, min_skin: float = 0.04, min_wound: float = 0.010) -> bool:
    """피부 맥락 + 상처 색상 지표 이중 검사.

    1단계) 정상 피부 픽셀이 4% 이상 — 피부가 없는 사물(사과 등) 제거
    2단계) 상처 지표 픽셀이 1% 이상 — 정상 피부 사진(상처 없음) 추가 제거
    """
    import numpy as np
    arr = np.array(pil_image.convert("RGB").resize((128, 128)), dtype=np.float32)
    R, G, B = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    Y  =  0.299 * R + 0.587 * G + 0.114 * B
    Cb = 128 - 0.16874 * R - 0.33126 * G + 0.5   * B
    Cr = 128 + 0.5     * R - 0.41869 * G - 0.08131 * B

    # 1단계: 정상 피부 픽셀 (Cr 133–167: 사과·토마토 황적색 제외)
    skin_mask = (Y > 60) & (Y < 220) & (Cb > 77) & (Cb < 127) & (Cr > 133) & (Cr < 167)
    if float(np.mean(skin_mask)) < min_skin:
        return False

    # 2단계: 상처 색상 지표 픽셀
    # 발적/찰과상/열상: 정상 피부 Cr 상한(167) 초과 — "피부보다 더 붉은" 영역
    wound_red = (Cr > 167) & (Y > 40) & (Y < 220) & (Cb < 125)
    # 멍/타박상: 어두운 자주빛 (낮은 밝기 + 피부 Cb 범주)
    bruise    = (Y < 78) & (Cr > 118) & (Cb > 82) & (Cb < 142)
    # 화상/물집: 극도로 밝은 창백 영역
    burn_pale = (Y > 212) & (Cb > 78) & (Cb < 122)

    wound_ratio = float(np.mean(wound_red | bruise | burn_pale))
    return wound_ratio >= min_wound


def _in_streamlit_ctx() -> bool:
    if "streamlit" not in sys.modules:
        return False
    try:
        from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


def fetch_nearby_hospitals(lat: float, lon: float, rows: int = 20) -> tuple[list[dict], str, dict]:
    """Returns (hospitals, error_msg, debug_info).
    1차: B551182/HsptlAsembySearchService (일반 병원 포함)
    실패 시 2차: B552657/ErmctInfoInqireService (응급기관 전용)
    """
    import requests, math
    debug = {}
    if not API_KEY:
        return [], "API_KEY가 설정되지 않았습니다.", debug

    def haversine(lat1, lon1, lat2, lon2):
        R = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
        return R * 2 * math.asin(math.sqrt(a))

    def parse_items(data):
        body = data.get("response", {}).get("body", {})
        items = body.get("items")
        if not items:
            return []
        item = items.get("item", [])
        return item if isinstance(item, list) else [item]

    # ── 1차 시도: 일반 병원 API ──────────────────────────────────────
    try:
        params = {
            "serviceKey": API_KEY, "WGS84_LAT": lat, "WGS84_LON": lon,
            "pageNo": 1, "numOfRows": rows, "_type": "json",
        }
        resp = requests.get(f"{API_BASE_HOSP}/getHsptlMdcncListInfoInqire",
                            params=params, timeout=8)
        debug["hosp_status"] = resp.status_code
        data = resp.json()
        header = data.get("response", {}).get("header", {})
        debug["hosp_resultCode"] = header.get("resultCode")
        debug["hosp_totalCount"] = data.get("response", {}).get("body", {}).get("totalCount", "?")

        if header.get("resultCode") == "00":
            hospitals = parse_items(data)
            if hospitals:
                for h in hospitals:
                    h_lat = h.get("wgs84Lat") or h.get("latitude")
                    h_lon = h.get("wgs84Lon") or h.get("longitude")
                    h["distance"] = round(haversine(lat, lon, float(h_lat), float(h_lon)), 2) if h_lat and h_lon else 9999
                debug["source"] = "HsptlAsembySearchService"
                return hospitals, "", debug
    except Exception as e:
        debug["hosp_error"] = str(e)

    # ── 2차 fallback: 응급의료기관 API ───────────────────────────────
    try:
        params = {
            "serviceKey": API_KEY, "WGS84_LAT": lat, "WGS84_LON": lon,
            "pageNo": 1, "numOfRows": 100, "_type": "json",
        }
        resp = requests.get(f"{API_BASE}/getEgytLcinfoInqire", params=params, timeout=8)
        debug["ermt_status"] = resp.status_code
        data = resp.json()
        header = data.get("response", {}).get("header", {})
        debug["ermt_resultCode"] = header.get("resultCode")
        debug["ermt_totalCount"] = data.get("response", {}).get("body", {}).get("totalCount", "?")

        if header.get("resultCode") != "00":
            return [], f"API 오류: {header.get('resultMsg', '알 수 없는 오류')}", debug

        hospitals = parse_items(data)
        for h in hospitals:
            h_lat = h.get("wgs84Lat") or h.get("latitude")
            h_lon = h.get("wgs84Lon") or h.get("longitude")
            h["distance"] = round(haversine(lat, lon, float(h_lat), float(h_lon)), 2) if h_lat and h_lon else 9999
        debug["source"] = "ErmctInfoInqireService (fallback)"
        return hospitals, "", debug
    except Exception as e:
        debug["ermt_error"] = str(e)
        return [], f"요청 실패: {e}", debug


def main():
    import streamlit as st
    from PIL import Image
    import numpy as np
    import folium
    from streamlit_folium import st_folium

    global API_KEY
    if not API_KEY:
        try:
            API_KEY = st.secrets["API_KEY"]
        except Exception:
            pass

    st.set_page_config(page_title="AI 외상 응급도 판별", page_icon="🏥", layout="centered")

    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }

        [data-testid="stAppViewContainer"] {
            background: #f0f2f6;
        }
        [data-testid="stHeader"] { background: transparent !important; }
        .block-container { padding-top: 1.8rem !important; }

        /* ── 헤더 ── */
        .header-card {
            background: linear-gradient(135deg, #0f2942 0%, #1a4a7a 55%, #1e6091 100%);
            border-radius: 22px;
            padding: 38px 32px 30px;
            color: white;
            margin-bottom: 24px;
            text-align: center;
            box-shadow: 0 12px 40px rgba(15,41,66,0.28), 0 2px 8px rgba(0,0,0,0.12);
            position: relative; overflow: hidden;
        }
        .header-card::after {
            content: '';
            position: absolute; top: -60px; right: -60px;
            width: 220px; height: 220px;
            background: rgba(255,255,255,0.06);
            border-radius: 50%; pointer-events: none;
        }
        .header-card h1 {
            font-size: 1.7rem; font-weight: 800;
            margin: 0 0 8px; letter-spacing: -0.4px;
        }
        .header-card p { font-size: 0.88rem; opacity: 0.7; margin: 0; }

        /* ── 결과 카드 ── */
        .result-card {
            border-radius: 20px; padding: 32px 28px 28px;
            margin-top: 18px; text-align: center;
            box-shadow: 0 4px 24px rgba(0,0,0,0.07), 0 1px 4px rgba(0,0,0,0.05);
        }
        .result-card .icon { font-size: 2.6rem; line-height: 1; }
        .result-card h2 { font-size: 1.55rem; font-weight: 700; margin: 12px 0 6px; }
        .result-card .sub { font-size: 0.83rem; opacity: 0.65; margin-top: 10px; }

        .level-0 { background: linear-gradient(145deg,#f0fdf6,#dcfce7); color:#14532d; }
        .level-1 { background: linear-gradient(145deg,#fffceb,#fef3c7); color:#713f12; }
        .level-2 { background: linear-gradient(145deg,#fff5f5,#fee2e2); color:#7f1d1d; }

        /* ── 가이드 박스 ── */
        .guide-box {
            background: white; border-radius: 18px;
            padding: 22px 26px; margin-top: 14px;
            border-left: 5px solid #6366f1;
            box-shadow: 0 2px 16px rgba(0,0,0,0.06);
            font-size: 0.92rem; line-height: 1.85; color: #1e293b;
        }
        .guide-box strong { font-size: 0.95rem; color: #312e81; }

        /* ── 외상 없음 카드 ── */
        .no-wound-card {
            background: white; border-radius: 20px;
            padding: 36px 28px; margin-top: 18px; text-align: center;
            box-shadow: 0 4px 24px rgba(0,0,0,0.07);
            border-top: 5px solid #f59e0b;
        }
        .no-wound-card h2 { color: #92400e; font-size: 1.4rem; font-weight: 700; margin: 12px 0 8px; }
        .no-wound-card p  { color: #78350f; font-size: 0.9rem; line-height: 1.7; margin: 0; }

        /* ── 병원 섹션 ── */
        .hospital-header {
            background: white; border-radius: 18px;
            padding: 20px 24px 16px; margin-top: 20px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        }
        .hospital-header h3 { font-size: 1.1rem; font-weight: 700; color: #0f172a; margin: 0; }

        /* ── 면책 ── */
        .disclaimer {
            background: rgba(255,255,255,0.6);
            border: 1px solid #e2e8f0;
            border-radius: 14px; padding: 16px 20px;
            font-size: 0.8rem; color: #94a3b8;
            margin-top: 32px; line-height: 1.6; text-align: center;
        }

        /* ── 업로드 영역 커스텀 ── */
        [data-testid="stFileUploader"] > div:first-child {
            border: 2px dashed #cbd5e1 !important;
            border-radius: 16px !important;
            background: white !important;
            transition: border-color .2s;
        }
        [data-testid="stFileUploader"] > div:first-child:hover {
            border-color: #6366f1 !important;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="header-card">
            <h1>🏥 AI 외상 응급도 판별</h1>
            <p>외상 사진을 업로드하면 AI가 상처 종류와 내원 필요 여부를 판단해 드립니다</p>
        </div>
    """, unsafe_allow_html=True)

    def load_models():
        import json
        load_err = ""
        try:
            from ai_edge_litert.interpreter import Interpreter
        except Exception as e:
            return None, None, False, None, None, False, f"ai_edge_litert import 실패: {e}"

        cwd = os.getcwd()
        type_interp, type_classes, use_type = None, None, False
        a_tflite = os.path.join(cwd, "wound_model.tflite")
        a_json   = os.path.join(cwd, "class_names.json")
        if os.path.exists(a_tflite) and os.path.exists(a_json):
            try:
                type_interp = Interpreter(a_tflite)
                type_interp.allocate_tensors()
                with open(a_json, encoding="utf-8") as f:
                    type_classes = json.load(f)
                use_type = True
            except Exception as e:
                load_err += f"Model A 로드 실패: {e} | "
        else:
            load_err += f"Model A 파일 없음 (cwd={cwd}) | "

        sev_interp, sev_classes, use_sev = None, None, False
        b_tflite = os.path.join(cwd, "severity_model.tflite")
        b_json   = os.path.join(cwd, "severity_class_names.json")
        if os.path.exists(b_tflite) and os.path.exists(b_json):
            try:
                sev_interp = Interpreter(b_tflite)
                sev_interp.allocate_tensors()
                with open(b_json, encoding="utf-8") as f:
                    sev_classes = json.load(f)
                use_sev = True
            except Exception as e:
                load_err += f"Model B 로드 실패: {e}"
        else:
            load_err += f"Model B 파일 없음"

        return type_interp, type_classes, use_type, sev_interp, sev_classes, use_sev, load_err

    type_model, type_classes, USE_TYPE, sev_model, sev_classes, USE_SEV, _load_err = load_models()
    if _load_err:
        st.warning(f"⚠️ 모델 로드 오류: {_load_err}")

    def _tflite_run(interp, arr):
        inp = interp.get_input_details()[0]['index']
        out = interp.get_output_details()[0]['index']
        interp.set_tensor(inp, arr)
        interp.invoke()
        return interp.get_tensor(out)[0]

    def preprocess(pil_image):
        # MobileNetV2 preprocess_input: x / 127.5 - 1.0
        arr = np.array(pil_image.convert("RGB").resize((224, 224)), dtype=np.float32)
        return np.expand_dims(arr / 127.5 - 1.0, 0)

    def analyze_wound(pil_image: Image.Image):
        """Returns (level, bar_pct, detected, primary_type, type_detail, sev_detail)"""

        # 피부색이 감지되지 않으면 외상 사진이 아님
        if not _has_skin(pil_image):
            return -1, 0.0, [], None, {}, {}

        if USE_TYPE or USE_SEV:
            arr = preprocess(pil_image)

            # Model A: 진단명
            if USE_TYPE:
                type_proba = _tflite_run(type_model, arr)
                top_idx = int(np.argmax(type_proba))
                top_conf = float(type_proba[top_idx])
                top_cls  = MODEL_TO_DISPLAY.get(type_classes[top_idx], type_classes[top_idx])

                # 신뢰도 미달 또는 정상 클래스 → 외상 아님
                if top_conf < WOUND_CONFIDENCE_THRESHOLD or top_cls == "정상":
                    return -1, 0.0, [], None, {}, {}

                raw = [type_classes[i] for i in np.argsort(type_proba)[::-1] if type_proba[i] >= 0.15][:2]
                disp = [MODEL_TO_DISPLAY.get(c, c) for c in raw]
                detected = sorted(
                    [d for d in disp if d in SEVERITY_ORDER],
                    key=lambda x: SEVERITY_ORDER.index(x),
                ) or disp[:1]
                primary_type = detected[0]
                type_detail = {MODEL_TO_DISPLAY.get(c, c): f"{type_proba[i]:.1%}" for i, c in enumerate(type_classes)
                               if MODEL_TO_DISPLAY.get(c, c) != "정상"}
            else:
                detected, primary_type, type_detail = ["부종·염좌 의심"], "부종·염좌 의심", {}

            # Model B: 심각도
            if USE_SEV:
                sev_proba = _tflite_run(sev_model, arr)
                sev_label = sev_classes[int(np.argmax(sev_proba))]
                level = SEV_LABEL_TO_LEVEL.get(sev_label, 0)
                # 각 클래스 확률을 구간 중심값으로 가중합산 → 게이지 위치
                bar_pct = round(sum(sev_proba[i] * ZONE_PCT.get(sev_classes[i], 49.5)
                                    for i in range(len(sev_classes))), 1)
                sev_detail = {c: f"{sev_proba[i]:.1%}" for i, c in enumerate(sev_classes)}
            else:
                level = TYPE_TO_LEVEL.get(primary_type, 0)
                bar_pct = {0: 16.5, 1: 49.5, 2: 83.0}[level]
                sev_detail = {}

        else:
            # HSV 폴백: 모델 없을 때 (색상 분석)
            img = pil_image.convert("RGB").resize((256, 256))
            arr = np.array(img, dtype=np.float32)
            r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
            brightness = (r + g + b) / 3.0

            red_ratio      = float(np.mean((r > 140) & (r > g + 25) & (r > b + 25)))
            deep_red_ratio = float(np.mean((r > 160) & (r > g + 50) & (r > b + 40)))
            dark_ratio     = float(np.mean(brightness < 60))
            color_score    = float(min(np.std(arr) / 80.0, 1.0))
            purple_ratio   = float(np.mean((b > 90) & (r > 70) & (b >= r * 0.85) & (g < b) & (brightness < 160)))
            yellow_ratio   = float(np.mean((r > 120) & (g > 100) & (b < 90) & (g > b * 1.5)))
            pale_ratio     = float(np.mean(brightness > 210))

            score = float(np.clip(
                red_ratio * 0.35 + deep_red_ratio * 0.30 + dark_ratio * 0.20 + color_score * 0.15, 0.0, 1.0
            ))

            # 색상 점수가 낮으면 외상 아님으로 판단
            if score < 0.12:
                return -1, 0.0, [], None, {}, {}

            detected = []
            if deep_red_ratio > 0.08:                                        detected.append("출혈성 상처")
            if yellow_ratio   > 0.04:                                        detected.append("감염 의심")
            if pale_ratio > 0.30 and red_ratio > 0.03:                       detected.append("화상")
            if red_ratio > 0.05 and color_score > 0.45:                      detected.append("열상")
            if purple_ratio > 0.06 or (dark_ratio > 0.10 and red_ratio < 0.10): detected.append("타박상")
            if red_ratio > 0.12 and deep_red_ratio < 0.06:                   detected.append("찰과상")
            if not detected:                                                  detected.append("부종·염좌 의심")

            detected = sorted(set(detected), key=lambda x: SEVERITY_ORDER.index(x))
            primary_type = detected[0]
            level = 2 if score >= 0.25 else (1 if score >= 0.10 else 0)
            if score < 0.10:
                bar_pct = round((score / 0.10) * 33, 1)
            elif score < 0.25:
                bar_pct = round(33 + ((score - 0.10) / 0.15) * 33, 1)
            else:
                bar_pct = round(66 + min((score - 0.25) / 0.75, 1.0) * 34, 1)
            type_detail = {
                "붉은 영역 비율": f"{red_ratio:.1%}",
                "짙은 적색 비율": f"{deep_red_ratio:.1%}",
                "암부(멍) 비율":  f"{dark_ratio:.1%}",
                "색상 복잡도":    f"{color_score:.2f}",
            }
            sev_detail = {}

        return level, bar_pct, detected, primary_type, type_detail, sev_detail

    # ── 업로드 UI ──────────────────────────────────────────────────────────────
    st.markdown(
        '<p style="text-align:center;color:#64748b;font-size:0.9rem;margin-bottom:8px;">'
        "📎 외상 부위를 촬영한 사진을 업로드해 주세요 (JPG · PNG · WEBP)</p>",
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        label="사진 업로드", type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    if uploaded is not None:
        image = Image.open(uploaded)
        st.image(image, caption="업로드된 외상 사진", use_container_width=True)

        # 어떤 모드로 동작 중인지 표시
        if USE_TYPE and USE_SEV:
            st.markdown(
                '<p style="text-align:center;margin:6px 0 10px;">'
                '<span style="background:#dcfce7;color:#15803d;font-size:0.82rem;font-weight:600;'
                'padding:4px 14px;border-radius:20px;border:1px solid #86efac;">🤖 AI 모델 분석 중</span></p>',
                unsafe_allow_html=True,
            )
        else:
            st.error(f"⚠️ AI 모델 미로드 — 색상 분석 모드 (신뢰도 낮음) | 오류: {_load_err}")

        with st.spinner("AI가 사진을 분석하고 있습니다..."):
            level, bar_pct, detected, primary_type, type_detail, sev_detail = analyze_wound(image)

        if level == -1:
            st.markdown("""
                <div class="no-wound-card">
                    <div style="font-size:2.8rem;">🔍</div>
                    <h2>외상이 감지되지 않았습니다</h2>
                    <p>업로드한 사진에서 상처나 외상을 찾을 수 없습니다.<br>
                    <strong>외상 부위를 가까이 촬영한 사진</strong>을 다시 업로드해 주세요.<br>
                    찰과상 · 열상 · 타박상 · 화상 · 출혈 부위 등의 사진에 최적화되어 있습니다.</p>
                </div>
            """, unsafe_allow_html=True)
            st.stop()

        meta = LEVELS[level]

        # 진단명 배지
        badges_html = ""
        for i, wt in enumerate(detected):
            bg, fg = BADGE_COLORS.get(wt, ("#f1f5f9", "#374151"))
            label = f"⚠ {wt}" if i == 0 and len(detected) > 1 else f"🩹 {wt}"
            badges_html += (
                f'<span style="display:inline-block;margin:3px 4px;background:{bg};color:{fg};'
                f'border:1.5px solid {fg}40;padding:4px 14px;border-radius:20px;'
                f'font-size:0.9rem;font-weight:{"700" if i == 0 else "500"};">{label}</span>'
            )

        st.markdown(f"""
            <div class="result-card {meta['css']}">
                <div style="font-size:2.4rem;">{meta['icon']}</div>
                <h2>{meta['label']}</h2>
                <div style="margin:10px 0 4px;">{badges_html}</div>
                <p style="margin-top:8px;font-size:0.85rem;opacity:0.75;">아래 응급도 게이지를 확인하세요</p>
            </div>
        """, unsafe_allow_html=True)

        # 응급도 게이지
        st.markdown(f"""
            <div style="margin:16px 0 6px;font-weight:600;font-size:0.9rem;color:#374151;">응급도 게이지</div>
            <div style="position:relative;height:18px;border-radius:10px;overflow:visible;
                        background:linear-gradient(to right,#16a34a 33%,#ca8a04 33% 66%,#dc2626 66%);">
                <div style="position:absolute;left:33%;top:0;bottom:0;width:2px;background:white;opacity:0.6;"></div>
                <div style="position:absolute;left:66%;top:0;bottom:0;width:2px;background:white;opacity:0.6;"></div>
                <div style="position:absolute;left:{bar_pct}%;top:50%;transform:translate(-50%,-50%);
                            width:22px;height:22px;background:white;border-radius:50%;
                            border:3px solid #374151;box-shadow:0 2px 6px rgba(0,0,0,0.25);z-index:10;"></div>
            </div>
            <div style="display:flex;justify-content:space-between;font-size:0.75rem;color:#64748b;margin-top:6px;">
                <span>🟢 자가처치</span><span>🟡 일반병원</span><span>🔴 응급실</span>
            </div>
        """, unsafe_allow_html=True)

        # 행동 가이드
        guide_text = WOUND_GUIDES.get(primary_type, "• 전문의와 상담하세요.")
        guide_html = "<br>".join(guide_text.strip().split("\n"))
        guide_note = f" (복합 외상 — '{primary_type}' 기준)" if len(detected) > 1 else f" — {primary_type}"
        st.markdown(
            f'<div class="guide-box"><strong>📋 행동 가이드{guide_note}</strong><br><br>{guide_html}</div>',
            unsafe_allow_html=True,
        )

        # 세부 수치
        if type_detail or sev_detail:
            with st.expander("🔬 세부 분석 수치 보기"):
                if type_detail:
                    st.caption("진단명별 확률 (상위 3)")
                    top3 = sorted(type_detail.items(), key=lambda x: float(x[1].rstrip('%')), reverse=True)[:3]
                    cols = st.columns(len(top3))
                    for col, (k, v) in zip(cols, top3):
                        col.metric(k, v)
                if sev_detail:
                    st.caption("심각도별 확률")
                    cols = st.columns(len(sev_detail))
                    for col, (k, v) in zip(cols, sev_detail.items()):
                        col.metric(k, v)

        # 병원 지도
        if True:
            import pandas as pd
            title = "🚨 가까운 응급실 찾기" if level == 2 else "🏥 가까운 병원 찾기"
            st.markdown(
                f'<div class="hospital-header"><h3>{title}</h3></div>',
                unsafe_allow_html=True,
            )

            for key, default in [("hospitals_raw", []), ("hospital_city", "서울")]:
                if key not in st.session_state:
                    st.session_state[key] = default

            city       = st.selectbox("현재 위치 (도시 선택)", options=list(CITIES.keys()))
            search_btn = st.button("병원 검색", type="primary", use_container_width=True)

            if search_btn:
                lat, lon = CITIES[city]
                with st.spinner("주변 응급의료기관을 검색 중입니다..."):
                    raw, err, _ = fetch_nearby_hospitals(lat, lon, rows=100)
                    st.session_state.hospitals_raw  = raw
                    st.session_state.hospital_error = err
                    st.session_state.hospital_city  = city

            if st.session_state.get("hospital_error"):
                st.error(f"병원 검색 오류: {st.session_state.hospital_error}")

            # radius 필터 없이 거리순 정렬 후 전체 표시
            hospitals = sorted(
                [h for h in st.session_state.hospitals_raw if h.get("distance") is not None],
                key=lambda h: float(h.get("distance", 9999)),
            )

            if search_btn and not hospitals and not st.session_state.get("hospital_error"):
                st.warning("검색 결과가 없습니다. 다른 도시를 선택해 보세요.")
                dbg = st.session_state.get("hospital_debug", {})
                if dbg:
                    with st.expander("🔍 API 진단 정보"):
                        st.json(dbg)

            if hospitals:
                lat, lon = CITIES[st.session_state.hospital_city]
                m = folium.Map(location=[lat, lon], zoom_start=11, tiles="CartoDB positron")

                folium.CircleMarker(
                    location=[lat, lon], radius=10, color="white", weight=3,
                    fill=True, fill_color="#1e40af", fill_opacity=1.0, tooltip="📍 현재 위치",
                ).add_to(m)

                pin_color  = "#dc2626" if level == 2 else "#d97706"
                pin_border = "#7f1d1d" if level == 2 else "#713f12"

                for h in hospitals:
                    h_lat = h.get("wgs84Lat") or h.get("latitude")
                    h_lon = h.get("wgs84Lon") or h.get("longitude")
                    if not h_lat or not h_lon:
                        continue
                    name = h.get("dutyName", "병원")
                    dist = h.get("distance", "-")
                    popup_html = (
                        f"<div style='font-size:13px;line-height:1.6'>"
                        f"<b style='font-size:14px'>{name}</b><br>"
                        f"🏷 {h.get('dutyDivName','-')}<br>"
                        f"📍 {h.get('dutyAddr','-')}<br>"
                        f"☎ {h.get('dutyTel1','-')}<br>"
                        f"📏 {dist}km</div>"
                    )
                    folium.CircleMarker(
                        location=[float(h_lat), float(h_lon)],
                        radius=9, color=pin_border, weight=2,
                        fill=True, fill_color=pin_color, fill_opacity=0.85,
                        tooltip=f"🏥 {name}  ({dist}km)",
                        popup=folium.Popup(popup_html, max_width=260),
                    ).add_to(m)

                st_folium(m, width=700, height=450, returned_objects=[])

                st.markdown(f"**가까운 응급의료기관 ({len(hospitals)}개) — 거리순**")
                st.dataframe(
                    pd.DataFrame([{
                        "기관명": h.get("dutyName", "-"), "분류": h.get("dutyDivName", "-"),
                        "거리(km)": h.get("distance", "-"), "전화번호": h.get("dutyTel1", "-"),
                        "주소": h.get("dutyAddr", "-"),
                    } for h in hospitals]),
                    use_container_width=True, hide_index=True,
                )

    else:
        st.markdown("""
            <div style="text-align:center;padding:52px 24px;background:white;
                border-radius:20px;border:2px dashed #cbd5e1;color:#94a3b8;margin-top:8px;
                box-shadow:0 2px 12px rgba(0,0,0,0.04);">
                <div style="font-size:3.4rem;margin-bottom:14px;filter:drop-shadow(0 2px 6px rgba(0,0,0,0.08));">📸</div>
                <p style="font-size:1.05rem;font-weight:700;color:#334155;margin:0 0 8px;letter-spacing:-0.2px;">
                    사진을 업로드하면 분석이 시작됩니다
                </p>
                <p style="font-size:0.85rem;margin:0;color:#94a3b8;line-height:1.6;">
                    찰과상 · 열상 · 타박상 · 화상 · 출혈 부위 사진을 올려주세요
                </p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("""
        <div class="disclaimer">
            ⚠️ 본 서비스는 AI 기반 초기 참고 정보를 제공하며 의학적 진단을 대체하지 않습니다.<br>
            최종 판단은 반드시 의료 전문가에게 확인하십시오. · 긴급 상황 시 → <strong>119</strong>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    if _in_streamlit_ctx():
        main()
    else:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", os.path.abspath(__file__),
             "--server.headless", "false", "--browser.gatherUsageStats", "false"],
            check=False,
        )
