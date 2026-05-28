"""
Kaggle wound-dataset 다운로드 및 학습용 폴더 구조 생성
  - dataset/          : Model A 진단명 분류용
  - dataset_severity/ : Model B 심각도 분류용

실행:
    pip install kaggle
    # kaggle.json → ~/.kaggle/ 배치 후
    python3 prepare_data.py
"""

import os, shutil, zipfile, random, json
from pathlib import Path

# ─────────────────────────────────────────────
# 클래스 매핑
# ─────────────────────────────────────────────
# Kaggle 원본 → 진단명
WOUND_TYPE_MAP = {
    "Trauma":        "열상",
    "Surgical":      "열상",
    "Cellulitis":    "감염_의심",
    "Arterial":      "출혈성_상처",
    "Miscellaneous": "찰과상",
    "Diabetic":      "부종_염좌",
    "Pressure":      "부종_염좌",
    "Venous":        "부종_염좌",
}

# Kaggle 원본 → 심각도 레벨
#   0: 자가 처치 가능
#   1: 일반 병원 진료 필요
#   2: 응급실 방문 필요
SEVERITY_MAP = {
    "Arterial":      2,   # 동맥 손상 → 응급
    "Cellulitis":    2,   # 감염 진행 → 응급
    "Trauma":        1,   # 외상 → 병원
    "Surgical":      1,   # 수술 후 처치 → 병원
    "Diabetic":      1,   # 당뇨 궤양 → 병원
    "Pressure":      1,   # 압박 궤양 → 병원
    "Venous":        0,   # 정맥성 → 자가/경과 관찰
    "Miscellaneous": 0,   # 기타 경미 → 자가
}

SEVERITY_LABEL = {0: "자가처치", 1: "일반병원", 2: "응급실"}

TRAIN_RATIO = 0.80
RAW_DIR     = Path("wound_raw")
IMG_EXTS    = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ─────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────
def copy_images(images: list[Path], dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    for img in images:
        dst = dest_dir / img.name
        # 파일명 충돌 방지
        if dst.exists():
            dst = dest_dir / f"{img.stem}_{img.parent.name}{img.suffix}"
        shutil.copy2(img, dst)


def print_summary(dataset_dir: Path, label: str):
    print(f"\n  [{label}]")
    for phase in ("train", "val"):
        d = dataset_dir / phase
        if not d.exists():
            continue
        print(f"  {phase}/")
        for cls_dir in sorted(d.iterdir()):
            n = len(list(cls_dir.glob("*")))
            print(f"    {cls_dir.name:20s} {n:4d}장")


# ─────────────────────────────────────────────
# Step 1. 다운로드
# ─────────────────────────────────────────────
def download():
    zip_path = Path("wound-dataset.zip")
    if zip_path.exists():
        print("✅ zip 파일 이미 존재 — 다운로드 생략")
        return
    print("📥 Kaggle 데이터셋 다운로드 중...")
    ret = os.system("kaggle datasets download -d yasinpratomo/wound-dataset")
    if ret != 0:
        print("\n❌ 실패. 확인사항:")
        print("  pip install kaggle")
        print("  ~/.kaggle/kaggle.json 존재 여부")
        print("  chmod 600 ~/.kaggle/kaggle.json")
        raise SystemExit(1)
    print("✅ 다운로드 완료")


# ─────────────────────────────────────────────
# Step 2. 압축 해제
# ─────────────────────────────────────────────
def extract():
    if RAW_DIR.exists():
        print("✅ 압축 해제 폴더 이미 존재 — 생략")
        return
    print("📦 압축 해제 중...")
    with zipfile.ZipFile("wound-dataset.zip") as z:
        z.extractall(RAW_DIR)
    print(f"✅ {RAW_DIR}/ 압축 해제 완료")

    print("\n📂 원본 구조 (2단계까지):")
    for p in sorted(RAW_DIR.rglob("*")):
        depth = len(p.relative_to(RAW_DIR).parts)
        if depth <= 2:
            print("  " * (depth - 1) + ("📁 " if p.is_dir() else "🖼  ") + p.name)


# ─────────────────────────────────────────────
# Step 3. wound_main 위치 탐색
# ─────────────────────────────────────────────
def find_wound_main() -> Path:
    for candidate in RAW_DIR.rglob("wound_main"):
        if candidate.is_dir():
            return candidate
    # fallback: 직접 클래스 폴더가 있는 곳 탐색
    for p in RAW_DIR.rglob("Trauma"):
        return p.parent
    raise FileNotFoundError("wound_main 폴더를 찾을 수 없습니다. 폴더 구조를 확인하세요.")


# ─────────────────────────────────────────────
# Step 4. 두 데이터셋 동시 구성
# ─────────────────────────────────────────────
def reorganize():
    dataset_a = Path("dataset")
    dataset_b = Path("dataset_severity")

    if dataset_a.exists() and dataset_b.exists():
        print("✅ dataset/, dataset_severity/ 이미 존재 — 생략")
        return

    wound_dir = find_wound_main()
    print(f"\n🔍 원본 폴더: {wound_dir}")

    # 클래스별 이미지 수집
    print("\n📊 원본 클래스별 이미지 수:")
    class_images: dict[str, list[Path]] = {}
    for cls_dir in sorted(wound_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        images = [f for f in cls_dir.rglob("*") if f.suffix.lower() in IMG_EXTS]
        if not images:
            continue
        class_images[cls_dir.name] = images
        wt  = WOUND_TYPE_MAP.get(cls_dir.name, cls_dir.name)
        sev = SEVERITY_LABEL.get(SEVERITY_MAP.get(cls_dir.name, -1), "?")
        print(f"  {cls_dir.name:15s} → 진단명:{wt:12s} / 심각도:{sev}  ({len(images)}장)")

    # 진단명 / 심각도 별로 이미지 합산
    type_images:     dict[str, list[Path]] = {}
    severity_images: dict[int,  list[Path]] = {}

    for orig_cls, images in class_images.items():
        wt  = WOUND_TYPE_MAP.get(orig_cls, orig_cls)
        sev = SEVERITY_MAP.get(orig_cls, 0)
        type_images.setdefault(wt,  []).extend(images)
        severity_images.setdefault(sev, []).extend(images)

    # ── Model A: 진단명 데이터셋 ──────────────────
    print("\n📁 dataset/ (진단명) 생성 중...")
    for cls_name, images in type_images.items():
        random.shuffle(images)
        split = int(len(images) * TRAIN_RATIO)
        copy_images(images[:split], dataset_a / "train" / cls_name)
        copy_images(images[split:], dataset_a / "val"   / cls_name)

    # ── Model B: 심각도 데이터셋 ─────────────────
    print("📁 dataset_severity/ (심각도) 생성 중...")
    for sev_level, images in severity_images.items():
        cls_name = SEVERITY_LABEL[sev_level]
        random.shuffle(images)
        split = int(len(images) * TRAIN_RATIO)
        copy_images(images[:split], dataset_b / "train" / cls_name)
        copy_images(images[split:], dataset_b / "val"   / cls_name)

    # ── 요약 출력 ────────────────────────────────
    print("\n✅ 완료!")
    print_summary(dataset_a, "Model A — 진단명")
    print_summary(dataset_b, "Model B — 심각도")

    # class_names 미리 저장 (학습 전 확인용)
    a_classes = sorted([d.name for d in (dataset_a / "train").iterdir()])
    b_classes = sorted([d.name for d in (dataset_b / "train").iterdir()])
    with open("class_names_preview.json", "w", encoding="utf-8") as f:
        json.dump({"wound_type": a_classes, "severity": b_classes},
                  f, ensure_ascii=False, indent=2)

    print(f"\n다음 단계:")
    print("  python3 train_model.py          ← Model A 학습 (진단명)")
    print("  python3 train_severity_model.py ← Model B 학습 (심각도)")


if __name__ == "__main__":
    random.seed(42)
    download()
    extract()
    reorganize()
