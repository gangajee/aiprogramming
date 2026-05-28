"""
Kaggle wound-dataset 다운로드 및 학습용 폴더 구조 변환
yasinpratomo/wound-dataset

실행 전 준비:
    pip install kaggle
    kaggle.json 을 ~/.kaggle/ 에 배치

실행:
    python3 prepare_data.py
"""

import os
import shutil
import zipfile
import random
from pathlib import Path

# ─────────────────────────────────────────────
# Kaggle 클래스 → 앱 진단명 매핑
# ─────────────────────────────────────────────
CLASS_MAP = {
    "Trauma":        "열상",        # 외상성 상처 → 열상으로 분류
    "Cellulitis":    "감염_의심",
    "Diabetic":      "부종_염좌",   # 만성 궤양류 → 부종으로 분류
    "Pressure":      "부종_염좌",
    "Venous":        "부종_염좌",
    "Arterial":      "출혈성_상처",
    "Surgical":      "열상",
    "Miscellaneous": "찰과상",
}

# 학습 / 검증 비율
TRAIN_RATIO = 0.80
DATASET_DIR = Path("dataset")
RAW_DIR     = Path("wound_raw")          # 압축 해제 위치


# ─────────────────────────────────────────────
# Step 1. Kaggle API로 다운로드
# ─────────────────────────────────────────────
def download():
    zip_path = Path("wound-dataset.zip")
    if zip_path.exists():
        print("✅ zip 파일 이미 존재 — 다운로드 생략")
        return

    print("📥 Kaggle에서 데이터셋 다운로드 중...")
    ret = os.system("kaggle datasets download -d yasinpratomo/wound-dataset")
    if ret != 0:
        print("\n❌ 다운로드 실패. 아래를 확인하세요:")
        print("  1. pip install kaggle")
        print("  2. kaggle.json 이 ~/.kaggle/ 에 있는지 확인")
        print("  3. chmod 600 ~/.kaggle/kaggle.json")
        raise SystemExit(1)
    print("✅ 다운로드 완료")


# ─────────────────────────────────────────────
# Step 2. 압축 해제
# ─────────────────────────────────────────────
def extract():
    zip_path = Path("wound-dataset.zip")
    if RAW_DIR.exists():
        print("✅ 압축 해제 폴더 이미 존재 — 생략")
        return

    print("📦 압축 해제 중...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(RAW_DIR)
    print(f"✅ {RAW_DIR}/ 에 압축 해제 완료")

    # 실제 폴더 구조 출력 (처음 3단계)
    print("\n📂 데이터셋 구조:")
    for p in sorted(RAW_DIR.rglob("*")):
        depth = len(p.relative_to(RAW_DIR).parts)
        if depth <= 2:
            indent = "  " * (depth - 1)
            print(f"{indent}{'📁' if p.is_dir() else '🖼'} {p.name}")


# ─────────────────────────────────────────────
# Step 3. wound_main 폴더에서 클래스별 이미지 탐색
# ─────────────────────────────────────────────
def find_wound_main() -> Path:
    candidates = list(RAW_DIR.rglob("wound_main"))
    if candidates:
        return candidates[0]
    # wound_main 이 없으면 직접 탐색
    for p in RAW_DIR.iterdir():
        if p.is_dir() and any(p.iterdir()):
            return p
    raise FileNotFoundError(f"{RAW_DIR} 안에서 이미지 폴더를 찾지 못했습니다.")


# ─────────────────────────────────────────────
# Step 4. train / val 폴더로 재구성
# ─────────────────────────────────────────────
def reorganize():
    if DATASET_DIR.exists():
        print("✅ dataset/ 폴더 이미 존재 — 생략 (삭제 후 재실행하면 초기화)")
        return

    wound_dir = find_wound_main()
    print(f"\n🔍 원본 폴더: {wound_dir}")

    # 클래스별 이미지 수집
    class_images: dict[str, list[Path]] = {}
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for cls_dir in sorted(wound_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        images = [f for f in cls_dir.rglob("*") if f.suffix.lower() in img_exts]
        if images:
            class_images[cls_dir.name] = images

    print("\n📊 원본 클래스별 이미지 수:")
    for cls, imgs in class_images.items():
        mapped = CLASS_MAP.get(cls, cls)
        print(f"  {cls:15s} → {mapped:15s}  ({len(imgs)}장)")

    # 매핑된 클래스로 합치기
    merged: dict[str, list[Path]] = {}
    for original_cls, images in class_images.items():
        target_cls = CLASS_MAP.get(original_cls, original_cls)
        merged.setdefault(target_cls, []).extend(images)

    # train / val 분할 및 복사
    print("\n📁 dataset/ 폴더 생성 중...")
    total_copied = 0
    for cls_name, images in merged.items():
        random.shuffle(images)
        split = int(len(images) * TRAIN_RATIO)
        splits = {"train": images[:split], "val": images[split:]}

        for phase, imgs in splits.items():
            dest_dir = DATASET_DIR / phase / cls_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for img in imgs:
                shutil.copy2(img, dest_dir / img.name)
            total_copied += len(imgs)

    # 결과 요약
    print("\n✅ 완료! dataset/ 구조:")
    for phase in ("train", "val"):
        phase_dir = DATASET_DIR / phase
        if not phase_dir.exists():
            continue
        print(f"\n  {phase}/")
        for cls_dir in sorted(phase_dir.iterdir()):
            count = len(list(cls_dir.iterdir()))
            print(f"    {cls_dir.name:20s} {count:4d}장")

    print(f"\n총 {total_copied}장 복사 완료 → python3 train_model.py 를 실행하세요")


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────
if __name__ == "__main__":
    random.seed(42)
    download()
    extract()
    reorganize()
