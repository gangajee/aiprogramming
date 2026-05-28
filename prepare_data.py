"""
멀티소스 wound 데이터셋 병합 및 학습용 폴더 구조 생성
  Sources:
    1. wound_raw/          : yasinpratomo/wound-dataset (기존)
    2. extra_raw/wound_classification/ : ibrahimfateen/wound-classification (추가)

  출력:
    - dataset/          : Model A 진단명 분류용
    - dataset_severity/ : Model B 심각도 분류용

실행:
    python3 prepare_data.py
"""

import os, shutil, zipfile, random, json
from pathlib import Path

IMG_EXTS   = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TRAIN_RATIO = 0.80

# ─────────────────────────────────────────────────────────────────────────────
# 소스별 클래스 매핑  →  (진단명, severity_level)
#   severity:  0=자가처치  1=일반병원  2=응급실
#   None 이면 해당 폴더 건너뜀
# ─────────────────────────────────────────────────────────────────────────────

# Source 1: yasinpratomo/wound-dataset
SOURCE1_MAP = {
    "Abrasions":    ("찰과상",      0),
    "Bruises":      ("타박상",      0),
    "Burns":        ("화상",        2),
    "Cut":          ("열상",        1),
    "Ingrown_nails":("부종_염좌",   0),
    "Laceration":   ("열상",        1),
    "Stab_wound":   ("출혈성_상처", 2),
}

# Source 2: ibrahimfateen/wound-classification
SOURCE2_MAP = {
    "Abrasions":       ("찰과상",    0),
    "Bruises":         ("타박상",    0),
    "Burns":           ("화상",      2),
    "Cut":             ("열상",      1),
    "Diabetic Wounds": ("감염_의심", 1),
    "Laseration":      ("열상",      1),   # 오타 폴더명
    "Normal":          None,               # 정상 이미지 — 건너뜀
    "Pressure Wounds": ("부종_염좌", 1),
    "Surgical Wounds": ("열상",      1),
    "Venous Wounds":   ("감염_의심", 1),
}

SEVERITY_LABEL = {0: "자가처치", 1: "일반병원", 2: "응급실"}


# ─────────────────────────────────────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────────────────────────────────────
def copy_images(images: list[Path], dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    for img in images:
        name = img.name
        if name in seen:
            name = f"{img.stem}_{img.parent.name}{img.suffix}"
        seen.add(name)
        dst = dest_dir / name
        if dst.exists():
            name = f"{img.stem}_{img.parent.parent.name}_{img.parent.name}{img.suffix}"
            dst = dest_dir / name
        shutil.copy2(img, dst)


def collect_from_source(root: Path, cls_map: dict) -> tuple[dict, dict]:
    """root 안의 폴더를 cls_map 으로 분류해 (type_images, severity_images) 반환"""
    type_images:     dict[str, list[Path]] = {}
    severity_images: dict[int,  list[Path]] = {}

    for cls_dir in sorted(root.iterdir()):
        if not cls_dir.is_dir():
            continue
        mapping = cls_map.get(cls_dir.name)
        if mapping is None:
            print(f"    skip: {cls_dir.name}")
            continue
        wt, sev = mapping
        images = [f for f in cls_dir.rglob("*") if f.suffix.lower() in IMG_EXTS]
        if not images:
            continue
        print(f"    {cls_dir.name:20s} → {wt:10s}  sev={sev}  ({len(images)}장)")
        type_images.setdefault(wt,  []).extend(images)
        severity_images.setdefault(sev, []).extend(images)

    return type_images, severity_images


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


# ─────────────────────────────────────────────────────────────────────────────
# Step 1. Source 1 위치 탐색
# ─────────────────────────────────────────────────────────────────────────────
def find_wound_main() -> Path:
    raw_dir = Path("wound_raw")
    known = set(SOURCE1_MAP.keys())
    for p in raw_dir.rglob("*"):
        if p.is_dir() and p.name in known:
            return p.parent
    raise FileNotFoundError(
        f"wound_raw/ 안에서 클래스 폴더를 찾지 못했습니다.\n예상: {sorted(known)}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Step 2. Source 2 위치 탐색
# ─────────────────────────────────────────────────────────────────────────────
def find_wound_classification() -> Path:
    base = Path("extra_raw/wound_classification")
    known = set(SOURCE2_MAP.keys())
    for p in base.rglob("*"):
        if p.is_dir() and p.name in known:
            return p.parent
    raise FileNotFoundError(
        f"extra_raw/wound_classification/ 안에서 클래스 폴더를 찾지 못했습니다."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Step 3. 병합 및 분할
# ─────────────────────────────────────────────────────────────────────────────
def reorganize():
    dataset_a = Path("dataset")
    dataset_b = Path("dataset_severity")

    if dataset_a.exists():
        shutil.rmtree(dataset_a)
        print("기존 dataset/ 삭제")
    if dataset_b.exists():
        shutil.rmtree(dataset_b)
        print("기존 dataset_severity/ 삭제")

    all_type:     dict[str, list[Path]] = {}
    all_severity: dict[int,  list[Path]] = {}

    def merge(t, s):
        for k, v in t.items(): all_type.setdefault(k, []).extend(v)
        for k, v in s.items(): all_severity.setdefault(k, []).extend(v)

    # ── Source 1 ──────────────────────────────────
    print("\n[Source 1] yasinpratomo/wound-dataset")
    src1 = find_wound_main()
    t, s = collect_from_source(src1, SOURCE1_MAP)
    merge(t, s)

    # ── Source 2 ──────────────────────────────────
    print("\n[Source 2] ibrahimfateen/wound-classification")
    src2 = find_wound_classification()
    t, s = collect_from_source(src2, SOURCE2_MAP)
    merge(t, s)

    # ── 총계 ──────────────────────────────────────
    total = sum(len(v) for v in all_type.values())
    print(f"\n병합 완료 — 총 {total}장")
    print("\n[진단명별]")
    for k, v in sorted(all_type.items()):
        print(f"  {k:12s}: {len(v):4d}장")
    print("\n[심각도별]")
    for k in sorted(all_severity):
        print(f"  {SEVERITY_LABEL[k]:6s}: {len(all_severity[k]):4d}장")

    # ── Model A: 진단명 데이터셋 ──────────────────
    print("\n📁 dataset/ (진단명) 생성 중...")
    for cls_name, images in all_type.items():
        random.shuffle(images)
        split = int(len(images) * TRAIN_RATIO)
        copy_images(images[:split], dataset_a / "train" / cls_name)
        copy_images(images[split:], dataset_a / "val"   / cls_name)

    # ── Model B: 심각도 데이터셋 ──────────────────
    print("📁 dataset_severity/ (심각도) 생성 중...")
    for sev_level, images in all_severity.items():
        cls_name = SEVERITY_LABEL[sev_level]
        random.shuffle(images)
        split = int(len(images) * TRAIN_RATIO)
        copy_images(images[:split], dataset_b / "train" / cls_name)
        copy_images(images[split:], dataset_b / "val"   / cls_name)

    # ── 요약 ────────────────────────────────────
    print("\n✅ 완료!")
    print_summary(dataset_a, "Model A — 진단명")
    print_summary(dataset_b, "Model B — 심각도")

    a_classes = sorted([d.name for d in (dataset_a / "train").iterdir()])
    b_classes = sorted([d.name for d in (dataset_b / "train").iterdir()])
    with open("class_names_preview.json", "w", encoding="utf-8") as f:
        json.dump({"wound_type": a_classes, "severity": b_classes},
                  f, ensure_ascii=False, indent=2)
    print(f"\n진단명 클래스: {a_classes}")
    print(f"심각도 클래스: {b_classes}")


if __name__ == "__main__":
    random.seed(42)
    reorganize()
