"""
멀티소스 wound 데이터셋 병합 및 학습용 폴더 구조 생성

Sources (로컬에 다운로드 필요):
  wound_raw/       ← kaggle datasets download -d yasinpratomo/wound-dataset
  extra_raw/       ← kaggle datasets download -d ibrahimfateen/wound-classification -p extra_raw/

출력:
  dataset/          Model A 진단명 분류용
  dataset_severity/ Model B 심각도 분류용

실행: python3 prepare_data.py
"""

import shutil, random, json
from pathlib import Path

IMG_EXTS    = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TRAIN_RATIO = 0.80

# 소스별 클래스 매핑  →  (진단명, severity_level)
#   severity: 0=자가처치  1=일반병원  2=응급실  None=건너뜀
SOURCE1_MAP = {
    "Abrasions":    ("찰과상",      0),
    "Bruises":      ("타박상",      0),
    "Burns":        ("화상",        2),
    "Cut":          ("열상",        1),
    "Ingrown_nails":("부종_염좌",   0),
    "Laceration":   ("열상",        1),
    "Stab_wound":   ("출혈성_상처", 2),
}

SOURCE2_MAP = {
    "Abrasions":       ("찰과상",    0),
    "Bruises":         ("타박상",    0),
    "Burns":           ("화상",      2),
    "Cut":             ("열상",      1),
    "Diabetic Wounds": ("감염_의심", 1),
    "Laseration":      ("열상",      1),
    "Normal":          ("정상", None),   # 음성 샘플 — Model A에만 포함, Model B 제외
    "Pressure Wounds": ("부종_염좌", 1),
    "Surgical Wounds": ("열상",      1),
    "Venous Wounds":   ("감염_의심", 1),
}

SEVERITY_LABEL = {0: "자가처치", 1: "일반병원", 2: "응급실"}


def copy_images(images: list[Path], dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    for img in images:
        name = img.name
        if name in seen:
            name = f"{img.stem}_{img.parent.name}{img.suffix}"
        if (dest_dir / name).exists():
            name = f"{img.stem}_{img.parent.parent.name}_{img.parent.name}{img.suffix}"
        seen.add(name)
        shutil.copy2(img, dest_dir / name)


def collect(root: Path, cls_map: dict) -> tuple[dict, dict]:
    type_imgs: dict[str, list[Path]] = {}
    sev_imgs:  dict[int, list[Path]] = {}
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
        print(f"    {cls_dir.name:20s} → {wt:12s}  sev={sev}  ({len(images)}장)")
        type_imgs.setdefault(wt, []).extend(images)
        if sev is not None:   # 정상 클래스는 심각도 모델에 포함하지 않음
            sev_imgs.setdefault(sev, []).extend(images)
    return type_imgs, sev_imgs


def find_source(base: Path, known_names: set) -> Path:
    for p in base.rglob("*"):
        if p.is_dir() and p.name in known_names:
            return p.parent
    raise FileNotFoundError(f"{base} 안에서 클래스 폴더를 찾지 못했습니다.")


def print_summary(dataset_dir: Path, label: str):
    print(f"\n  [{label}]")
    for phase in ("train", "val"):
        d = dataset_dir / phase
        if not d.exists():
            continue
        print(f"  {phase}/")
        for cls_dir in sorted(d.iterdir()):
            print(f"    {cls_dir.name:20s} {len(list(cls_dir.glob('*'))):4d}장")


def reorganize():
    dataset_a = Path("dataset")
    dataset_b = Path("dataset_severity")

    for d in (dataset_a, dataset_b):
        if d.exists():
            shutil.rmtree(d)
            print(f"기존 {d}/ 삭제")

    all_type: dict[str, list[Path]] = {}
    all_sev:  dict[int, list[Path]] = {}

    def merge(t, s):
        for k, v in t.items(): all_type.setdefault(k, []).extend(v)
        for k, v in s.items(): all_sev.setdefault(k, []).extend(v)

    print("\n[Source 1] yasinpratomo/wound-dataset")
    merge(*collect(find_source(Path("wound_raw"), set(SOURCE1_MAP)), SOURCE1_MAP))

    print("\n[Source 2] ibrahimfateen/wound-classification")
    merge(*collect(find_source(Path("extra_raw/wound_classification"), set(SOURCE2_MAP)), SOURCE2_MAP))

    total = sum(len(v) for v in all_type.values())
    print(f"\n병합 완료 — 총 {total}장")
    print("\n[진단명별]")
    for k, v in sorted(all_type.items()):
        print(f"  {k:12s}: {len(v):4d}장")
    print("\n[심각도별]")
    for k in sorted(all_sev):
        print(f"  {SEVERITY_LABEL[k]:6s}: {len(all_sev[k]):4d}장")

    print("\n📁 dataset/ (진단명) 생성 중...")
    for cls_name, images in all_type.items():
        random.shuffle(images)
        split = int(len(images) * TRAIN_RATIO)
        copy_images(images[:split], dataset_a / "train" / cls_name)
        copy_images(images[split:], dataset_a / "val"   / cls_name)

    print("📁 dataset_severity/ (심각도) 생성 중...")
    for sev_level, images in all_sev.items():
        cls_name = SEVERITY_LABEL[sev_level]
        random.shuffle(images)
        split = int(len(images) * TRAIN_RATIO)
        copy_images(images[:split], dataset_b / "train" / cls_name)
        copy_images(images[split:], dataset_b / "val"   / cls_name)

    print("\n✅ 완료!")
    print_summary(dataset_a, "Model A — 진단명")
    print_summary(dataset_b, "Model B — 심각도")

    a_classes = sorted(d.name for d in (dataset_a / "train").iterdir())
    b_classes = sorted(d.name for d in (dataset_b / "train").iterdir())
    with open("class_names_preview.json", "w", encoding="utf-8") as f:
        json.dump({"wound_type": a_classes, "severity": b_classes}, f, ensure_ascii=False, indent=2)
    print(f"\n진단명 클래스: {a_classes}")
    print(f"심각도 클래스: {b_classes}")


if __name__ == "__main__":
    random.seed(42)
    reorganize()
