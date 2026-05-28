"""
외상 이미지 분류 모델 학습
MobileNetV2 전이학습 + 데이터 증강

실행: python3 train_model.py
결과: wound_model.keras (저장된 모델 파일)
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report
import json

# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────
DATASET_DIR  = "dataset"        # 데이터셋 루트 폴더
MODEL_OUTPUT = "wound_model.keras"
IMG_SIZE     = (224, 224)       # MobileNetV2 입력 크기
BATCH_SIZE   = 16
EPOCHS_FROZEN  = 10             # base 고정 학습
EPOCHS_FINETUNE = 10            # base 일부 해제 후 미세조정

CLASS_NAMES = [
    "찰과상",
    "열상",
    "타박상",
    "화상",
    "출혈성_상처",
    "감염_의심",
    "부종_염좌",
]


# ─────────────────────────────────────────────
# 1. 데이터 로드 + 증강
# ─────────────────────────────────────────────
def load_datasets():
    # 증강 레이어 (훈련셋에만 적용)
    augmentation = keras.Sequential([
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.15),
        keras.layers.RandomZoom(0.10),
        keras.layers.RandomBrightness(0.20),
        keras.layers.RandomContrast(0.15),
    ], name="augmentation")

    train_ds = keras.utils.image_dataset_from_directory(
        os.path.join(DATASET_DIR, "train"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=True,
        seed=42,
    )
    val_ds = keras.utils.image_dataset_from_directory(
        os.path.join(DATASET_DIR, "val"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False,
        seed=42,
    )

    # 실제 감지된 클래스명 저장
    class_names = train_ds.class_names
    print(f"\n감지된 클래스 ({len(class_names)}개): {class_names}")

    # 전처리: 픽셀 [0,255] → MobileNetV2 입력 범위 [-1,1]
    preprocess = keras.applications.mobilenet_v2.preprocess_input

    train_ds = (
        train_ds
        .map(lambda x, y: (augmentation(x, training=True), y),
             num_parallel_calls=tf.data.AUTOTUNE)
        .map(lambda x, y: (preprocess(x), y),
             num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        val_ds
        .map(lambda x, y: (preprocess(x), y),
             num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds, val_ds, class_names


# ─────────────────────────────────────────────
# 2. MobileNetV2 전이학습 모델 구성
# ─────────────────────────────────────────────
def build_model(num_classes: int) -> keras.Model:
    base = keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",      # ImageNet 사전학습 가중치
    )
    base.trainable = False       # 1단계: base 전체 고정

    inputs  = keras.Input(shape=(*IMG_SIZE, 3))
    x       = base(inputs, training=False)
    x       = keras.layers.GlobalAveragePooling2D()(x)
    x       = keras.layers.Dropout(0.30)(x)
    x       = keras.layers.Dense(128, activation="relu")(x)
    x       = keras.layers.Dropout(0.20)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs, name="wound_classifier")


# ─────────────────────────────────────────────
# 3. 학습
# ─────────────────────────────────────────────
def train():
    print("=" * 55)
    print("  외상 분류 모델 학습 시작")
    print("=" * 55)

    train_ds, val_ds, class_names = load_datasets()
    num_classes = len(class_names)
    model = build_model(num_classes)
    model.summary()

    # ── Phase 1: base 고정, 분류 헤드만 학습 ──────
    print(f"\n[Phase 1] 분류 헤드 학습 ({EPOCHS_FROZEN} epochs)")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_FROZEN,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        ],
    )

    # ── Phase 2: base 상위 30% 레이어 해제 후 미세조정 ──
    print(f"\n[Phase 2] 미세조정 ({EPOCHS_FINETUNE} epochs)")
    base_model = model.layers[1]           # MobileNetV2 레이어
    base_model.trainable = True
    fine_tune_from = int(len(base_model.layers) * 0.70)
    for layer in base_model.layers[:fine_tune_from]:
        layer.trainable = False            # 하위 70%는 계속 고정

    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),   # 작은 lr로 미세조정
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_FINETUNE,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
            keras.callbacks.ModelCheckpoint(
                MODEL_OUTPUT, save_best_only=True, monitor="val_accuracy"
            ),
        ],
    )

    # ── 평가 ──────────────────────────────────────
    print("\n[최종 평가]")
    y_true, y_pred = [], []
    for x_batch, y_batch in val_ds:
        preds = model.predict(x_batch, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(y_batch.numpy())

    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    print("혼동 행렬:")
    print(cm)

    # 클래스명 저장 (앱에서 사용)
    with open("class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False)

    print(f"\n모델 저장 완료: {MODEL_OUTPUT}")
    print(f"클래스 목록 저장 완료: class_names.json")


if __name__ == "__main__":
    train()
