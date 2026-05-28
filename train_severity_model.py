"""
Model B: 심각도 분류 모델 학습
  입력: 외상 이미지
  출력: 자가처치(0) / 일반병원(1) / 응급실(2)

실행: python3 train_severity_model.py
결과: severity_model.keras
"""

import os, json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report

DATASET_DIR    = "dataset_severity"
MODEL_OUTPUT   = "severity_model.keras"
NAMES_OUTPUT   = "severity_class_names.json"
IMG_SIZE       = (224, 224)
BATCH_SIZE     = 16
EPOCHS_FROZEN  = 10
EPOCHS_FINETUNE = 10


def load_datasets():
    augmentation = keras.Sequential([
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.15),
        keras.layers.RandomZoom(0.10),
        keras.layers.RandomBrightness(0.20),
        keras.layers.RandomContrast(0.15),
    ])
    preprocess = keras.applications.mobilenet_v2.preprocess_input

    train_ds = keras.utils.image_dataset_from_directory(
        os.path.join(DATASET_DIR, "train"),
        image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        label_mode="int", shuffle=True, seed=42,
    )
    val_ds = keras.utils.image_dataset_from_directory(
        os.path.join(DATASET_DIR, "val"),
        image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        label_mode="int", shuffle=False,
    )
    class_names = train_ds.class_names
    print(f"심각도 클래스: {class_names}")

    train_ds = (train_ds
        .map(lambda x, y: (augmentation(x, training=True), y),
             num_parallel_calls=tf.data.AUTOTUNE)
        .map(lambda x, y: (preprocess(x), y),
             num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE))
    val_ds = (val_ds
        .map(lambda x, y: (preprocess(x), y),
             num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE))

    return train_ds, val_ds, class_names


def build_model(num_classes: int) -> keras.Model:
    base = keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False

    inputs  = keras.Input(shape=(*IMG_SIZE, 3))
    x       = base(inputs, training=False)
    x       = keras.layers.GlobalAveragePooling2D()(x)
    x       = keras.layers.Dropout(0.30)(x)
    x       = keras.layers.Dense(64, activation="relu")(x)
    x       = keras.layers.Dropout(0.20)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="severity_classifier")


def train():
    print("=" * 55)
    print("  Model B: 심각도 분류 모델 학습")
    print("=" * 55)

    train_ds, val_ds, class_names = load_datasets()
    num_classes = len(class_names)
    model = build_model(num_classes)

    # Phase 1: 헤드만 학습
    print(f"\n[Phase 1] 분류 헤드 학습 ({EPOCHS_FROZEN} epochs)")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FROZEN,
              callbacks=[keras.callbacks.EarlyStopping(patience=3,
                         restore_best_weights=True)])

    # Phase 2: 상위 레이어 미세조정
    print(f"\n[Phase 2] 미세조정 ({EPOCHS_FINETUNE} epochs)")
    base_model = model.layers[1]
    base_model.trainable = True
    fine_tune_from = int(len(base_model.layers) * 0.70)
    for layer in base_model.layers[:fine_tune_from]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINETUNE,
              callbacks=[
                  keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
                  keras.callbacks.ModelCheckpoint(
                      MODEL_OUTPUT, save_best_only=True, monitor="val_accuracy"),
              ])

    # 평가
    print("\n[최종 평가]")
    y_true, y_pred = [], []
    for x_batch, y_batch in val_ds:
        preds = model.predict(x_batch, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(y_batch.numpy())

    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    print("혼동 행렬:")
    header = f"  {'':10s}" + "".join(f"{n:>8s}" for n in class_names)
    print(header)
    for i, row_label in enumerate(class_names):
        print(f"  {row_label:10s}" + "".join(f"{cm[i][j]:8d}" for j in range(num_classes)))

    # 응급 환자 미감지율 (FNR)
    emergency_idx = class_names.index("응급실") if "응급실" in class_names else None
    if emergency_idx is not None:
        fn = sum(cm[emergency_idx]) - cm[emergency_idx][emergency_idx]
        fnr = fn / sum(cm[emergency_idx]) if sum(cm[emergency_idx]) > 0 else 0
        print(f"\n⚠️  응급 환자 미감지율 (FNR): {fnr:.1%}")

    with open(NAMES_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False)

    print(f"\n✅ 저장 완료: {MODEL_OUTPUT}, {NAMES_OUTPUT}")


if __name__ == "__main__":
    train()
