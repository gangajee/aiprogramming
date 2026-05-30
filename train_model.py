"""
Model A: 외상 진단명 분류 모델 학습
  입력: 외상 이미지
  출력: 찰과상 / 열상 / 타박상 / 화상 / 출혈성_상처 / 감염_의심 / 부종_염좌

실행: python3 train_model.py
결과: wound_model.keras, class_names.json
"""

import os, json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report

DATASET_DIR   = "dataset"
MODEL_OUTPUT  = "wound_model.keras"
NAMES_OUTPUT  = "class_names.json"
IMG_SIZE      = (224, 224)
BATCH_SIZE    = 16
EPOCHS_FROZEN = 10
EPOCHS_FINE   = 10


def load_datasets():
    augmentation = keras.Sequential([
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomFlip("vertical"),
        keras.layers.RandomRotation(0.25),
        keras.layers.RandomZoom(0.15),
        keras.layers.RandomTranslation(0.10, 0.10),
        # 색상 과적합 방지: 붉은 기/밝기에 의존하지 않도록 범위 확대
        keras.layers.RandomBrightness(0.40),
        keras.layers.RandomContrast(0.40),
        # 색조 변환 — 붉은 사과·토마토 등 오인 방지
        keras.layers.Lambda(
            lambda x: tf.image.random_hue(x / 255.0, 0.08) * 255.0,
            name="random_hue",
        ),
        keras.layers.Lambda(
            lambda x: tf.image.random_saturation(x / 255.0, 0.6, 1.5) * 255.0,
            name="random_saturation",
        ),
    ], name="augmentation")
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
    print(f"클래스 ({len(class_names)}개): {class_names}")

    train_ds = (train_ds
        .map(lambda x, y: (augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
        .map(lambda x, y: (preprocess(x), y), num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE))
    val_ds = (val_ds
        .map(lambda x, y: (preprocess(x), y), num_parallel_calls=tf.data.AUTOTUNE)
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
    x       = keras.layers.Dense(128, activation="relu")(x)
    x       = keras.layers.Dropout(0.20)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="wound_classifier")


def train():
    print("=" * 55)
    print("  Model A: 외상 진단명 분류 모델 학습")
    print("=" * 55)

    train_ds, val_ds, class_names = load_datasets()
    num_classes = len(class_names)
    model = build_model(num_classes)
    model.summary()

    print(f"\n[Phase 1] 분류 헤드 학습 ({EPOCHS_FROZEN} epochs)")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_ds, validation_data=val_ds, epochs=EPOCHS_FROZEN,
        callbacks=[keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)],
    )

    print(f"\n[Phase 2] 미세조정 ({EPOCHS_FINE} epochs)")
    base_model = model.layers[1]
    base_model.trainable = True
    for layer in base_model.layers[:int(len(base_model.layers) * 0.70)]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_ds, validation_data=val_ds, epochs=EPOCHS_FINE,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
            keras.callbacks.ModelCheckpoint(MODEL_OUTPUT, save_best_only=True, monitor="val_accuracy"),
        ],
    )

    print("\n[최종 평가]")
    y_true, y_pred = [], []
    for x_batch, y_batch in val_ds:
        y_pred.extend(np.argmax(model.predict(x_batch, verbose=0), axis=1))
        y_true.extend(y_batch.numpy())

    print(classification_report(y_true, y_pred, target_names=class_names))
    print("혼동 행렬:")
    print(confusion_matrix(y_true, y_pred))

    with open(NAMES_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False)
    print(f"\n✅ 저장 완료: {MODEL_OUTPUT}, {NAMES_OUTPUT}")


if __name__ == "__main__":
    train()
