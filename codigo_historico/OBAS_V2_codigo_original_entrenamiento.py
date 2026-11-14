# OBAS V2 — CÓDIGO HISTÓRICO DE ENTRENAMIENTO
# Recuperado del notebook original.
# No ejecutar sin preparar el entorno original.


# =======================================================
# CELDA 9 — Carga de datos
# =======================================================
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# ============================================================
# RUTAS KAGGLE
# ============================================================

DATASET_DIR = Path(
    "/kaggle/input/datasets/santiagofallaamaya/obas-v2-burned-area-segmentation"
)

PATCH_DIR = DATASET_DIR / "OBAS_V2_common_patches" / "common_patches"

INDEX_PATH = DATASET_DIR / "patch_index_spatial_v2.csv"

# ============================================================
# ÍNDICE
# ============================================================

df = pd.read_csv(INDEX_PATH)

print("=" * 70)
print("CARGANDO PATCHES DESDE KAGGLE")
print("=" * 70)

print("Train:", (df["split"] == "train").sum())
print("Val:  ", (df["split"] == "val").sum())
print("Test: ", (df["split"] == "test").sum())


# ============================================================
# FUNCIÓN DE CARGA
# ============================================================

def load_patch(row):
    image_path = PATCH_DIR / row["image_file"]
    mask_path = PATCH_DIR / row["mask_file"]

    image = np.load(image_path).astype(np.float32)
    mask = np.load(mask_path).astype(np.float32)

    # CHW -> HWC
    if image.shape == (3, 128, 128):
        image = np.transpose(image, (1, 2, 0))

    # HWC -> HWC por seguridad
    elif image.shape != (128, 128, 3):
        raise ValueError(
            f"Forma inesperada: {image.shape} | {image_path}"
        )

    # Máscara
    if mask.shape != (128, 128):
        raise ValueError(
            f"Forma inesperada máscara: {mask.shape} | {mask_path}"
        )

    mask = (mask >= 0.5).astype(np.float32)

    return image, mask


# ============================================================
# CARGADOR PARA UN SPLIT
# ============================================================

def load_split(split_name, workers=16):

    subset = df[df["split"] == split_name].reset_index(drop=True)

    print(f"\nCargando {split_name}: {len(subset)} patches...")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(
            executor.map(load_patch, [row for _, row in subset.iterrows()])
        )

    X = np.stack([r[0] for r in results], axis=0)
    Y = np.stack([r[1] for r in results], axis=0)

    Y = Y[..., np.newaxis]

    print(f"{split_name}:")
    print("  X:", X.shape, X.dtype)
    print("  Y:", Y.shape, Y.dtype)
    print("  X min/max:", X.min(), X.max())
    print("  Y valores:", np.unique(Y))

    return X, Y


# ============================================================
# CARGA
# ============================================================

X_train, Y_train = load_split("train")
X_val, Y_val = load_split("val")
X_test, Y_test = load_split("test")

print("\n" + "=" * 70)
print("CARGA COMPLETADA")
print("=" * 70)

# =======================================================
# CELDA 11 — Normalización
# =======================================================
# ============================================================
# PASO 4 — NORMALIZACIÓN V2
# Estadísticas calculadas SOLO con TRAIN
# ============================================================

print("=" * 70)
print("NORMALIZACIÓN V2")
print("=" * 70)

# Percentiles calculados únicamente sobre TRAIN
p01 = np.percentile(X_train, 1)
p99 = np.percentile(X_train, 99)

print("TRAIN P01:", p01)
print("TRAIN P99:", p99)

# Normalización robusta
X_train_norm = np.clip(
    (X_train - p01) / (p99 - p01),
    0.0,
    1.0
).astype(np.float32)

X_val_norm = np.clip(
    (X_val - p01) / (p99 - p01),
    0.0,
    1.0
).astype(np.float32)

X_test_norm = np.clip(
    (X_test - p01) / (p99 - p01),
    0.0,
    1.0
).astype(np.float32)

print("\nRANGOS DESPUÉS DE NORMALIZAR")
print("-" * 70)

print(
    "TRAIN:",
    X_train_norm.min(),
    X_train_norm.max(),
    X_train_norm.mean(),
    X_train_norm.std()
)

print(
    "VAL:",
    X_val_norm.min(),
    X_val_norm.max(),
    X_val_norm.mean(),
    X_val_norm.std()
)

print(
    "TEST:",
    X_test_norm.min(),
    X_test_norm.max(),
    X_test_norm.mean(),
    X_test_norm.std()
)

print("\nFORMAS")
print("X_train_norm:", X_train_norm.shape)
print("X_val_norm:  ", X_val_norm.shape)
print("X_test_norm: ", X_test_norm.shape)

print("\n" + "=" * 70)
print("NORMALIZACIÓN COMPLETADA")
print("=" * 70)

# =======================================================
# CELDA 14 — Preparación y aumentación
# =======================================================
import tensorflow as tf

print("=" * 70)
print("PASO 5 — TF.DATA + AUGMENTACIÓN")
print("=" * 70)

BATCH_SIZE = 16
AUTOTUNE = tf.data.AUTOTUNE

# ------------------------------------------------------------
# DATASETS BASE
# ------------------------------------------------------------

train_ds = tf.data.Dataset.from_tensor_slices(
    (X_train_norm, Y_train)
)

val_ds = tf.data.Dataset.from_tensor_slices(
    (X_val_norm, Y_val)
)

test_ds = tf.data.Dataset.from_tensor_slices(
    (X_test_norm, Y_test)
)

# ------------------------------------------------------------
# AUGMENTACIÓN
# Espacial: imagen + máscara juntas
# Fotométrica: solamente imagen
# ------------------------------------------------------------

def augment_batch(image, mask):

    # Unir para que la misma transformación espacial
    # se aplique exactamente a imagen y máscara
    combined = tf.concat([image, mask], axis=-1)

    combined = tf.image.random_flip_left_right(combined)
    combined = tf.image.random_flip_up_down(combined)

    # Rotación 90° aleatoria
    k = tf.random.uniform(
        shape=[],
        minval=0,
        maxval=4,
        dtype=tf.int32
    )

    combined = tf.image.rot90(combined, k=k)

    image = combined[..., :3]
    mask = combined[..., 3:]

    # --------------------------------------------------------
    # Transformaciones fotométricas SOLO sobre imagen
    # --------------------------------------------------------

    image = tf.image.random_brightness(
        image,
        max_delta=0.10
    )

    image = tf.image.random_contrast(
        image,
        lower=0.90,
        upper=1.10
    )

    image = tf.clip_by_value(
        image,
        0.0,
        1.0
    )

    mask = tf.cast(
        mask >= 0.5,
        tf.float32
    )

    return image, mask


# ------------------------------------------------------------
# PIPELINE
# ------------------------------------------------------------

train_ds_aug = (
    train_ds
    .shuffle(
        buffer_size=len(X_train_norm),
        seed=42,
        reshuffle_each_iteration=True
    )
    .batch(BATCH_SIZE)
    .map(
        augment_batch,
        num_parallel_calls=AUTOTUNE
    )
    .prefetch(AUTOTUNE)
)

val_ds = (
    val_ds
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

test_ds = (
    test_ds
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

# ------------------------------------------------------------
# VERIFICACIÓN
# ------------------------------------------------------------

sample_images, sample_masks = next(iter(train_ds_aug))

print("Batch imagen:", sample_images.shape)
print("Batch máscara:", sample_masks.shape)

print("Imagen dtype:", sample_images.dtype)
print("Máscara dtype:", sample_masks.dtype)

print("Imagen min/max:",
      float(tf.reduce_min(sample_images)),
      float(tf.reduce_max(sample_images)))

print("Máscara valores:",
      tf.unique(tf.reshape(sample_masks, [-1])).y.numpy())

print("=" * 70)
print("TF.DATA + AUGMENTACIÓN CORRECTOS")
print("=" * 70)

# =======================================================
# CELDA 19 — Arquitectura Teacher
# =======================================================
# ============================================================
# PASO 6 — TEACHER V2: ResNet50 + U-Net
# ============================================================

import tensorflow as tf
from tensorflow.keras import layers, Model

print("=" * 70)
print("PASO 6 — CONSTRUCCIÓN DEL TEACHER RESNET50-U-NET V2")
print("=" * 70)

# ------------------------------------------------------------
# 1. ENTRADA
# ------------------------------------------------------------

INPUT_SHAPE = (128, 128, 3)

inputs = layers.Input(
    shape=INPUT_SHAPE,
    name="input_image"
)

# ------------------------------------------------------------
# 2. ENCODER — ResNet50 ImageNet
# ------------------------------------------------------------

encoder = tf.keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_tensor=inputs
)

# Teacher completo entrenable
encoder.trainable = True

# ------------------------------------------------------------
# 3. SKIP CONNECTIONS
# ------------------------------------------------------------

skip1 = encoder.get_layer("conv1_relu").output
skip2 = encoder.get_layer("conv2_block3_out").output
skip3 = encoder.get_layer("conv3_block4_out").output
skip4 = encoder.get_layer("conv4_block6_out").output

# Características profundas
bridge = encoder.get_layer("conv5_block3_out").output

print("Encoder ResNet50 cargado correctamente.")
print("Bridge:", bridge.shape)
print("Skip 1:", skip1.shape)
print("Skip 2:", skip2.shape)
print("Skip 3:", skip3.shape)
print("Skip 4:", skip4.shape)

# ------------------------------------------------------------
# 4. DECODER
# ------------------------------------------------------------

# Decoder 1: 4x4 -> 8x8
x = layers.UpSampling2D(
    size=(2, 2),
    interpolation="bilinear",
    name="decoder1_upsample"
)(bridge)

x = layers.Concatenate(
    name="decoder1_concat"
)([x, skip4])

x = layers.Conv2D(
    256,
    kernel_size=3,
    padding="same",
    activation="relu",
    name="decoder1_conv"
)(x)

# Decoder 2: 8x8 -> 16x16
x = layers.UpSampling2D(
    size=(2, 2),
    interpolation="bilinear",
    name="decoder2_upsample"
)(x)

x = layers.Concatenate(
    name="decoder2_concat"
)([x, skip3])

x = layers.Conv2D(
    128,
    kernel_size=3,
    padding="same",
    activation="relu",
    name="decoder2_conv"
)(x)

# Decoder 3: 16x16 -> 32x32
x = layers.UpSampling2D(
    size=(2, 2),
    interpolation="bilinear",
    name="decoder3_upsample"
)(x)

x = layers.Concatenate(
    name="decoder3_concat"
)([x, skip2])

x = layers.Conv2D(
    64,
    kernel_size=3,
    padding="same",
    activation="relu",
    name="decoder3_conv"
)(x)

# Decoder 4: 32x32 -> 64x64
x = layers.UpSampling2D(
    size=(2, 2),
    interpolation="bilinear",
    name="decoder4_upsample"
)(x)

x = layers.Concatenate(
    name="decoder4_concat"
)([x, skip1])

x = layers.Conv2D(
    32,
    kernel_size=3,
    padding="same",
    activation="relu",
    name="decoder4_conv"
)(x)

# ------------------------------------------------------------
# 5. RECONSTRUCCIÓN FINAL: 64x64 -> 128x128
# ------------------------------------------------------------

x = layers.UpSampling2D(
    size=(2, 2),
    interpolation="bilinear",
    name="final_upsample"
)(x)

x = layers.Conv2D(
    16,
    kernel_size=3,
    padding="same",
    activation="relu",
    name="final_conv"
)(x)

# ------------------------------------------------------------
# 6. SALIDA BINARIA
# ------------------------------------------------------------

outputs = layers.Conv2D(
    1,
    kernel_size=1,
    activation="sigmoid",
    name="burned_area_probability"
)(x)

# ------------------------------------------------------------
# 7. MODELO FINAL
# ------------------------------------------------------------

teacher = Model(
    inputs=inputs,
    outputs=outputs,
    name="Teacher_ResNet50_UNet_V2"
)

print()
print("=" * 70)
print("TEACHER V2 CONSTRUIDO")
print("=" * 70)

print("Entrada:", teacher.input_shape)
print("Salida:", teacher.output_shape)
print("Parámetros:", f"{teacher.count_params():,}")

# Verificación rápida
sample_output = teacher(
    tf.zeros((1, 128, 128, 3), dtype=tf.float32),
    training=False
)

print("Salida de prueba:", sample_output.shape)
print("Rango salida:", 
      float(tf.reduce_min(sample_output)),
      float(tf.reduce_max(sample_output)))

print()
print("PASO 6 — COMPLETADO")

# =======================================================
# CELDA 26 — Pérdida y métricas
# =======================================================
# ============================================================
# PASO 7 — LOSS + MÉTRICAS + COMPILACIÓN
# ============================================================

import tensorflow as tf

print("=" * 70)
print("PASO 7 — LOSS + MÉTRICAS + COMPILACIÓN")
print("=" * 70)


# ------------------------------------------------------------
# Dice
# ------------------------------------------------------------

def dice_metric(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

    intersection = tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)

    dice = (2.0 * intersection + 1e-7) / (denominator + 1e-7)

    return dice


# ------------------------------------------------------------
# IoU
# ------------------------------------------------------------

def iou_metric(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    y_pred = tf.cast(y_pred >= 0.5, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred)
    union = (
        tf.reduce_sum(y_true)
        + tf.reduce_sum(y_pred)
        - intersection
    )

    return (intersection + 1e-7) / (union + 1e-7)


# ------------------------------------------------------------
# Dice Loss
# ------------------------------------------------------------

def dice_loss(y_true, y_pred):
    return 1.0 - dice_metric(y_true, y_pred)


# ------------------------------------------------------------
# Binary Cross Entropy + Dice
# ------------------------------------------------------------

def combined_loss(y_true, y_pred):

    bce = tf.reduce_mean(
        tf.keras.losses.binary_crossentropy(
            y_true,
            y_pred
        )
    )

    dloss = dice_loss(
        y_true,
        y_pred
    )

    return 0.5 * bce + 0.5 * dloss


# ------------------------------------------------------------
# Optimizer
# ------------------------------------------------------------

optimizer = tf.keras.optimizers.Adam(
    learning_rate=1e-4
)


# ------------------------------------------------------------
# Compilar
# ------------------------------------------------------------

teacher.compile(
    optimizer=optimizer,
    loss=combined_loss,
    metrics=[
        tf.keras.metrics.BinaryAccuracy(
            name="accuracy"
        ),
        tf.keras.metrics.Precision(
            name="precision"
        ),
        tf.keras.metrics.Recall(
            name="recall"
        ),
        dice_metric,
        iou_metric
    ]
)


print()
print("Modelo:", teacher.name)
print("Loss: 0.5 BCE + 0.5 Dice")
print("Optimizer: Adam")
print("Learning rate: 1e-4")
print()
print("PASO 7 — COMPILACIÓN COMPLETADA")

# =======================================================
# CELDA 35 — Entrenamiento Teacher
# =======================================================
# ============================================================
# PASO 10 — ENTRENAMIENTO TEACHER V2
# ============================================================

import time
import tensorflow as tf

print("=" * 70)
print("PASO 10 — ENTRENAMIENTO TEACHER V2")
print("=" * 70)

# ------------------------------------------------------------
# CONFIGURACIÓN
# ------------------------------------------------------------

EPOCHS = 50
BATCH_SIZE = 16

print("Modelo:", teacher.name)
print("Parámetros:", f"{teacher.count_params():,}")
print("Batch size:", BATCH_SIZE)
print("Épocas:", EPOCHS)
print("GPUs:", tf.config.list_physical_devices("GPU"))

print()
print("Train batches:", tf.data.experimental.cardinality(train_ds_aug).numpy())
print("Validation batches:", tf.data.experimental.cardinality(val_ds).numpy())

print()
print("=" * 70)
print("INICIANDO ENTRENAMIENTO")
print("50 ÉPOCAS OBLIGATORIAS")
print("=" * 70)

# ------------------------------------------------------------
# TIEMPO DE INICIO
# ------------------------------------------------------------

start_time = time.time()

# ------------------------------------------------------------
# ENTRENAMIENTO
# ------------------------------------------------------------

history_teacher = teacher.fit(
    train_ds_aug,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# ------------------------------------------------------------
# TIEMPO FINAL
# ------------------------------------------------------------

elapsed_time = time.time() - start_time

print()
print("=" * 70)
print("ENTRENAMIENTO FINALIZADO")
print("=" * 70)

print(
    f"Tiempo total: {elapsed_time / 3600:.2f} horas"
)

print(
    "Épocas ejecutadas:",
    len(history_teacher.history["loss"])
)

print()
print("Métricas finales:")

for metric_name in history_teacher.history:
    values = history_teacher.history[metric_name]

    print(
        f"{metric_name:20s}: "
        f"{values[-1]:.6f}"
    )

print()
print("=" * 70)
print("PASO 10 — COMPLETADO")
print("=" * 70)

# =======================================================
# CELDA 41 — Arquitectura Student
# =======================================================
# ============================================================
# PASO 12 — STUDENT LIGERO: MobileNetV2-U-Net
# ============================================================

import tensorflow as tf
from tensorflow.keras import layers, Model

print("=" * 70)
print("PASO 12 — CONSTRUCCIÓN DEL STUDENT V2")
print("=" * 70)

# ------------------------------------------------------------
# 1. INPUT
# ------------------------------------------------------------

student_input = layers.Input(
    shape=(128, 128, 3),
    name="student_input"
)

# ------------------------------------------------------------
# 2. ENCODER MobileNetV2
# ------------------------------------------------------------

mobilenet = tf.keras.applications.MobileNetV2(
    input_shape=(128, 128, 3),
    include_top=False,
    weights="imagenet"
)

mobilenet.trainable = True

# Features espaciales para U-Net
skip1 = mobilenet.get_layer("block_1_expand_relu").output
skip2 = mobilenet.get_layer("block_3_expand_relu").output
skip3 = mobilenet.get_layer("block_6_expand_relu").output
skip4 = mobilenet.get_layer("block_13_expand_relu").output
bridge = mobilenet.get_layer("out_relu").output

encoder = Model(
    inputs=mobilenet.input,
    outputs=[
        skip1,
        skip2,
        skip3,
        skip4,
        bridge
    ],
    name="MobileNetV2_Encoder"
)

s1, s2, s3, s4, b = encoder(student_input)

# ------------------------------------------------------------
# 3. DECODER
# ------------------------------------------------------------

def decoder_block(x, skip, filters, name):
    x = layers.UpSampling2D(
        size=(2, 2),
        interpolation="bilinear",
        name=f"{name}_upsample"
    )(x)

    x = layers.Concatenate(
        name=f"{name}_concat"
    )([x, skip])

    x = layers.Conv2D(
        filters,
        3,
        padding="same",
        activation="relu",
        name=f"{name}_conv1"
    )(x)

    x = layers.BatchNormalization(
        name=f"{name}_bn1"
    )(x)

    x = layers.Conv2D(
        filters,
        3,
        padding="same",
        activation="relu",
        name=f"{name}_conv2"
    )(x)

    x = layers.BatchNormalization(
        name=f"{name}_bn2"
    )(x)

    return x


x = decoder_block(
    b, s4, 256, "decoder1"
)

x = decoder_block(
    x, s3, 128, "decoder2"
)

x = decoder_block(
    x, s2, 64, "decoder3"
)

x = decoder_block(
    x, s1, 32, "decoder4"
)

x = layers.UpSampling2D(
    size=(2, 2),
    interpolation="bilinear",
    name="decoder_final_upsample"
)(x)

x = layers.Conv2D(
    16,
    3,
    padding="same",
    activation="relu",
    name="decoder_final_conv"
)(x)

student_output = layers.Conv2D(
    1,
    1,
    padding="same",
    activation="sigmoid",
    name="burned_area_probability"
)(x)

student = Model(
    inputs=student_input,
    outputs=student_output,
    name="Student_MobileNetV2_UNet_V2"
)

# ------------------------------------------------------------
# 4. MOSTRAR ARQUITECTURA
# ------------------------------------------------------------

student.summary()

print()
print("=" * 70)
print("PARÁMETROS")
print("=" * 70)

teacher_params = teacher_best.count_params()
student_params = student.count_params()

print(f"Teacher : {teacher_params:,}")
print(f"Student : {student_params:,}")

print(
    f"Reducción de parámetros: "
    f"{teacher_params / student_params:.2f}x"
)

print(
    f"Compresión aproximada: "
    f"{100 * (1 - student_params / teacher_params):.2f}%"
)

# ------------------------------------------------------------
# 5. PRUEBA DE FORMA
# ------------------------------------------------------------

sample = X_train_norm[:1]

student_pred = student(sample, training=False)

print()
print("Entrada :", sample.shape)
print("Salida  :", student_pred.shape)
print(
    "Rango salida:",
    float(tf.reduce_min(student_pred)),
    "→",
    float(tf.reduce_max(student_pred))
)

print()
print("PASO 12 — COMPLETADO")

# =======================================================
# CELDA 44 — Entrenamiento Student baseline
# =======================================================
# ============================================================
# PASO 13 — STUDENT BASELINE
# ============================================================

import os
import tensorflow as tf

print("=" * 70)
print("PASO 13 — ENTRENAMIENTO STUDENT BASELINE")
print("=" * 70)

# ------------------------------------------------------------
# CONFIGURACIÓN
# ------------------------------------------------------------

STUDENT_BASELINE_PATH = (
    "/kaggle/working/models/"
    "Student_MobileNetV2_UNet_V2_baseline_best.keras"
)

STUDENT_BASELINE_LOG = (
    "/kaggle/working/results/"
    "Student_Baseline_V2_training_log.csv"
)

os.makedirs("/kaggle/working/models", exist_ok=True)
os.makedirs("/kaggle/working/results", exist_ok=True)

BATCH_SIZE = 16
EPOCHS = 50

# ------------------------------------------------------------
# COMPILAR STUDENT
# ------------------------------------------------------------

student.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-4
    ),
    loss=combined_loss,
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        dice_metric,
        iou_metric
    ]
)

# ------------------------------------------------------------
# CALLBACKS
# ------------------------------------------------------------

callbacks_student = [

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=4,
        min_lr=1e-7,
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=8,
        start_from_epoch=50,
        restore_best_weights=True,
        verbose=1
    ),

    tf.keras.callbacks.ModelCheckpoint(
        STUDENT_BASELINE_PATH,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),

    tf.keras.callbacks.CSVLogger(
        STUDENT_BASELINE_LOG,
        append=False
    )
]

# ------------------------------------------------------------
# VERIFICAR BATCH
# ------------------------------------------------------------

xb, yb = next(iter(train_ds_aug))

print()
print("Batch de entrenamiento:")
print("X:", xb.shape, xb.dtype)
print("Y:", yb.shape, yb.dtype)
print(
    "X rango:",
    float(tf.reduce_min(xb)),
    "→",
    float(tf.reduce_max(xb))
)
print(
    "Y valores:",
    tf.unique(tf.reshape(yb, [-1]))[0].numpy()
)

# ------------------------------------------------------------
# ENTRENAMIENTO
# ------------------------------------------------------------

print()
print("=" * 70)
print("INICIANDO STUDENT BASELINE — 50 ÉPOCAS")
print("=" * 70)

student_history = student.fit(
    train_ds_aug,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks_student,
    verbose=1
)

print()
print("=" * 70)
print("STUDENT BASELINE — ENTRENAMIENTO TERMINADO")
print("=" * 70)

print("Checkpoint:")
print(STUDENT_BASELINE_PATH)

print("Log:")
print(STUDENT_BASELINE_LOG)

print(
    "Épocas ejecutadas:",
    len(student_history.history["loss"])
)