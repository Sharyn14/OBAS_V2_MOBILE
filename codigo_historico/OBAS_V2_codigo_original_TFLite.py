# OBAS V2 — CÓDIGO HISTÓRICO DE CONVERSIÓN TFLITE
# Extraído del notebook original.
# No ejecuta una nueva conversión.


# =======================================================
# CELDA ORIGINAL 198
# =======================================================

from pathlib import Path
import tensorflow as tf

BACKUP_DIR = Path(
    "/kaggle/input/datasets/santiagofallaamaya/obas-v2-final-backup"
)
MODEL_PATH = (
    BACKUP_DIR / "models" /
    "Student_MobileNetV2_UNet_V2_baseline_best.keras"
)

OUTPUT_DIR = Path("/kaggle/working/obas_v2_mobile")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TFLITE_PATH = OUTPUT_DIR / "student_baseline_float32.tflite"

print("=" * 75)
print("PASO 7 — CONVERSIÓN STUDENT BASELINE A TFLITE")
print("=" * 75)

print("TensorFlow:", tf.__version__)
print("Modelo encontrado:", MODEL_PATH.is_file())

# Cargar el modelo con las funciones personalizadas del proyecto.
# compile=False evita necesitar la función de pérdida para inferencia.
custom_objects = {}

try:
    from tensorflow.keras.utils import get_custom_objects
    custom_objects.update(get_custom_objects())
except Exception:
    pass

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects=custom_objects,
    compile=False
)

print("Modelo cargado:", model.name)
print("Entrada:", model.input_shape)
print("Salida:", model.output_shape)
print("Parámetros:", f"{model.count_params():,}")

# Conversión inicial, sin cuantización
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

TFLITE_PATH.write_bytes(tflite_model)

original_mb = MODEL_PATH.stat().st_size / (1024 ** 2)
tflite_mb = TFLITE_PATH.stat().st_size / (1024 ** 2)

print("\nConversión completada.")
print("Archivo:", TFLITE_PATH)
print(f"Modelo original: {original_mb:.2f} MB")
print(f"Modelo TFLite:   {tflite_mb:.2f} MB")
print(f"Reducción:       {(1 - tflite_mb / original_mb) * 100:.2f}%")


# =======================================================
# CELDA ORIGINAL 209
# =======================================================

# PASO 23 — Conversión TFLite con cuantización float16

from pathlib import Path
import tensorflow as tf
import pandas as pd

KERAS_PATH = Path(
    "/kaggle/input/datasets/santiagofallaamaya/"
    "obas-v2-final-backup/models/"
    "Student_MobileNetV2_UNet_V2_baseline_best.keras"
)

OUT_DIR = Path("/kaggle/working/obas_v2_mobile")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TFLITE_FLOAT16 = OUT_DIR / "student_baseline_float16.tflite"

assert KERAS_PATH.is_file(), f"No se encontró: {KERAS_PATH}"

model = tf.keras.models.load_model(KERAS_PATH, compile=False)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

tflite_float16 = converter.convert()
TFLITE_FLOAT16.write_bytes(tflite_float16)

original_bytes = KERAS_PATH.stat().st_size
float16_bytes = TFLITE_FLOAT16.stat().st_size

resumen = pd.DataFrame([{
    "Modelo": "Student baseline TFLite float16",
    "Tamaño_MB": float16_bytes / (1024 ** 2),
    "Tamaño_original_Keras_MB": original_bytes / (1024 ** 2),
    "Reduccion_vs_Keras_pct": (
        1 - float16_bytes / original_bytes
    ) * 100,
    "Ruta": str(TFLITE_FLOAT16)
}])

display(resumen.round(3))
print("Conversión completada:", TFLITE_FLOAT16)
