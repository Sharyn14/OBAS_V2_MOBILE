# OBAS V2 — CÓDIGO HISTÓRICO DE DESTILACIÓN
# Extraído del notebook original.
# No ejecutar sin preparar las dependencias originales.


# ============================================================
# CELDA ORIGINAL 52
# ============================================================
# ============================================================
# PASO 15 — KNOWLEDGE DISTILLATION — CORREGIDO
# ============================================================

import os
import tensorflow as tf
import numpy as np

print("=" * 70)
print("PASO 15 — KNOWLEDGE DISTILLATION")
print("=" * 70)

# ------------------------------------------------------------
# 1. TEACHER BEST — CONGELADO
# ------------------------------------------------------------

TEACHER_PATH = (
    "/kaggle/working/models/"
    "Teacher_ResNet50_UNet_V2_best.keras"
)

teacher_kd = tf.keras.models.load_model(
    TEACHER_PATH,
    custom_objects={
        "combined_loss": combined_loss,
        "dice_metric": dice_metric,
        "iou_metric": iou_metric
    }
)

teacher_kd.trainable = False

print("Teacher cargado.")
print("Teacher parámetros:", f"{teacher_kd.count_params():,}")
print("Teacher entrenable:", teacher_kd.trainable)

# ------------------------------------------------------------
# 2. CREAR STUDENT NUEVO A PARTIR DE LA ARQUITECTURA
# ------------------------------------------------------------
# NO usamos clone_model porque Keras 3 intenta reconstruir
# la configuración de compilación y falla con combined_loss.

student_kd = tf.keras.models.Model(
    inputs=student_baseline.input,
    outputs=student_baseline.output,
    name="Student_MobileNetV2_UNet_V2_KD"
)

# ------------------------------------------------------------
# 3. COPIAR PESOS DEL STUDENT BASELINE
# ------------------------------------------------------------

student_kd.set_weights(
    student_baseline.get_weights()
)

print()
print("Student KD creado desde Student Baseline.")
print(
    "Student parámetros:",
    f"{student_kd.count_params():,}"
)

# ------------------------------------------------------------
# 4. CONFIGURACIÓN KD
# ------------------------------------------------------------

TEMPERATURE = 4.0
ALPHA = 0.8

print()
print("Configuración:")
print("Temperature T:", TEMPERATURE)
print("Alpha:", ALPHA)
print("Hard loss weight:", 1.0 - ALPHA)
print("Soft loss weight:", ALPHA)

# ------------------------------------------------------------
# 5. KL BINARIA
# ------------------------------------------------------------

def binary_kl_divergence(
    teacher_prob,
    student_prob,
    temperature=4.0
):

    eps = tf.keras.backend.epsilon()

    teacher_prob = tf.clip_by_value(
        tf.cast(teacher_prob, tf.float32),
        eps,
        1.0 - eps
    )

    student_prob = tf.clip_by_value(
        tf.cast(student_prob, tf.float32),
        eps,
        1.0 - eps
    )

    teacher_logit = tf.math.log(
        teacher_prob / (1.0 - teacher_prob)
    )

    student_logit = tf.math.log(
        student_prob / (1.0 - student_prob)
    )

    teacher_soft = tf.sigmoid(
        teacher_logit / temperature
    )

    student_soft = tf.sigmoid(
        student_logit / temperature
    )

    kl = (
        teacher_soft *
        tf.math.log(
            (teacher_soft + eps) /
            (student_soft + eps)
        )
        +
        (1.0 - teacher_soft) *
        tf.math.log(
            (1.0 - teacher_soft + eps) /
            (1.0 - student_soft + eps)
        )
    )

    return tf.reduce_mean(kl) * (temperature ** 2)


# ------------------------------------------------------------
# 6. CLASE KD
# ------------------------------------------------------------

class KnowledgeDistillationModel(tf.keras.Model):

    def __init__(
        self,
        student,
        teacher,
        temperature=4.0,
        alpha=0.8
    ):
        super().__init__()

        self.student = student
        self.teacher = teacher

        self.temperature = temperature
        self.alpha = alpha

        self.loss_tracker = tf.keras.metrics.Mean(
            name="loss"
        )

        self.hard_loss_tracker = tf.keras.metrics.Mean(
            name="hard_loss"
        )

        self.soft_loss_tracker = tf.keras.metrics.Mean(
            name="soft_loss"
        )

        self.accuracy_metric = tf.keras.metrics.BinaryAccuracy(
            name="accuracy"
        )

        self.precision_metric = tf.keras.metrics.Precision(
            name="precision"
        )

        self.recall_metric = tf.keras.metrics.Recall(
            name="recall"
        )

    @property
    def metrics(self):

        return [
            self.loss_tracker,
            self.hard_loss_tracker,
            self.soft_loss_tracker,
            self.accuracy_metric,
            self.precision_metric,
            self.recall_metric
        ]

    def train_step(self, data):

        images, y_true = data

        # ----------------------------------------------------
        # Teacher: siempre congelado
        # ----------------------------------------------------

        teacher_pred = self.teacher(
            images,
            training=False
        )

        # ----------------------------------------------------
        # Student
        # ----------------------------------------------------

        with tf.GradientTape() as tape:

            student_pred = self.student(
                images,
                training=True
            )

            # Pérdida supervisada real
            hard_loss = combined_loss(
                y_true,
                student_pred
            )

            # Pérdida de destilación
            soft_loss = binary_kl_divergence(
                teacher_pred,
                student_pred,
                self.temperature
            )

            # Pérdida total
            total_loss = (
                (1.0 - self.alpha) * hard_loss
                +
                self.alpha * soft_loss
            )

        # ----------------------------------------------------
        # Actualizar únicamente Student
        # ----------------------------------------------------

        gradients = tape.gradient(
            total_loss,
            self.student.trainable_variables
        )

        self.optimizer.apply_gradients(
            zip(
                gradients,
                self.student.trainable_variables
            )
        )

        # ----------------------------------------------------
        # Métricas
        # ----------------------------------------------------

        self.loss_tracker.update_state(
            total_loss
        )

        self.hard_loss_tracker.update_state(
            hard_loss
        )

        self.soft_loss_tracker.update_state(
            soft_loss
        )

        self.accuracy_metric.update_state(
            y_true,
            student_pred
        )

        self.precision_metric.update_state(
            y_true,
            student_pred
        )

        self.recall_metric.update_state(
            y_true,
            student_pred
        )

        return {
            "loss": self.loss_tracker.result(),
            "hard_loss": self.hard_loss_tracker.result(),
            "soft_loss": self.soft_loss_tracker.result(),
            "accuracy": self.accuracy_metric.result(),
            "precision": self.precision_metric.result(),
            "recall": self.recall_metric.result()
        }

    def call(
        self,
        inputs,
        training=False
    ):

        return self.student(
            inputs,
            training=training
        )


# ------------------------------------------------------------
# 7. CREAR MODELO KD
# ------------------------------------------------------------

kd_model = KnowledgeDistillationModel(
    student=student_kd,
    teacher=teacher_kd,
    temperature=TEMPERATURE,
    alpha=ALPHA
)

kd_model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-4
    )
)

# ------------------------------------------------------------
# 8. PRUEBA DE FORWARD
# ------------------------------------------------------------

sample_x = X_train_norm[:1]

teacher_test = teacher_kd(
    sample_x,
    training=False
)

student_test = student_kd(
    sample_x,
    training=False
)

kl_test = binary_kl_divergence(
    teacher_test,
    student_test,
    TEMPERATURE
)

print()
print("=" * 70)
print("VERIFICACIÓN")
print("=" * 70)

print("Teacher output:", teacher_test.shape)
print("Student output:", student_test.shape)
print("KL inicial:", float(kl_test))

print()
print("Teacher entrenable:", teacher_kd.trainable)
print("Student entrenable:", student_kd.trainable)

print()
print("=" * 70)
print("PASO 15 — COMPLETADO CORRECTAMENTE")
print("=" * 70)

# ============================================================
# CELDA ORIGINAL 54
# ============================================================
# ============================================================
# CORRECCIÓN PASO 16 — TEST_STEP PARA VALIDACIÓN KD
# ============================================================

class KnowledgeDistillationModel(tf.keras.Model):

    def __init__(
        self,
        student,
        teacher,
        temperature=4.0,
        alpha=0.8
    ):
        super().__init__()

        self.student = student
        self.teacher = teacher

        self.temperature = temperature
        self.alpha = alpha

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.hard_loss_tracker = tf.keras.metrics.Mean(name="hard_loss")
        self.soft_loss_tracker = tf.keras.metrics.Mean(name="soft_loss")

        self.accuracy_metric = tf.keras.metrics.BinaryAccuracy(
            name="accuracy"
        )

        self.precision_metric = tf.keras.metrics.Precision(
            name="precision"
        )

        self.recall_metric = tf.keras.metrics.Recall(
            name="recall"
        )

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.hard_loss_tracker,
            self.soft_loss_tracker,
            self.accuracy_metric,
            self.precision_metric,
            self.recall_metric
        ]

    # --------------------------------------------------------
    # TRAIN STEP
    # --------------------------------------------------------

    def train_step(self, data):

        images, y_true = data

        teacher_pred = self.teacher(
            images,
            training=False
        )

        with tf.GradientTape() as tape:

            student_pred = self.student(
                images,
                training=True
            )

            hard_loss = combined_loss(
                y_true,
                student_pred
            )

            soft_loss = binary_kl_divergence(
                teacher_pred,
                student_pred,
                self.temperature
            )

            total_loss = (
                (1.0 - self.alpha) * hard_loss
                +
                self.alpha * soft_loss
            )

        gradients = tape.gradient(
            total_loss,
            self.student.trainable_variables
        )

        self.optimizer.apply_gradients(
            zip(
                gradients,
                self.student.trainable_variables
            )
        )

        self.loss_tracker.update_state(total_loss)
        self.hard_loss_tracker.update_state(hard_loss)
        self.soft_loss_tracker.update_state(soft_loss)

        self.accuracy_metric.update_state(
            y_true,
            student_pred
        )

        self.precision_metric.update_state(
            y_true,
            student_pred
        )

        self.recall_metric.update_state(
            y_true,
            student_pred
        )

        return {
            "loss": self.loss_tracker.result(),
            "hard_loss": self.hard_loss_tracker.result(),
            "soft_loss": self.soft_loss_tracker.result(),
            "accuracy": self.accuracy_metric.result(),
            "precision": self.precision_metric.result(),
            "recall": self.recall_metric.result()
        }

    # --------------------------------------------------------
    # TEST STEP — VALIDACIÓN
    # --------------------------------------------------------

    def test_step(self, data):

        images, y_true = data

        # Teacher congelado
        teacher_pred = self.teacher(
            images,
            training=False
        )

        # Student
        student_pred = self.student(
            images,
            training=False
        )

        # Hard loss
        hard_loss = combined_loss(
            y_true,
            student_pred
        )

        # Soft loss
        soft_loss = binary_kl_divergence(
            teacher_pred,
            student_pred,
            self.temperature
        )

        # Loss total
        total_loss = (
            (1.0 - self.alpha) * hard_loss
            +
            self.alpha * soft_loss
        )

        # Actualizar métricas
        self.loss_tracker.update_state(
            total_loss
        )

        self.hard_loss_tracker.update_state(
            hard_loss
        )

        self.soft_loss_tracker.update_state(
            soft_loss
        )

        self.accuracy_metric.update_state(
            y_true,
            student_pred
        )

        self.precision_metric.update_state(
            y_true,
            student_pred
        )

        self.recall_metric.update_state(
            y_true,
            student_pred
        )

        return {
            "loss": self.loss_tracker.result(),
            "hard_loss": self.hard_loss_tracker.result(),
            "soft_loss": self.soft_loss_tracker.result(),
            "accuracy": self.accuracy_metric.result(),
            "precision": self.precision_metric.result(),
            "recall": self.recall_metric.result()
        }

    def call(
        self,
        inputs,
        training=False
    ):
        return self.student(
            inputs,
            training=training
        )


# ============================================================
# RECONSTRUIR MODELO KD
# ============================================================

kd_model = KnowledgeDistillationModel(
    student=student_kd,
    teacher=teacher_kd,
    temperature=TEMPERATURE,
    alpha=ALPHA
)

kd_model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-4
    )
)

print("=" * 70)
print("MODELO KD CORREGIDO")
print("=" * 70)

print("Teacher entrenable:", teacher_kd.trainable)
print("Student entrenable:", student_kd.trainable)
print("Student parámetros:", f"{student_kd.count_params():,}")
print("Temperature:", TEMPERATURE)
print("Alpha:", ALPHA)

print()
print("PASO 16 CORREGIDO — LISTO PARA ENTRENAR")

# ============================================================
# CELDA ORIGINAL 55
# ============================================================
# ============================================================
# PASO 16B — CONTINUAR ENTRENAMIENTO STUDENT KD
# ============================================================

import os
import tensorflow as tf

print("=" * 70)
print("PASO 16B — STUDENT KD: ÉPOCAS 2 → 50")
print("=" * 70)

KD_LOG_PATH = (
    "/kaggle/working/results/"
    "Student_KD_V2_training_log.csv"
)

KD_WEIGHTS_PATH = (
    "/kaggle/working/models/"
    "Student_MobileNetV2_UNet_V2_KD_best.weights.h5"
)

# ------------------------------------------------------------
# CALLBACK PARA GUARDAR MEJORES PESOS
# ------------------------------------------------------------

class KDCheckpoint(tf.keras.callbacks.Callback):

    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath
        self.best = float("inf")

    def on_epoch_end(self, epoch, logs=None):

        logs = logs or {}
        current = logs.get("val_loss")

        if current is None:
            return

        if current < self.best:

            self.best = current

            self.model.student.save_weights(
                self.filepath
            )

            print(
                f"\n✓ Mejor Student KD guardado — "
                f"epoch {epoch + 1} — "
                f"val_loss={current:.6f}"
            )


# ------------------------------------------------------------
# CALLBACKS
# ------------------------------------------------------------

kd_callbacks = [

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
        restore_best_weights=False,
        verbose=1
    ),

    tf.keras.callbacks.CSVLogger(
        KD_LOG_PATH,
        append=False
    ),

    KDCheckpoint(
        KD_WEIGHTS_PATH
    )
]

# ------------------------------------------------------------
# CONTINUAR DESDE EPOCH 2
# ------------------------------------------------------------

kd_history = kd_model.fit(
    train_ds_aug,
    validation_data=val_ds,
    initial_epoch=1,
    epochs=50,
    callbacks=kd_callbacks,
    verbose=1
)

print()
print("=" * 70)
print("STUDENT KD — ENTRENAMIENTO TERMINADO")
print("=" * 70)

print(
    "Épocas adicionales ejecutadas:",
    len(kd_history.history["loss"])
)

print()
print("Log:")
print(KD_LOG_PATH)

print()
print("Mejores pesos:")
print(KD_WEIGHTS_PATH)

print()
print("PASO 16B — COMPLETADO")