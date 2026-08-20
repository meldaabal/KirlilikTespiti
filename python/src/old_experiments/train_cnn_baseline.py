import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf


# ===============================
# 1. Temel ayarlar
# ===============================

IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

TRAIN_CSV = "../results/tables/train_data.csv"
VAL_CSV = "../results/tables/val_data.csv"
TEST_CSV = "../results/tables/test_data.csv"

MODEL_SAVE_PATH = "../../models/cnn_baseline_model.h5"

GRAPH_SAVE_PATH = "../../results/graphs/cnn_baseline_accuracy_loss.png"
CONF_MATRIX_SAVE_PATH = "../../results/confusion_matrices/cnn_baseline_confusion_matrix.png"
REPORT_SAVE_PATH = "../../results/tables/cnn_baseline_classification_report.csv"


# ===============================
# 2. Verileri oku
# ===============================

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)
test_df = pd.read_csv(TEST_CSV)

print("Train veri sayısı:", len(train_df))
print("Validation veri sayısı:", len(val_df))
print("Test veri sayısı:", len(test_df))


# ===============================
# 3. Görselleri modele hazırla
# ===============================

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)

val_test_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col="image_path",
    y_col="label",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

val_generator = val_test_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col="image_path",
    y_col="label",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

test_generator = val_test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col="image_path",
    y_col="label",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

num_classes = len(train_generator.class_indices)

print("\nSınıf indeksleri:")
print(train_generator.class_indices)


# ===============================
# 4. Baseline CNN modeli
# ===============================

model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),

    Flatten(),

    Dense(128, activation="relu"),
    Dropout(0.3),

    Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nModel özeti:")
model.summary()


# ===============================
# 5. Model eğitimi
# ===============================

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)


# ===============================
# 6. Modeli kaydet
# ===============================

model.save(MODEL_SAVE_PATH)
print("\nModel kaydedildi:")
print(MODEL_SAVE_PATH)


# ===============================
# 7. Accuracy / Loss grafiği kaydet
# ===============================

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("CNN Baseline Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("CNN Baseline Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig(GRAPH_SAVE_PATH)
plt.show()

print("\nAccuracy/Loss grafiği kaydedildi:")
print(GRAPH_SAVE_PATH)


# ===============================
# 8. Test verisi ile değerlendirme
# ===============================

test_loss, test_accuracy = model.evaluate(test_generator)

print("\nTest Loss:", test_loss)
print("Test Accuracy:", test_accuracy)


# ===============================
# 9. Confusion Matrix ve rapor
# ===============================

predictions = model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes

class_labels = list(test_generator.class_indices.keys())

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
plt.imshow(cm)
plt.title("CNN Baseline Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(np.arange(len(class_labels)), class_labels, rotation=45)
plt.yticks(np.arange(len(class_labels)), class_labels)
plt.colorbar()

for i in range(len(class_labels)):
    for j in range(len(class_labels)):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()
plt.savefig(CONF_MATRIX_SAVE_PATH)
plt.show()

print("\nConfusion matrix kaydedildi:")
print(CONF_MATRIX_SAVE_PATH)


report = classification_report(
    y_true,
    y_pred,
    target_names=class_labels,
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()
report_df.to_csv(REPORT_SAVE_PATH)

print("\nClassification report kaydedildi:")
print(REPORT_SAVE_PATH)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_labels))