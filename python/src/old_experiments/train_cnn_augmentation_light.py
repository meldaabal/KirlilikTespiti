import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix


# ===============================
# AYARLAR
# ===============================

IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

TRAIN_CSV = "../results/tables/train_data.csv"
VAL_CSV = "../results/tables/val_data.csv"
TEST_CSV = "../results/tables/test_data.csv"

MODEL_SAVE_PATH = "../../models/cnn_aug_light.keras"

GRAPH_SAVE_PATH = "../../results/graphs/cnn_aug_light_accuracy_loss.png"
CONF_MATRIX_SAVE_PATH = "../../results/confusion_matrices/cnn_aug_light_confusion_matrix.png"
REPORT_SAVE_PATH = "../../results/tables/cnn_aug_light_report.csv"


# ===============================
# CSV OKU
# ===============================

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)
test_df = pd.read_csv(TEST_CSV)


# ===============================
# AUGMENTATION
# ===============================

train_datagen = ImageDataGenerator(
    rescale=1.0/255,

    rotation_range=15,
    zoom_range=0.10,

    width_shift_range=0.10,
    height_shift_range=0.10,

    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1.0/255)


train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col="image_path",
    y_col="label",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

val_generator = test_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col="image_path",
    y_col="label",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col="image_path",
    y_col="label",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)


num_classes = len(train_generator.class_indices)


# ===============================
# MODEL (Baseline Aynı)
# ===============================

model = Sequential([
    Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

    Conv2D(32, (3,3), activation="relu"),
    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D((2,2)),

    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D((2,2)),

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


# ===============================
# EĞİTİM
# ===============================

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)


# ===============================
# KAYDET
# ===============================

model.save(MODEL_SAVE_PATH)


# ===============================
# GRAFİK
# ===============================

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Aug Light Accuracy")
plt.legend(["train","val"])

plt.subplot(1,2,2)
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Aug Light Loss")
plt.legend(["train","val"])

plt.tight_layout()
plt.savefig(GRAPH_SAVE_PATH)
plt.show()


# ===============================
# TEST
# ===============================

test_loss, test_accuracy = model.evaluate(test_generator)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)


# ===============================
# RAPOR
# ===============================

pred = model.predict(test_generator)

y_pred = np.argmax(pred, axis=1)
y_true = test_generator.classes

labels = list(test_generator.class_indices.keys())

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8,6))
plt.imshow(cm)
plt.title("CNN Aug Light CM")
plt.colorbar()
plt.xticks(np.arange(len(labels)), labels, rotation=45)
plt.yticks(np.arange(len(labels)), labels)

for i in range(len(labels)):
    for j in range(len(labels)):
        plt.text(j, i, cm[i,j], ha="center", va="center")

plt.tight_layout()
plt.savefig(CONF_MATRIX_SAVE_PATH)
plt.show()


report = classification_report(
    y_true,
    y_pred,
    target_names=labels,
    output_dict=True
)

pd.DataFrame(report).transpose().to_csv(REPORT_SAVE_PATH)

print(classification_report(y_true, y_pred, target_names=labels))