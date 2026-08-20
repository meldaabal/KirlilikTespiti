import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    GlobalAveragePooling2D,
    Dense,
    Dropout
)
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# ===============================
# DENEY AYARLARI
# ===============================

EXPERIMENT_NAME = "mobilenetv2_transfer_learning"

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.0001

TRAIN_CSV = "../results/tables/train_data.csv"
VAL_CSV = "../results/tables/val_data.csv"
TEST_CSV = "../results/tables/test_data.csv"

MODEL_SAVE_PATH = f"../models/{EXPERIMENT_NAME}.keras"
GRAPH_SAVE_PATH = f"../results/graphs/{EXPERIMENT_NAME}_graph.png"
CONF_MATRIX_SAVE_PATH = f"../results/confusion_matrices/{EXPERIMENT_NAME}_cm.png"
REPORT_SAVE_PATH = f"../results/tables/{EXPERIMENT_NAME}_report.csv"
SUMMARY_SAVE_PATH = "../results/tables/all_experiments_results.csv"


# ===============================
# VERİYİ OKU
# ===============================

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)
test_df = pd.read_csv(TEST_CSV)

print("Deney:", EXPERIMENT_NAME)
print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))


# ===============================
# DATA GENERATOR
# ===============================

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=15,
    zoom_range=0.10,
    width_shift_range=0.10,
    height_shift_range=0.10,
    horizontal_flip=True
)

val_test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
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
# MOBILENETV2 MODELİ
# ===============================

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# İlk aşamada hazır modeli donduruyoruz
base_model.trainable = False

inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))

x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
outputs = Dense(num_classes, activation="softmax")(x)

model = Model(inputs, outputs)

model.compile(
    optimizer=RMSprop(learning_rate=LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# ===============================
# CALLBACKS
# ===============================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    MODEL_SAVE_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)


# ===============================
# EĞİTİM
# ===============================

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[early_stop, checkpoint]
)


# ===============================
# GRAFİK
# ===============================

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title(f"{EXPERIMENT_NAME} Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title(f"{EXPERIMENT_NAME} Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig(GRAPH_SAVE_PATH)
plt.show()


# ===============================
# TEST
# ===============================

test_loss, test_acc = model.evaluate(test_generator)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_acc)


# ===============================
# REPORT
# ===============================

pred = model.predict(test_generator)

y_pred = np.argmax(pred, axis=1)
y_true = test_generator.classes

class_names = list(test_generator.class_indices.keys())

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    output_dict=True
)

print(classification_report(y_true, y_pred, target_names=class_names))

pd.DataFrame(report).transpose().to_csv(REPORT_SAVE_PATH)


# ===============================
# CONFUSION MATRIX
# ===============================

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
plt.imshow(cm)
plt.title(EXPERIMENT_NAME)
plt.colorbar()
plt.xticks(np.arange(len(class_names)), class_names, rotation=45)
plt.yticks(np.arange(len(class_names)), class_names)

for i in range(len(class_names)):
    for j in range(len(class_names)):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()
plt.savefig(CONF_MATRIX_SAVE_PATH)
plt.show()


# ===============================
# ANA TABLOYA EKLE
# ===============================

new_row = pd.DataFrame([{
    "experiment_name": EXPERIMENT_NAME,
    "test_accuracy": test_acc,
    "test_loss": test_loss,
    "macro_f1": report["macro avg"]["f1-score"],
    "weighted_f1": report["weighted avg"]["f1-score"],
    "model_type": "Transfer Learning",
    "base_model": "MobileNetV2",
    "augmentation": "light",
    "optimizer": "RMSprop",
    "learning_rate": LEARNING_RATE,
    "activation": "ReLU",
    "output_activation": "Softmax",
    "epochs": len(history.history["accuracy"])
}])

if os.path.exists(SUMMARY_SAVE_PATH):
    old = pd.read_csv(SUMMARY_SAVE_PATH)
    all_df = pd.concat([old, new_row], ignore_index=True)
else:
    all_df = new_row

all_df.to_csv(SUMMARY_SAVE_PATH, index=False)

print("Sonuç tabloya eklendi:")
print(SUMMARY_SAVE_PATH)