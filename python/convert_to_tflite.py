import tensorflow as tf

model_path = "models/mobilenetv2_transfer_learning.keras"
tflite_path = "models/mobilenetv2_transfer_learning.tflite"

model = tf.keras.models.load_model(model_path)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open(tflite_path, "wb") as f:
    f.write(tflite_model)

print("TFLite model oluşturuldu:", tflite_path)