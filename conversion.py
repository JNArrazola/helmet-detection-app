import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model("best_v1")  # carpeta generada por onnx2tf
tflite_model = converter.convert()
with open("best_v1.tflite", "wb") as f:
    f.write(tflite_model)
