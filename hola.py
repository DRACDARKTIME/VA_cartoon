import tensorflow as tf

print("GPUs available:", len(tf.config.list_physical_devices("GPU")))
print(tf.config.list_physical_devices("GPU"))
