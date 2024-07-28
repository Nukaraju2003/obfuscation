import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import deserialize as layer_deserialize
from tensorflow.keras.optimizers import deserialize as optimizer_deserialize
from tensorflow.keras.losses import get as get_loss

# Ensure LSTM and any other custom layers are registered
custom_objects = {
    'LSTM': tf.keras.layers.LSTM,
    'Dense': tf.keras.layers.Dense,  # Add other layers as needed
    'Dropout': tf.keras.layers.Dropout
}

# Load model information from the JSON file
json_path = 'jsonfiles\\model_info.json'
with open(json_path, 'r') as f:
    model_info = json.load(f)

# Reconstruct the model
model = Sequential()

# Add layers
for layer_info in model_info['layers']:
    layer = layer_deserialize({
        'class_name': layer_info['class_name'],
        'config': layer_info['config']
    }, custom_objects=custom_objects)
    model.add(layer)

# Logging function to check weight shapes
def log_weights(layer, weights):
    print(f"Layer name: {layer.name}")
    print(f"Layer expects {len(layer.weights)} weights.")
    print(f"Provided weights: {len(weights)}")
    for i, (w, lw) in enumerate(zip(weights, layer.weights)):
        print(f"Weight {i} shape (provided): {np.array(w).shape}")
        print(f"Weight {i} shape (expected): {lw.shape}")

# Set weights
for layer, layer_info in zip(model.layers, model_info['layers']):
    weights = [np.array(w) for w in layer_info['weights']]
    log_weights(layer, weights)
    try:
        layer.set_weights(weights)
    except ValueError as e:
        print(f"Error setting weights for layer {layer.name}: {e}")

# Compile the model
optimizer = optimizer_deserialize({
    'class_name': model_info['optimizer']['class_name'],
    'config': model_info['optimizer']['config']
})

# Ensure loss is in the correct format
loss = model_info['loss']
if isinstance(loss, str):
    loss = get_loss(loss)

model.compile(optimizer=optimizer, loss=loss, metrics=model_info['metrics'])

# Save the reconstructed model as an .h5 file
model.save('reconstructed_model.h5')

print("Model reconstructed and saved successfully as 'reconstructed_model.h5'.")
