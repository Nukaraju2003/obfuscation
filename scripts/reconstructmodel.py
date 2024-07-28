import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Layer

# Load the model information from the JSON file
json_file_path = 'jsonfiles\\model_info.json'
with open(json_file_path, 'r') as f:
    model_info = json.load(f)

# Function to recreate a layer from the config
def recreate_layer(layer_info):
    class_name = layer_info['class_name']
    config = layer_info['config']
    layer_class = getattr(tf.keras.layers, class_name)
    layer = layer_class.from_config(config)
    return layer

# Reconstruct the model
reconstructed_model = Sequential()
for layer_info in model_info['layers']:
    layer = recreate_layer(layer_info)
    reconstructed_model.add(layer)

# Debugging: Print layer configurations
for layer in reconstructed_model.layers:
    print(f"Layer {layer.name}: {layer.get_config()}")

# Set the weights for each layer
for i, layer_info in enumerate(model_info['layers']):
    weights = [np.array(w) for w in layer_info['weights']]
    print(f"Setting weights for layer {reconstructed_model.layers[i].name}")
    try:
        reconstructed_model.layers[i].set_weights(weights)
    except ValueError as e:
        print(f"Error setting weights for layer {reconstructed_model.layers[i].name}: {e}")

# Compile the reconstructed model if needed
reconstructed_model.compile(optimizer='adam', loss='mean_squared_error')

# Save the reconstructed model to verify it works correctly
reconstructed_model.save('h5 models\\reconstructed_model.h5')
