import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the trained model
model_path = 'h5 models\\trymodel.h5'
model = load_model(model_path)

def extract_model_info(model):
    model_info = {
        'input_shape': model.input_shape,
        'output_shape': model.output_shape,
        'layers': [],
        'optimizer': {
            'class_name': model.optimizer.__class__.__name__,
            'config': model.optimizer.get_config()
        },
        'loss': model.loss if isinstance(model.loss, str) else model.loss.__name__,
        'metrics': model.metrics_names
    }
    for layer in model.layers:
        layer_info = {
            'name': layer.name,
            'class_name': layer.__class__.__name__,
            'config': layer.get_config(),
            'weights': [w.tolist() for w in layer.get_weights()]
        }
        model_info['layers'].append(layer_info)
    return model_info

# Extract model information
model_info = extract_model_info(model)

# Save model information to a JSON file
with open('jsonfiles\\model_info.json', 'w') as f:
    json.dump(model_info, f, indent=4)

print("Model information saved successfully.")
