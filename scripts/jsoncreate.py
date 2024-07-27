import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the trained model
model_path = 'h5 models\\trymodel.h5'
model = load_model(model_path)

import json
import numpy as np

def extract_model_info(model):
    model_info = {
        'layers': []
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

# Assuming `model` is already defined and loaded
model_info = extract_model_info(model)

# Save model information to a JSON file
with open('jsonfiles\\model_info.json', 'w') as f:
    json.dump(model_info, f, indent=4)

