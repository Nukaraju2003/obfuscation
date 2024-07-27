import json
import base64
import numpy as np
from tensorflow.keras.models import model_from_json, load_model

# Load the original model
model = load_model('h5 models\\trymodel.h5')

# Get model configuration and weights
model_config = model.to_json()
model_weights = model.get_weights()

# Verify original model configuration
original_model_config_dict = json.loads(model_config)
# print(json.dumps(original_model_config_dict, indent=4))

# Modify model configuration to obfuscate layer names
model_config_dict = json.loads(model_config)
for layer in model_config_dict['config']['layers']:
    layer['config']['name'] = layer['config']['name'] + '_obf'

# Verify obfuscated model configuration
# print(json.dumps(model_config_dict, indent=4))

# Save the obfuscated configuration
obfuscated_model_config = json.dumps(model_config_dict)
with open('jsonfiles\\obfuscated_model_config.json', 'w') as f:
    f.write(obfuscated_model_config)

def encode_weights(weights):
    encoded_weights = []
    for w in weights:
        encoded_w = base64.b64encode(w.tobytes())
        encoded_weights.append(encoded_w)
    return encoded_weights

def decode_weights(encoded_weights, original_shapes):
    decoded_weights = []
    for encoded_w, shape in zip(encoded_weights, original_shapes):
        decoded_w = np.frombuffer(base64.b64decode(encoded_w), dtype=np.float32)
        decoded_w = decoded_w.reshape(shape)
        decoded_weights.append(decoded_w)
    return decoded_weights

# Encode and decode weights for verification
encoded_weights = encode_weights(model_weights)
original_shapes = [w.shape for w in model_weights]
decoded_weights = decode_weights(encoded_weights, original_shapes)

# Verify weight encoding and decoding
for original, decoded in zip(model_weights, decoded_weights):
    if not np.allclose(original, decoded):
        print("Mismatch found in weights.")
        break
else:
    print("Weights match after encoding and decoding.")

# Recreate the model from the obfuscated configuration
obfuscated_model = model_from_json(obfuscated_model_config)

# Decode weights and set them to the model
obfuscated_model.set_weights(decoded_weights)

# Verify model predictions
some_test_data = np.random.rand(10, *model.input_shape[1:])  # Replace with actual test data
original_predictions = model.predict(some_test_data)
obfuscated_predictions = obfuscated_model.predict(some_test_data)

if np.allclose(original_predictions, obfuscated_predictions):
    print("Model reconstruction is successful.")
else:
    print("Model reconstruction has issues.")

# Save the obfuscated model
obfuscated_model.save('newobfuscated_model.h5')
