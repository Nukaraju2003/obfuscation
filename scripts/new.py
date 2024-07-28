# import json
# from tensorflow.keras.models import model_from_json
# from tensorflow.keras.models import load_model

# # Load the original model
# model = load_model('h5 models\\trymodel.h5')

# # Get model configuration and weights
# model_config = model.to_json()
# model_weights = model.get_weights()

# # Modify model configuration to obfuscate layer names
# model_config_dict = json.loads(model_config)
# for layer in model_config_dict['config']['layers']:
#     layer['config']['name'] = layer['config']['name'] + '_obf'

# # Save the obfuscated configuration
# obfuscated_model_config = json.dumps(model_config_dict)
# with open('jsonfiles\\obfuscated_model_config.json', 'w') as f:
#     f.write(obfuscated_model_config)

# # Recreate the model from the obfuscated configuration
# obfuscated_model = model_from_json(obfuscated_model_config)

# # Set weights to the model
# obfuscated_model.set_weights(model_weights)

# # Save the obfuscated model
# obfuscated_model.save('h5 models\\newobfuscated_model.h5')


import json
import numpy as np
from tensorflow.keras.models import model_from_json, load_model

# Load the original model
model = load_model('h5 models//trymodel.h5')

# Get model configuration and weights
model_config = model.to_json()
model_weights = model.get_weights()

# Modify model configuration to obfuscate layer names
model_config_dict = json.loads(model_config)
for layer in model_config_dict['config']['layers']:
    layer['config']['name'] = layer['config']['name'] + '_obf'

# Save the obfuscated configuration
obfuscated_model_config = json.dumps(model_config_dict)
with open('jsonfiles//obfuscated_model_config.json', 'w') as f:
    f.write(obfuscated_model_config)

# Recreate the model from the obfuscated configuration
obfuscated_model = model_from_json(obfuscated_model_config)

# Set the original weights to the model
obfuscated_model.set_weights(model_weights)

# Save the obfuscated model
obfuscated_model.save('h5 models//newobfuscated_model.h5')

# Verify the obfuscated model
test_data = np.random.random((10, *model.input_shape[1:]))
original_predictions = model.predict(test_data)
obfuscated_predictions = obfuscated_model.predict(test_data)

# Compare predictions
assert np.allclose(original_predictions, obfuscated_predictions), "Model predictions differ after obfuscation."
