# import json
# import random
# import string
# from cryptography.fernet import Fernet

# # Generate a key for encryption
# key = Fernet.generate_key()
# cipher_suite = Fernet(key)

# def random_string(length=10):
#     return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

# def obfuscate_layer_names(layers):
#     for layer in layers:
#         layer['name'] = random_string()
#     return layers

# def encrypt_config(config):
#     config_str = json.dumps(config)
#     encrypted_config = cipher_suite.encrypt(config_str.encode()).decode()
#     return encrypted_config

# def obfuscate_model_architecture(model_architecture):
#     model_architecture['config']['layers'] = obfuscate_layer_names(model_architecture['config']['layers'])
#     model_architecture['config'] = encrypt_config(model_architecture['config'])
#     model_architecture['compile_config'] = encrypt_config(model_architecture['compile_config'])
#     return model_architecture

# # Load model architecture from JSON file
# with open('..\\jsonfiles\\temp\\model_architecture.json', 'r') as f:
#     model_architecture = json.load(f)

# # Obfuscate model architecture
# obfuscated_model_architecture = obfuscate_model_architecture(model_architecture)

# # Save obfuscated model architecture to a new JSON file
# with open('obfuscated_model_architecture.json', 'w') as f:
#     json.dump(obfuscated_model_architecture, f, indent=4)

# # Save the encryption key for later decryption
# with open('encryption_key.txt', 'wb') as f:
#     f.write(key)

# print("Model architecture obfuscated and saved.")


import json

# Load the original model architecture JSON
with open('jsonfiles\\model_architecture.json', 'r') as file:
    original_json = json.load(file)

def obfuscate_key(key):
    # Implement a function to obfuscate keys; this can be more sophisticated
    key_map = {
        "class_name": "cls_name",
        "config": "cfg",
        "name": "nm",
        "trainable": "trbl",
        "dtype": "dt",
        "module": "mod",
        "registered_name": "reg_nm",
        "layers": "lyrs",
        "batch_shape": "bsh",
        "units": "u",
        "activation": "act",
        "recurrent_activation": "rec_act",
        "kernel_initializer": "kern_init",
        "bias_initializer": "bias_init",
        "dropout": "drop",
        "rate": "r",
        "seed": "sd",
        "input_shape": "in_shp",
        "learning_rate": "l_rate",
        "loss": "ls",
        "metrics": "mtcs"
    }
    return key_map.get(key, key)

def obfuscate_value(value):
    # Implement a function to obfuscate values; this can be more sophisticated
    return value

def obfuscate_dict(d):
    obfuscated_dict = {}
    for key, value in d.items():
        new_key = obfuscate_key(key)
        if isinstance(value, dict):
            obfuscated_dict[new_key] = obfuscate_dict(value)
        elif isinstance(value, list):
            obfuscated_dict[new_key] = [obfuscate_dict(item) if isinstance(item, dict) else obfuscate_value(item) for item in value]
        else:
            obfuscated_dict[new_key] = obfuscate_value(value)
    return obfuscated_dict

# Obfuscate the model architecture JSON
obfuscated_json = obfuscate_dict(original_json)

# Save the obfuscated model architecture JSON to a new file
with open('model_architecture_obfuscated.json', 'w') as file:
    json.dump(obfuscated_json, file, indent=2)

print("Obfuscation complete. The obfuscated JSON is saved as 'model_architecture_obfuscated.json'.")

