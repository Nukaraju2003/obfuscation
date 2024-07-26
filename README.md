# Obfuscation

## Code extracts and obfuscates a TensorFlow Keras model's layer details, modifies the architecture by adding shortcuts and extra layers, and then reconstructs and saves the obfuscated model.

1 .  The code loads a trained Keras model, extracts detailed information about its layers, and saves this information to a JSON file. <br>
2 .  Obfuscate Layer Names 


### Steps followed here are
- Parameter encapsulation
- Neural structure obfuscation
- Shortcut injection
- Extra layer injection

## Reference paper
[ModelObfuscator: Obfuscating Model Information to Protect Deployed ML-Based Systems](https://arxiv.org/pdf/2306.06112)<br>
ISSTA ’23, July 17–21, 2023, Seattle, WA, USA Mingyi Zhou, Xiang Gao, Jing Wu, John Grundy, Xiao Chen, Chunyang Chen, and Li Li
