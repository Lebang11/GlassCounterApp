import os
import json

def repair_json(model_path):
    print(f"Repairing {model_path}...")
    with open(model_path, 'r') as f:
        data = json.load(f)
    
    # 1. Ensure top-level TFJS structure
    # If the file was just a Keras JSON, move everything under 'modelTopology'
    if 'model_config' in data:
        print("Restructuring Keras to TFJS format...")
        new_data = {
            'format': 'layers-model',
            'generatedBy': 'keras v3.13.2',
            'convertedBy': 'Manual Repair Script',
            'modelTopology': data['model_config'],
            'weightsManifest': data['weightsManifest']
        }
        data = new_data

    topology = data['modelTopology']
    layers = topology['config']['layers']
    
    # 2. Fix InputLayer (batch_input_shape is required by TFJS)
    if layers[0]['class_name'] == 'InputLayer':
        print("Fixing InputLayer...")
        layers[0]['config']['batch_input_shape'] = [None, 1024, 1]
    
    # 3. Use fully qualified names (layer_name/weight_name) to avoid collisions
    print("Standardizing layer and weight name mapping...")
    
    # Keras 3 often uses these paths: sequential_4/batch_normalization_2/gamma
    # We want: batch_normalization_2/gamma
    
    manifest = data['weightsManifest'][0]
    for w in manifest['weights']:
        parts = w['name'].split('/')
        if len(parts) > 1:
            # Keep the last two parts: e.g. batch_normalization_2/gamma
            w['name'] = "/".join(parts[-2:])
        else:
            # Fallback if it's already flat
            print(f"Warning: Weight {w['name']} is already flat.")

    for layer in layers:
        old_name = layer['config']['name']
        layer['config']['name'] = old_name.split('/')[-1]
        
    with open(model_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print("SUCCESS: model.json is now compatible with TFJS Layers API.")

if __name__ == "__main__":
    repair_json('assets/web_model/model.json')
