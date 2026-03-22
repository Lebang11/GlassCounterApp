import os
import json
import h5py
import sys
import numpy as np

# Mocking modules that fail but aren't needed for H5 conversion
sys.modules['tensorflow_decision_forests'] = type('Mock', (), {})
sys.modules['tensorflow_hub'] = type('Mock', (), {})

import tensorflowjs
from tensorflowjs import write_weights
from tensorflowjs.converters import keras_h5_conversion

# Mocking jax too
sys.modules['jax'] = type('Mock', (), {})
sys.modules['jaxlib'] = type('Mock', (), {})

def manual_convert(h5_path, output_dir):
    print(f"Loading {h5_path}...")
    f = h5py.File(h5_path, 'r')
    
    print("Converting to TFJS format...")
    # This generates the model topology and organizes weights into groups
    model_json, groups = keras_h5_conversion.h5_merged_saved_model_to_tfjs_format(f)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Writing weights to {output_dir}...")
    # write_weights returns the manifest part needed for model.json
    manifest = write_weights.write_weights(groups, output_dir, write_manifest=False)
    
    model_json['weightsManifest'] = manifest
    
    model_path = os.path.join(output_dir, 'model.json')
    print(f"Writing metadata to {model_path}...")
    with open(model_path, 'w') as j:
        json.dump(model_json, j)
    
    print("SUCCESS: Model converted for mobile app.")

if __name__ == "__main__":
    manual_convert('glass_counter_hybrid_finetuned.h5', 'assets/web_model')
