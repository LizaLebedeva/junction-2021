import os
from PIL import Image
import pandas as pd
import pickle


def load_image(data_path, site_name):
    image_file = os.path.join(data_path, site_name, f"{site_name}.png")
    return Image.open(image_file)
    
def load_device_position(data_path, site_name):
    device_file = os.path.join(data_path, site_name, f"{site_name}.json")
    df_positions = pd.read_json(device_file)
    return df_positions

def load_model(model_path, model_name):
    model_file = os.path.join(model_path, model_name)
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
        return model
