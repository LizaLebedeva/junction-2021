import os
from PIL import Image
import pandas as pd


def load_image(data_path, site_name):
    image_file = os.path.join(data_path, site_name, f"{site_name}.png")
    return Image.open(image_file)
    
def load_device_position(data_path, site_name):
    device_file = os.path.join(data_path, site_name, f"{site_name}.json")
    df_positions = pd.read_json(device_file)
    return df_positions
