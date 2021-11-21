import io

from flask import Flask
from flask import send_file
from flask import render_template
from flask import jsonify

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from core import configuration
from core.io import load_image
from core.io import load_device_position
from core.io import load_model

from lumen.device import Device

memory_database = {
    'current_site': 'site_1',
    'devices': {},
    'model': None
}

app = Flask(__name__)
tile_data = np.array(load_image(configuration.DATA_PATH, memory_database['current_site']), dtype="uint8")

_gray = np.zeros(shape=(configuration.TILE_SIZE, configuration.TILE_SIZE, 3), dtype="uint8")
_gray[:, :, 0] = 255
_gray[:, :, 1] = 255
_gray[:, :, 2] = 255


model_cache = {}
for site_name in configuration.SITE_MODEL_MAP.keys():
    print(f"... loading model for {site_name}")
    model_cache[site_name] = load_model(configuration.MODEL_PATH, configuration.SITE_MODEL_MAP.get(site_name))

# Pre load first model for current_site
memory_database['model'] = model_cache[memory_database['current_site']]


@app.route("/load_memory_database/<site_name>")
def load_memory_database(site_name):
    global tile_data
    assert site_name in ('site_1', 'site_2', 'site_3', 'site_4', 'site_5')
    # Update current site name
    memory_database['current_site'] = site_name
    # Clear previous devices
    memory_database['devices'] = {}
    # Load new relevant model
    memory_database['model'] = model_cache[site_name]
    tile_data = np.array(load_image(configuration.DATA_PATH, site_name), dtype="uint8")
    return jsonify({})


@app.route("/read_memory_database")
def read_memory_database():    
    return jsonify({'memory_database': memory_database})


@app.route("/checked_id/<real_device_id>/<device_id>")
def checked_id(real_device_id, device_id):    
    real_device_id = int(real_device_id)
    device_id = int(device_id)
    memory_database['devices'][real_device_id] = device_id
    return jsonify({})


@app.route("/load_model_database")
def load_model_database():
    site_name = memory_database['current_site']
    
    return jsonify({})


@app.route("/model_predict")
def model_predict():
    model = memory_database['model']
    print(">>>> model: ", model)
    
    devices = []
    df_positions = load_device_position(configuration.DATA_PATH, memory_database['current_site'])
    device_id_list = sorted(df_positions.deviceid.unique())
    known_devices = set(memory_database['devices'].values())

    for device_id in device_id_list:
        point = df_positions.loc[device_id]
        position = [point.x, point.y]
        _device = Device(position=position, device_id=None)    
        if device_id in known_devices:
            # Add as known device
            _device.set_device_id(device_id)
        devices.append(_device)

    for device in devices:
        print(device)

    # model.predict(devices, test_devices_list)    
    return jsonify({})


def _build_device_list(known_devices):
    devices = []
    df_positions = load_device_position(configuration.DATA_PATH, memory_database['current_site'])
    device_id_list = sorted(df_positions.deviceid.unique())
    for device_id in device_id_list:
        point = df_positions.loc[device_id]
        position = [point.x, point.y]
        _device = Device(position=position, device_id=None)    
        if device_id in known_devices:
            # Add as known device
            _device.set_device_id(device_id)
        devices.append(_device)
    return devices

def _build_fallback_recommendations(real_device_id):
    mapped_devices = []
    if memory_database['devices'].get(real_device_id, None) is None:
        df_positions = load_device_position(configuration.DATA_PATH, memory_database['current_site'])
        all_devices = set(df_positions.deviceid.unique())
        known_devices = set(memory_database['devices'].values())
        candidates = sorted(all_devices - known_devices)
        mapped_devices = [int(x) for x in candidates[:6]] + [real_device_id]
    return mapped_devices


@app.route("/recommendations/<real_device_id>")
def recommendations(real_device_id):
    real_device_id = int(real_device_id)
    known_devices = set(memory_database['devices'].values())
    if len(known_devices) == 0:
        print("..... getting fallback recommendations ...")
        mapped_devices = _build_fallback_recommendations(real_device_id)
    else:
        print("..... getting model recommendations ...")
        df_positions = load_device_position(configuration.DATA_PATH, memory_database['current_site'])
        all_devices = set(df_positions.deviceid.unique())
        candidates = sorted(all_devices - known_devices)
        devices = _build_device_list(known_devices)
        model = memory_database['model']
        model_devices = model.predict(devices, candidates)
        mapped_devices = [int(x.device_id) for x in model_devices[:6]]
        if real_device_id not in mapped_devices:
            mapped_devices = mapped_devices[:5] + [real_device_id]

    return jsonify({'recommender': mapped_devices})



@app.route("/tiles/<z>/<x>/<y>")
def tiles(z, x, y):
    # Scale transformation
    max_size_y = tile_data.shape[0]
    max_size_x = tile_data.shape[1]
    image_ratio = max_size_x/max_size_y    
    z = int(z)
    x = int(x)
    y = int(y)
    scale = 2 ** (z+1)
    x_start = int(max_size_x*(x+0)/(scale))
    x_end = int(max_size_x*(x+1)/(scale))
    y_start = int(max_size_y*image_ratio*(y+0)/(scale))
    y_end = int(max_size_y*image_ratio*(y+1)/(scale))
    
    # Check limits
    if (x_start >= 0) and (x_start <= max_size_x) and (y_start >= 0) and (y_start <= max_size_y) and \
       (x_end  >= 0) and (x_end <= max_size_x) and (y_end >= 0) and (y_end <= max_size_y):
        # Crop valid region
        _t = tile_data[y_start:y_end, x_start:x_end, :]
        print(".... tile shape: ", _t.shape)
        tile = Image.fromarray(_t.astype('uint8')).resize((configuration.TILE_SIZE, configuration.TILE_SIZE))
    else:
        # Invalid region
        tile = Image.fromarray(_gray.astype('uint8')).resize((configuration.TILE_SIZE, configuration.TILE_SIZE))

    # Prepare tile response as png format
    file_object = io.BytesIO()
    tile.save(file_object, 'PNG')
    file_object.seek(0)
    return send_file(file_object, mimetype='image/PNG')


@app.route("/devices/<zoom>")
def devices(zoom):
    zoom = int(zoom)
    max_size_y = tile_data.shape[0]
    max_size_x = tile_data.shape[1]
    image_ratio = max_size_x/max_size_y
    scale = 2 ** (zoom+1)
    x_scale = configuration.TILE_SIZE / int(max_size_x/(scale))
    y_scale = configuration.TILE_SIZE / int(max_size_y*image_ratio/(scale))
    df_positions = load_device_position(configuration.DATA_PATH, memory_database['current_site'])
    device_list = []
    for i, row in df_positions.iterrows():
        # Recover ids
        real_device_id = int(row.deviceid)
        device_id = None
        color = '#FF0000'
        if real_device_id in memory_database['devices']:
            device_id = memory_database['devices'].get(real_device_id)
            color = '#00FF00'
        device_list.append({
            'device_id': device_id,
            'real_device_id': real_device_id,
            'x': int(row.x*x_scale),
            'y': int(row.y*y_scale),
            'color': color
        })
    return jsonify({'devices': device_list})


@app.route('/')
def index():
    context = {
        "version": "v1.0"
    }
    return render_template('index.html', context=context)


if __name__ == '__main__':
    print("------------------------------------------")
    print("- Lumen Loop 3.0 ")
    print("------------------------------------------")
    app.run(host='127.0.0.1', port=8080, debug=True)
    # app.run(host='0.0.0.0', port=8080, debug=False)
