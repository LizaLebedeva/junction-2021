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


@app.route("/load_memory_database/<site_name>")
def load_memory_database(site_name):
    global tile_data
    assert site_name in ('site_1', 'site_2', 'site_3', 'site_4', 'site_5')
    memory_database['current_site'] = site_name
    memory_database['devices'] = {}
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


@app.route("/recommendations/<real_device_id>")
def recommendations(real_device_id):
    real_device_id = int(real_device_id)
    mapped_devices = []
    if memory_database['devices'].get(real_device_id, None) is None:
        df_positions = load_device_position(configuration.DATA_PATH, memory_database['current_site'])
        all_devices = set(df_positions.deviceid.unique())
        known_devices = set(memory_database['devices'].values())
        candidates = sorted(all_devices - known_devices)
        mapped_devices = [int(x) for x in candidates[:5]]
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
