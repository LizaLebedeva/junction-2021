import numpy as np
from PIL import Image, ImageDraw, ImageFont


def extract_device_events(df_events):
    """ Read events from dataframe
        return dict of device_id -> list(events)
    """
    events_map = {}
    devices = sorted(df_events.deviceid.unique())
    for device_id in devices:
        events = list(df_events[df_events.deviceid == device_id].timestamp)
        events_map[device_id] = events
    
    return events_map


def plot_devices(devices, device_id_list=None, to_file=None, text_size=12, radius=40, train_device_id_list=None, real_device_id_to_position_map=None):
    
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", text_size)
    x_min = int(np.floor(min([d.position[0] for d in devices]))) - radius
    x_max = int(np.floor(max([d.position[0] for d in devices]))) + radius
    y_min = int(np.floor(min([d.position[1] for d in devices]))) - radius
    y_max = int(np.floor(max([d.position[1] for d in devices]))) + radius
    x_size = x_max - x_min
    y_size = y_max - y_min

    device_id_list = device_id_list if device_id_list is not None else []
    train_device_id_list = train_device_id_list if train_device_id_list is not None else []
    real_device_id_to_position_map = real_device_id_to_position_map if real_device_id_to_position_map is not None else {}

    image = Image.new('RGBA', (x_size, y_size), (255,255,255,255))
    draw = ImageDraw.Draw(image)

    for device in devices:
        device_id = device.device_id
        offset_pos = [int(device.position[0]-x_min), int(device.position[1]-y_min)]

        # Select colors
        bg_color = "white"
        if device_id in train_device_id_list:
            bg_color = "#50ff50"

        shape = [offset_pos[0], offset_pos[1], offset_pos[0]+radius, offset_pos[1]+radius]
        draw.ellipse(shape, fill=bg_color, outline='black')
        draw.text([offset_pos[0]-text_size, offset_pos[1]-text_size], f"{device_id}", font=font, fill="black")

    # Write real ids
    for device_id in device_id_list:
        position = real_device_id_to_position_map.get(device_id)
        offset_pos = [int(position[0]-x_min), int(position[1]-y_min)]
        draw.text([offset_pos[0]+text_size, offset_pos[1]+text_size], f"{device_id}", font=font, fill="red")
        
    # Save to file image
    if to_file:
        image.save(to_file)

    return image
