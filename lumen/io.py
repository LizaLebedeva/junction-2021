import pandas as pd


def read_events(site_name):
    df_events = pd.read_pickle(f'./starter_kit/data/{site_name}/{site_name}.pkl', compression='gzip')
    df_events.loc[:, 'timestamp'] = (pd.to_datetime(df_events['timestamp'], utc=True).dt.tz_convert('Europe/Helsinki').dt.tz_localize(None))
    return df_events


def read_positions(site_name):
    df_devices = pd.read_json(f'./starter_kit/data/{site_name}/{site_name}.json')
    return df_devices


# def read_floor_plan(site_name):
#     with open(f'./starter_kit/data/{site_name}/{site_name}.png', "rb") as image_file:
#         encoded_string = base64.b64encode(image_file.read()).decode()
#         img = iio.imread(f'./data/{site}/{site}.png')
#         return img
