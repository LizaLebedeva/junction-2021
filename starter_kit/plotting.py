import numpy as np
from plotly import graph_objs as go


class Sensor:
    def __init__(self, x, y):
        self.device = dict(
            marker=dict(
                color='red',
                size=20,
                opacity=0.0
            ),
            x=x,
            y=y,
        )

    def update(self, b):
        b = np.clip(b, 0.25, 1.0)
        self.device['marker']['opacity'] = b


class Plotting:
    def __init__(self, bg_img, dims, df_devices, speed=None, scaling_factor=1):
        self.__scaling_factor = scaling_factor
        self.__IMG_WIDTH = dims[0]//scaling_factor
        self.__IMG_HEIGHT = dims[1]//scaling_factor
        self.__df_devices = self.__parse_devices(df_devices.copy())
        self.__speed = speed if speed else 500
        self.__init_figure(bg_img="data:image/png;base64," + bg_img)
        self.__parse_luminaires()
        self.__create_labels()

    def __parse_devices(self, df_devices):
        df_devices.loc[:, 'x'] = df_devices['x']//self.__scaling_factor
        df_devices.loc[:, 'y'] = df_devices['y']//self.__scaling_factor
        return df_devices

    def __init_figure(self, bg_img):
        # Initialise empty figure members
        data, layout, frames = [], dict(), []

        # Set bounds by visualizing an emtpy scatter plot
        data.append({
            'type': 'scatter',
            'x': [0, self.__IMG_WIDTH],
            'y': [0, self.__IMG_HEIGHT],
            'mode': 'markers',
        })
        layout['width'] = self.__IMG_WIDTH
        layout['height'] = self.__IMG_HEIGHT
        layout['xaxis'] = {'visible': False, 'showgrid': False}
        layout['yaxis'] = {'visible': False, 'showgrid': False}
        layout['images'] = [{
            'source': bg_img,
            'x': 0,
            'y': self.__IMG_HEIGHT,
            'xref': 'x',
            'yref': 'y',
            'sizex': self.__IMG_WIDTH,
            'sizey': self.__IMG_HEIGHT,
            'sizing': 'stretch',
            'layer': 'below',
            'opacity': 1.0,
        }]
        layout['updatemenus'] = [{
            'type': 'buttons',
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": self.__speed, "redraw": True},
                                    "fromcurrent": True, "transition": {"duration": 50,
                                                                        "easing": "linear"}
                                    }],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                      "mode": "immediate",
                                      "transition": {"duration": 0}}],
                    "label": "Pause",
                    "method": "animate"
                }
            ]
        }]
        self.__figure_dict = {
            'data': data,
            'layout': layout,
            'frames': frames,
        }

    def __parse_luminaires(self,):
        self.luminaires = dict()
        x, y = [], []
        for item in self.__df_devices.to_dict(orient='records'):
            self.luminaires[item['deviceid']] = Sensor(
                x=item['x'], y=item['y'])
            x.append(item['x'])
            y.append(np.abs(self.__IMG_HEIGHT - item['y']))
        self.__figure_dict['data'].append({
            "x": x,
            "y": y,
            "mode": "markers",
            "marker": {
                    "color": 'red',
                "size": 20,
                "opacity": 1.0
            }
        })

    def __create_labels(self,):
        self.__figure_dict['layout']['annotations'] = [{
            'text': k,
            'x': v.device['x'],
            'y': np.abs(self.__IMG_HEIGHT - v.device['y']),
        } for k, v in self.luminaires.items()]

    def populate_data(self, frames, ts):
        self.__figure_dict['data'] = {
            'type': 'scatter',
            'x': [0, self.__IMG_WIDTH],
            'y': [0, self.__IMG_HEIGHT],
            'mode': 'markers',
        }
        for idx, frame in enumerate(frames):
            x, y, color, opacity = [], [], [], []
            if len(frame) > 0:
                for k, v in frame.items():
                    self.luminaires[k].update(b=v)
                    x.append(self.luminaires[k].device['x'])
                    y.append(np.abs(self.__IMG_HEIGHT -
                             self.luminaires[k].device['y']))
                    color.append(self.luminaires[k].device['marker']['color'])
                    opacity.append(
                        self.luminaires[k].device['marker']['opacity'])
            self.__figure_dict['frames'].append({
                'data': {
                    "x": x,
                    "y": y,
                    "mode": "markers",
                    "marker": {
                        "color": color,
                        "size": 20,
                        "opacity": opacity
                    }
                },
                'layout': {'annotations': [{
                    'x': 5,
                    'y': 0,
                    'text': ts[idx]['index']}]
                }
            })

    def run(self, renderer):
        fig = go.Figure(self.__figure_dict)
        fig.show(renderer=renderer)
