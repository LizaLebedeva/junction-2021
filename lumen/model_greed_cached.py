from copy import deepcopy
import pandas as pd
from scipy.stats.stats import pearsonr


### Some ideas for combined correlation
def timeseries_from_pandas(df, device_id):
    sensor_events = df[df.deviceid==device_id]
    return sensor_events.timestamp.tolist()

def bucketed_time_series(df, device_id, resolution):
    sensor_events = df[df.deviceid==device_id]
    return sensor_events['timestamp'].dt.floor(resolution).tolist()
    

class EventCorrelator(object):
    def time_series_distance(self, ts1, ts2):
        pass
    
class DummySimultaneousFireCorrelator(EventCorrelator):
    increment_when_match = None
    
    def __init__(self, increment_when_match=1):
        self.increment_when_match=increment_when_match
        
    
    def time_series_distance(self, ts1, ts2):
        episodes = max(len(ts1), len(ts2))
        matches = 0
        left = 0
        right = 0
        while left < len(ts1) and right < len(ts2):
            if ts1[left] == ts2[right]:
                matches += self.increment_when_match
                left +=1 
                right +=1
            elif ts1[left] < ts2[right]:
                left +=1 
            elif ts1[left] > ts2[right]:
                right +=1
                
        return matches / episodes

class DummyCorrelator2(EventCorrelator):
    def time_series_distance(self, ts1, ts2):
        print("Dummy")

class CombinedCorrelator(object):
    def time_series_distance(self, df, device1, device2):
        one_sec_correlator = DummySimultaneousFireCorrelator(increment_when_match=5)
        five_sec_correlator = DummySimultaneousFireCorrelator(increment_when_match=2)
        ts_one_sec_1 = bucketed_time_series(df, device1, '1s')
        ts_one_sec_2 = bucketed_time_series(df, device2, '1s')
        one_sec_corr = one_sec_correlator.time_series_distance(ts_one_sec_1, ts_one_sec_2)
        ts_five_sec_1 = bucketed_time_series(df, device1, '5s')
        ts_five_sec_2 = bucketed_time_series(df, device2, '5s')
        five_sec_corr = five_sec_correlator.time_series_distance(ts_five_sec_1, ts_five_sec_2)
        
        ts_10_sec_1 = bucketed_time_series(df, device1, '10s')
        ts_10_sec_2 = bucketed_time_series(df, device2, '10s')
        ten_sec_corr = five_sec_correlator.time_series_distance(ts_10_sec_1, ts_10_sec_2)
        
        return (one_sec_corr, five_sec_corr, ten_sec_corr)


###


class ModelGreedCached:

    def __init__(self, time_resolution='10s', top_near_size=3):
        self.time_resolution = time_resolution
        self.top_near_size = top_near_size
        self.series = None
        self.cache = {}


    def compute_correlations(self, devices_list):
        self.cache = {}
        device_count = len(devices_list)
        for i in range(device_count):
            ref_device = devices_list[i]
            for j in range(i, device_count):
                second_device = devices_list[j]
                correlation = self._calculate_similarity(ref_device, second_device)
                dict_key = self.get_dict_key_for_cache_from_ids(ref_device, second_device)
                self.cache[dict_key] = correlation
               # print(f'Stored correlation {correlation} for {dict_key}')

    def get_dict_key_for_cache_from_ids(self, id1, id2):
        ordered = sorted([id1, id2])
        dict_key = tuple(i for i in ordered)
        return dict_key

    def get_correlation(self, device1, device2):
        return self.cache[self.get_dict_key_for_cache_from_ids(device1, device2)]        

    def fit(self, df_events):
        df = df_events.copy()
        df.timestamp = df.timestamp.dt.floor(self.time_resolution)
        min_time = df.timestamp.min()
        max_time = df.timestamp.max()

        self.series = pd.DataFrame()
        self.series['time'] = pd.date_range(start=min_time, end=max_time, freq=self.time_resolution)
        self.series = self.series.set_index('time')
        for device_id in sorted(df.deviceid.unique()):
            _events = list(df[df.deviceid == device_id].timestamp)
            self.series[device_id] = 0
            self.series[device_id].loc[_events] = 1

        self.compute_correlations(df_events.deviceid.unique())

    def get_unknown_devices(self, devices):
        return [device for device in devices if not device.has_device_id()]
    
    def get_know_devices(self, devices):
        return [device for device in devices if device.has_device_id()]

    def predict(self, input_devices, device_id_candidates):
        devices = deepcopy(input_devices)
        # make a copy
        device_id_candidates = set([x for x in device_id_candidates])
        unknown_devices = self.get_unknown_devices(devices)
        while len(unknown_devices) > 0:
            device_to_map = unknown_devices[0]
            ref_position = device_to_map.position
            # Find know near devices
            edges_space = []
            for device in self.get_know_devices(devices):
                dist = ((device.position[0] - ref_position[0])**2 + (device.position[1] - ref_position[1])**2)**0.5
                edges_space.append([device, dist])
            near_devices = sorted(edges_space, key=lambda x:x[1])[:self.top_near_size]        
    
            # Find best candidate
            candidates = []
            for candidate_device_id in sorted(device_id_candidates):
                score = 0
                for near_node in near_devices:
                    ref_device = near_node[0]
                    corr = self.get_correlation(ref_device.device_id, candidate_device_id)
                    score += corr
                candidates.append([candidate_device_id, score])
            
            best_id = sorted(candidates, key=lambda x:x[1], reverse=True)[0][0]
            # Map device
            device_to_map.set_device_id(best_id)
            # Remove mapped id from candidates
            device_id_candidates = device_id_candidates - {best_id}
            # Update list of unknown
            unknown_devices = self.get_unknown_devices(devices)
        return devices

    def _calculate_similarity(self, ref_decide_id, candidate_device_id):
        return pearsonr(self.series[ref_decide_id].values, self.series[candidate_device_id].values)[0]
