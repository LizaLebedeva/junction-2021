
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

