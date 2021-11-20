
class Device:
    
    def __init__(self, position, device_id):
        self.device_id = device_id
        self.position =position
        self.events = None
    
    def has_position(self):
        return self.position is not None
    
    def has_device_id(self):
        return self.device_id is not None
    
    def set_position(self, position):
        self.position = position
        
    def set_device_id(self, device_id):
        self.device_id = device_id
    
    def set_events(self, events):
        self.events = events

    def __str__(self):
        return f"Device ID:{self.device_id} at {self.position}"
    
    def __repr__(self):
        return self.__str__()
