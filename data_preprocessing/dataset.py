"""
Unified TSDataSet class for all HAR datasets.
"""

class TSDataSet:
    """
    Time-Series Dataset container for Human Activity Recognition.
    
    Attributes:
        data: numpy array of sensor readings, shape (seq_len, n_channels)
        label: integer activity label
        length: sequence length (number of time steps)
        user_id: optional user identifier (default None for datasets without user info)
    """
    def __init__(self, data, label, length, user_id=None):
        self.data = data
        self.label = int(label)
        self.length = int(length)
        self.user_id = int(user_id) if user_id is not None else None
