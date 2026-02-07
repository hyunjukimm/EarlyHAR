from glob import glob
import numpy as np
import pandas as pd

class TSDataSet:
    def __init__(self, data, label, length):
        self.data = data
        self.label = int(label)
        self.length = int(length)

# ARAS Loader using time-based segmentation (not value change or downsampling)
def arasLoader(file_name_pattern, timespan, min_seq):
    print("Loading ARAS Dataset --------------------------------------")
    
    file_list = sorted(glob(file_name_pattern))
    dataset_list = []
    label_list = []

    for file_path in file_list:
        df = pd.read_csv(file_path, sep=' ', header=None).to_numpy()
        print(file_path)

        if len(df) == 0:
            continue

        # Initialize with the first row
        current_label = [df[0, 20], df[0, 21]]     # Activity labels for resident 1 and 2
        current_time = 0                           # Reference time index
        current_data = df[0, 0:20]                 # Initial sensor data
        temp_dataset = np.array([current_data])    # Accumulate sensor readings
        last_active_resident = 0                   # Track which resident's activity (0 or 1 index)

        for i in range(1, len(df)):
            # Accumulate data every `timespan` seconds
            if (i - current_time) >= timespan:
                current_time = i
                sensor_data = df[i, 0:20]
                label_r1, label_r2 = df[i, 20], df[i, 21]

                # If activity labels are unchanged
                if (current_label[0] == label_r1 and current_label[1] == label_r2):
                    # Add if sensor data has changed
                    if (current_data != sensor_data).any():
                        temp_dataset = np.concatenate((temp_dataset, [sensor_data]), axis=0)
                        current_data = sensor_data
                else:
                    # Save the segment if long enough
                    if len(temp_dataset) >= min_seq:
                        # Use the label of whichever resident changed their activity
                        label_to_use = current_label[0] if current_label[0] != label_r1 else current_label[1]
                        dataset_list.append(TSDataSet(temp_dataset, label_to_use, len(temp_dataset)))
                        label_list.append(current_label)

                    # Track which resident changed for the new segment
                    last_active_resident = 0 if current_label[0] != label_r1 else 1

                    # Start a new segment
                    temp_dataset = np.array([sensor_data])
                    current_label = [label_r1, label_r2]
                    current_data = sensor_data

        # Save the last segment
        # Use the label of the resident who last changed activity
        if len(temp_dataset) >= min_seq:
            label_to_use = current_label[last_active_resident]
            dataset_list.append(TSDataSet(temp_dataset, label_to_use, len(temp_dataset)))
            label_list.append(current_label)

    print("Loading ARAS Dataset Finished --------------------------------------")
    print(f"Total sequences loaded: {len(dataset_list)}")
    return dataset_list