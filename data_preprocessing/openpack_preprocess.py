from glob import glob
import numpy as np
import pandas as pd
from collections import Counter
import os
import re
import logging
from .dataset import TSDataSet

def openpackLoader(file_name_pattern, timespan, min_seq):
    """
    Load and preprocess OpenPack dataset from CSV files.
    
    The OpenPack dataset contains wearable IMU sensor data from 4 activity trackers (atr).
    Each file is named U{user_id}-S{session_id}.csv with 41 sensor channels 
    (acc_xyz, gyro_xyz, quat_xyzw for each atr) and activity labels (0-10).
    
    Args:
        file_name_pattern (str): Glob pattern to match OpenPack CSV files (e.g., "data/openpack/U*-S*.csv")
        timespan (int): Sampling interval in milliseconds to reduce temporal resolution
        min_seq (int): Minimum sequence length to keep (shorter sequences are discarded)
    
    Returns:
        list: List of TSDataSet objects, each containing:
            - data: numpy array of shape (seq_len, 41) with sensor readings
            - label: activity label (0-9, label 10 is filtered out)
            - length: sequence length
            - user_id: user identifier extracted from filename
    """
    logging.info("Loading OpenPack Dataset --------------------------------------")

    file_list = sorted(glob(file_name_pattern))
    dataset_list = []
    total_raw_pointers = 0  # Before timespan filtering
    total_data_pointers = 0  # After timespan filtering
    label_list = []
    id_list = []

    for file_path in file_list:
        print(file_path)

        # change according to the file path
        # Extract the file name from the full file path
        basename = os.path.basename(file_path)  # Example: "U0101-S0100.csv"
        # Use regular expression to extract numeric part from "U0101"
        # Pattern explanation:
        # - U0*   : Matches 'U' followed by zero or more '0's
        # - (\d+): Captures one or more digits after leading zeros
        match = re.search(r'U0*(\d+)', basename)
        # If match found, use the captured group (e.g., '101'); otherwise default to '0'
        user_id_str = match.group(1) if match else '0'

        # Keep track of unique user IDs
        if user_id_str not in id_list:
            id_list.append(user_id_str)


        df = pd.read_csv(file_path, sep=',', header=0).to_numpy(dtype=np.float64)

        if len(df) > 0:
            total_raw_pointers += len(df)

            current_label = df[0, 1]
            current_time = df[0, 0]
            temp_dataset = [df[0, 2:43]]

            for i in range(1, len(df)):
                row_time = df[i, 0]
                row_label = df[i, 1]
                row_sensor = df[i, 2:43]

                if (row_time - current_time) >= timespan:
                    current_time = row_time
                    if current_label == row_label:
                        temp_dataset.append(row_sensor)
                    else:
                        if current_label != 10 and len(temp_dataset) >= min_seq:
                            seq_array = np.array(temp_dataset)
                            dataset_list.append(TSDataSet(seq_array, current_label, len(seq_array), int(user_id_str)))
                            total_data_pointers += len(seq_array)
                        label_list.append(current_label)
                        temp_dataset = [row_sensor]
                        current_label = row_label

            if current_label != 10 and len(temp_dataset) >= min_seq:
                seq_array = np.array(temp_dataset)
                dataset_list.append(TSDataSet(seq_array, current_label, len(seq_array), int(user_id_str)))
                total_data_pointers += len(seq_array)
            label_list.append(current_label)

    # Summary
    sensor_channels = 41  # Columns 2 to 42
    activity_counts = Counter(seq.label for seq in dataset_list)
    num_activity_types = len(activity_counts)
    total_sequences = len(dataset_list)

    logging.info(f"User id: {id_list}")
    logging.info("Loading OpenPack Dataset Finished --------------------------------------")
    logging.info("====== Dataset Summary ======")
    logging.info(f"Sensor channels: {sensor_channels}")
    logging.info(f"Raw data points (before timespan filtering): {total_raw_pointers}")
    logging.info(f"Total data points (after timespan filtering): {total_data_pointers}")
    logging.info(f"Total activities (sequences): {total_sequences}")
    logging.info(f"Number of activity types: {num_activity_types}")
    logging.info("Activity sequence counts and data points:")
    for label in sorted(activity_counts.keys()):
        count = activity_counts[label]
        total_points = sum(seq.length for seq in dataset_list if seq.label == label)
        logging.info(f"  Activity {int(label)}: {count} sequences, {total_points} data points")

    return dataset_list

