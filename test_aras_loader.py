import sys
sys.path.insert(0, '/home/juice/EarlyHAR/EarlyHAR')

from data_preprocessing.aras_preprocess import arasLoader
import numpy as np

# Load ARAS dataset
file_pattern = '/home/juice/EarlyHAR/EarlyHAR/data/aras/HouseA/DAY_*.txt'
timespan = 5
min_seq = 10

print(f"Parameters: timespan={timespan}, min_seq={min_seq}\n")

dataset_list = arasLoader(file_pattern, timespan, min_seq)

print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)
print(f"Total sequences: {len(dataset_list)}")

# Analyze sequence lengths
lengths = [ds.length for ds in dataset_list]
print(f"\nSequence lengths:")
print(f"  Min: {min(lengths)}")
print(f"  Max: {max(lengths)}")
print(f"  Mean: {np.mean(lengths):.1f}")
print(f"  Median: {np.median(lengths):.1f}")

# Analyze labels
labels = [ds.label for ds in dataset_list]
unique_labels = np.unique(labels)
print(f"\nUnique activity labels: {len(unique_labels)}")
print(f"  Labels: {sorted(unique_labels)}")

# Count top 10 most frequent activities
from collections import Counter
label_counts = Counter(labels)
print(f"\nTop 10 most frequent activities:")
for label, count in label_counts.most_common(10):
    percentage = (count / len(dataset_list)) * 100
    print(f"  Label {label:2d}: {count:4d} sequences ({percentage:5.1f}%)")

# Show first 3 sequences as examples
print(f"\n" + "="*70)
print("FIRST 3 SEQUENCES (Examples)")
print("="*70)
for i in range(min(3, len(dataset_list))):
    ds = dataset_list[i]
    print(f"\nSequence {i}:")
    print(f"  Label: {ds.label}")
    print(f"  Length: {ds.length}")
    print(f"  Data shape: {ds.data.shape}")
    print(f"  First row (sensors): {ds.data[0]}")
    print(f"  Last row (sensors):  {ds.data[-1]}")
