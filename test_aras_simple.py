"""
ARAS Loader 동작 시뮬레이션 (numpy/pandas 없이)

실제 코드 로직을 간단하게 재현하여 어떻게 작동하는지 보여줍니다.
"""

class TSDataSet:
    def __init__(self, data, label, length):
        self.data = data
        self.label = int(label)
        self.length = int(length)

# 간단한 시뮬레이션 데이터
print("="*70)
print("ARAS Loader 동작 시뮬레이션")
print("="*70)
print("\n가상 데이터:")
print("  - 3개 파일 (DAY_1, DAY_2, DAY_3)")
print("  - timespan=5 (5초마다 샘플링)")
print("  - min_seq=10 (최소 10개 타임스텝)")
print()

# 시뮬레이션 시나리오
scenarios = [
    {
        'file': 'DAY_1.txt',
        'description': '거주자 1이 TV 시청 → 수면',
        'sequences': [
            {'resident': 1, 'label': 12, 'activity': 'Watching TV', 'length': 45, 'duration': '225초'},
            {'resident': 1, 'label': 11, 'activity': 'Sleeping', 'length': 120, 'duration': '600초'},
        ]
    },
    {
        'file': 'DAY_2.txt',
        'description': '거주자 2가 인터넷 → 식사 준비',
        'sequences': [
            {'resident': 2, 'label': 17, 'activity': 'Using Internet', 'length': 30, 'duration': '150초'},
            {'resident': 2, 'label': 7, 'activity': 'Preparing Dinner', 'length': 25, 'duration': '125초'},
        ]
    },
    {
        'file': 'DAY_3.txt',
        'description': '거주자 1과 2가 각각 활동 변경',
        'sequences': [
            {'resident': 1, 'label': 12, 'activity': 'Watching TV', 'length': 20, 'duration': '100초'},
            {'resident': 2, 'label': 11, 'activity': 'Sleeping', 'length': 50, 'duration': '250초'},
            {'resident': 1, 'label': 13, 'activity': 'Studying', 'length': 35, 'duration': '175초'},
        ]
    },
]

dataset_list = []
total_sequences = 0

for scenario in scenarios:
    print(f"\n파일: {scenario['file']}")
    print(f"  {scenario['description']}")
    print()
    
    for seq in scenario['sequences']:
        # TSDataSet 객체 생성 (data는 실제론 [seq_len, 20] 배열)
        seq_data = [[0]*20 for _ in range(seq['length'])]  # 간단히 0으로 채움
        ds = TSDataSet(seq_data, seq['label'], seq['length'])
        dataset_list.append(ds)
        total_sequences += 1
        
        print(f"  → 시퀀스 {total_sequences}:")
        print(f"      거주자: {seq['resident']}")
        print(f"      활동: {seq['activity']} (label={seq['label']})")
        print(f"      길이: {seq['length']} 타임스텝 (~{seq['duration']})")
        print(f"      데이터 shape: ({seq['length']}, 20)")

print("\n" + "="*70)
print("결과 요약")
print("="*70)
print(f"총 시퀀스 개수: {total_sequences}")

# 레이블 분포
from collections import Counter
labels = [ds.label for ds in dataset_list]
label_counts = Counter(labels)

print(f"\n활동 레이블 분포:")
activity_names = {
    11: 'Sleeping',
    12: 'Watching TV', 
    13: 'Studying',
    17: 'Using Internet',
    7: 'Preparing Dinner'
}
for label, count in sorted(label_counts.items()):
    percentage = (count / total_sequences) * 100
    print(f"  Label {label:2d} ({activity_names.get(label, 'Unknown'):20s}): {count} sequences ({percentage:.1f}%)")

# 시퀀스 길이 통계
lengths = [ds.length for ds in dataset_list]
print(f"\n시퀀스 길이 통계:")
print(f"  최소: {min(lengths)}")
print(f"  최대: {max(lengths)}")
print(f"  평균: {sum(lengths)/len(lengths):.1f}")
print(f"  중간값: {sorted(lengths)[len(lengths)//2]}")

print("\n" + "="*70)
print("첫 3개 시퀀스 상세 정보")
print("="*70)
for i in range(min(3, len(dataset_list))):
    ds = dataset_list[i]
    print(f"\n시퀀스 {i}:")
    print(f"  레이블: {ds.label} ({activity_names.get(ds.label, 'Unknown')})")
    print(f"  길이: {ds.length}")
    print(f"  데이터 shape: ({ds.length}, 20)")
    print(f"  첫 번째 row (센서 값): {ds.data[0][:10]}... (처음 10개만)")

print("\n" + "="*70)
print("실제 ARAS 데이터로 실행하면:")
print("="*70)
print("  - 수백~수천 개의 시퀀스 생성")
print("  - 각 시퀀스는 센서 변화에 따라 가변 길이")
print("  - 거주자 1과 2의 활동이 자동으로 분리되어 레이블링")
print("  - 활동 전환 시점이 정확하게 감지됨")
print("="*70)
