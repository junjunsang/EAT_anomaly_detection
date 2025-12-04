import torch
from dataset import DCASE_Dataset
import random
from collections import Counter

def verify():
    print("--- 데이터셋 검증을 시작합니다 ---")

    # 1. 학습(train) 데이터셋 로드 및 개수 확인
    print("\n[1] 학습 데이터셋 로딩...")
    try:
        train_dataset = DCASE_Dataset(root_dir="./dev_data", mode='train')
        print(f"  - 총 학습 데이터 개수: {len(train_dataset)}")
        
        # 타입별 개수 카운트
        train_types = [d[1]['machine_type'].item() for d in train_dataset]
        type_map = {v: k for k, v in train_dataset.label_maps['machine_type'].items()}
        counts = Counter([type_map[t] for t in train_types])
        print("  - 타입별 개수:")
        for t, c in counts.items():
            print(f"    - {t}: {c}개")

    except Exception as e:
        print(f"  - 학습 데이터셋 로딩 실패: {e}")
        return

    # 2. 검증(test) 데이터셋 로드 및 개수 확인
    print("\n[2] 검증 데이터셋 로딩...")
    try:
        test_dataset = DCASE_Dataset(root_dir="./dev_data", mode='test')
        print(f"  - 총 검증 데이터 개수: {len(test_dataset)}")

        # 타입별 개수 카운트
        test_types = [d[1]['machine_type'].item() for d in test_dataset]
        type_map = {v: k for k, v in test_dataset.label_maps['machine_type'].items()}
        counts = Counter([type_map[t] for t in test_types])
        print("  - 타입별 개수:")
        for t, c in counts.items():
            print(f"    - {t}: {c}개")

    except Exception as e:
        print(f"  - 검증 데이터셋 로딩 실패: {e}")
        return

    # 3. 데이터 형태 확인 (첫 번째 데이터 샘플 기준)
    print("\n[3] 데이터 형태(Shape) 확인...")
    if len(train_dataset) > 0:
        spec, labels_dict, is_normal = train_dataset[0]
        print(f"  - 스펙트로그램(spec) 형태: {spec.shape}")
        print(f"  - 라벨(labels) 타입: {type(labels_dict)}")
        print(f"  - 전체 라벨 키(key) 개수: {len(labels_dict)}")
    else:
        print("  - 학습 데이터가 없어 형태를 확인할 수 없습니다.")

    # 4. 라벨 5개 샘플 플롯 (랜덤 추출)
    print("\n[4] 학습 데이터 라벨 샘플 5개 확인 (랜덤 추출)...")
    if len(train_dataset) >= 5:
        indices_to_plot = random.sample(range(len(train_dataset)), 5)
        for i in indices_to_plot:
            _, labels_dict, _ = train_dataset[i]
            
            # -100이 아닌, 실제 값이 할당된 라벨만 필터링하여 출력
            actual_labels = {key: val.item() for key, val in labels_dict.items() if val.item() != -100}
            
            print(f"\n  ▶ 샘플 인덱스 #{i}")
            print(f"    {actual_labels}")
    else:
        print("  - 데이터가 5개 미만이라 샘플을 출력할 수 없습니다.")
    
    print("\n--- 데이터셋 검증 완료 ---")

if __name__ == "__main__":
    # preprocessing.py가 필요하므로, 임시로 빈 함수를 만듭니다.
    # 실제 파일이 있다면 이 부분은 필요 없습니다.
    try:
        from preprocessing import mel_spectrogram
    except ImportError:
        def mel_spectrogram(waveform, sample_rate):
            return torch.randn(1, 80, 1024) # 임의의 텐서 반환
    
    verify()