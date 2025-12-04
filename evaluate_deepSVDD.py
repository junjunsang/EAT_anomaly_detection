import torch
from torch.utils.data import DataLoader
from dataset import DCASE_Dataset
from tqdm import tqdm
import os
import numpy as np
from sklearn.metrics import roc_auc_score
import joblib

from transformers import AutoModel
from peft import LoraConfig, get_peft_model
from transformers.utils import logging

def evaluate_per_type_model():
    logging.set_verbosity_error()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 모델들이 저장된 디렉토리
    model_dir = "svdd_models_per_type"

    if not os.path.exists(model_dir):
        print(f"오류: 모델 디렉토리 '{model_dir}'를 찾을 수 없습니다.")
        print("먼저 train_deepSVDD.py를 실행하여 타입별 모델을 훈련해주세요.")
        return

    # 각 타입별 모델, 스케일러, PCA 로드
    models = {}
    for machine_type_file in os.listdir(model_dir):
        if machine_type_file.startswith('deep_svdd_'):
            machine_type = machine_type_file.replace('deep_svdd_', '').replace('.joblib', '')
            try:
                models[machine_type] = {
                    'clf': joblib.load(os.path.join(model_dir, f"deep_svdd_{machine_type}.joblib")),
                    'scaler': joblib.load(os.path.join(model_dir, f"scaler_{machine_type}.joblib")),
                    'pca': joblib.load(os.path.join(model_dir, f"pca_{machine_type}.joblib")),
                }
            except FileNotFoundError:
                print(f"경고: [{machine_type}]의 모델 파일 일부를 찾을 수 없습니다. 해당 타입은 건너뜁니다.")
                continue
    
    print(f"총 {len(models)}개 타입의 모델을 로드했습니다: {list(models.keys())}")

    encoder = AutoModel.from_pretrained("worstchan/EAT-base_epoch30_pretrain", trust_remote_code=True)
    lora_config = LoraConfig(r=8, lora_alpha=32, target_modules=["qkv", "proj"])
    encoder = get_peft_model(encoder, lora_config)
    encoder.load_state_dict(torch.load("best_encoder_model.pth", weights_only=True))
    encoder.to(device)
    encoder.eval()

    test_dataset = DCASE_Dataset("./dev_data", mode='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    ground_truth_labels = []
    anomaly_scores = []
    machine_types_list = []

    with torch.no_grad():
        for spec, _, is_normal_batch, machine_type_str_batch in tqdm(test_loader, desc="평가 진행 중"):
            spec = spec.to(device)
            machine_type = machine_type_str_batch[0]
            
            # 해당 타입의 모델이 없으면 건너뛰기
            if machine_type not in models:
                continue

            # 1. 임베딩 추출
            outputs = encoder(spec)
            hidden_states = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs
            test_embedding = hidden_states[:, 0, :].cpu().numpy()
            
            # 2. 해당 타입에 맞는 스케일러와 PCA 적용
            scaled_embedding = models[machine_type]['scaler'].transform(test_embedding)
            pca_embedding = models[machine_type]['pca'].transform(scaled_embedding)

            # 3. 해당 타입에 맞는 SVDD 모델로 점수 계산
            score = models[machine_type]['clf'].decision_function(pca_embedding)[0]
            
            anomaly_scores.append(score)
            ground_truth_labels.append(0 if is_normal_batch[0].item() else 1) # 정상:0, 비정상:1
            machine_types_list.append(machine_type)

    # --- 전체 결과 출력 ---
    print("\n--- 분석 완료 ---")
    
    overall_auroc = roc_auc_score(ground_truth_labels, anomaly_scores)
    print(f"전체 평균 성능 (Overall AUROC): {overall_auroc:.4f}")
    print()

    # --- 기계 타입별 성능 출력 ---
    print("--- 기계 타입별 성능 ---")
    unique_machine_types = sorted(list(set(machine_types_list)))
    scores = np.array(anomaly_scores)
    labels = np.array(ground_truth_labels)
    types = np.array(machine_types_list)
    
    for machine_type in unique_machine_types:
        mask = (types == machine_type)
        type_scores = scores[mask]
        type_labels = labels[mask]
        
        # 해당 타입에 정상/비정상 데이터가 모두 있는지 확인
        if len(np.unique(type_labels)) > 1:
            type_auroc = roc_auc_score(type_labels, type_scores)
            print(f"- {machine_type.ljust(10)}: {type_auroc:.4f}")
        else:
            print(f"- {machine_type.ljust(10)}: (N/A - 단일 클래스)")

if __name__ == "__main__":
    evaluate_per_type_model()