import torch
from torch.utils.data import DataLoader
from dataset import DCASE_Dataset
from tqdm import tqdm
import os
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import torch.nn.functional as F
import argparse

from transformers import AutoModel
from peft import LoraConfig, get_peft_model

def evaluate_ensemble(k, w_emb=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    encoder_save_path = "best_encoder_model.pth"
    embedding_library_path = "normal_embeddings.pt"
    stats_library_path = "normal_stats.pt"

    if not all(os.path.exists(p) for p in [encoder_save_path, embedding_library_path, stats_library_path]):
        print(f"필요한 파일 중 일부가 없습니다.")
        return
        
    print("훈련된 인코더와 정상 라이브러리를 로딩합니다...")
    encoder = AutoModel.from_pretrained("worstchan/EAT-base_epoch30_pretrain", trust_remote_code=True)
    lora_config = LoraConfig(r=8, lora_alpha=32, target_modules=["qkv", "proj"])
    encoder = get_peft_model(encoder, lora_config)
    encoder.load_state_dict(torch.load(encoder_save_path))
    encoder.to(device)
    encoder.eval()

    library_emb = torch.load(embedding_library_path)
    normal_embeddings = library_emb['embeddings'].to(device)
    normal_stats = torch.load(stats_library_path).to(device)

    print("정상 데이터의 통계적 분포(평균, 공분산)를 계산합니다...")
    mean_stats = torch.mean(normal_stats, dim=0)
    cov_stats = torch.cov(normal_stats.T)
    cov_inv_stats = torch.linalg.inv(cov_stats + 1e-6 * torch.eye(cov_stats.size(0)).to(device))

    print(f"테스트 데이터셋의 이상 점수를 계산합니다 (K={k}, w_emb={w_emb})...")
    test_dataset = DCASE_Dataset("./dev_data", mode='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    ground_truth_labels = []
    scores_emb = []
    scores_stats = []
    machine_types = []

    with torch.no_grad():
        for spec, stats, _, is_normal_batch, machine_type_str_batch in tqdm(test_loader, desc="Evaluating"):
            spec, stats = spec.to(device), stats.to(device)
            is_normal = is_normal_batch[0].item()
            machine_type = machine_type_str_batch[0]
            
            outputs = encoder(spec)
            test_embedding = outputs.last_hidden_state[:, 0, :] if hasattr(outputs, "last_hidden_state") else outputs[:, 0, :]
            distances = 1 - F.cosine_similarity(test_embedding, normal_embeddings)
            top_k_distances, _ = torch.topk(distances, k, largest=False)
            score_emb = torch.mean(top_k_distances).item()
            
            delta = stats - mean_stats
            score_stat = torch.sqrt(torch.diag(delta @ cov_inv_stats @ delta.T)).item()

            scores_emb.append(score_emb)
            scores_stats.append(score_stat)
            ground_truth_labels.append(0 if is_normal else 1)
            machine_types.append(machine_type)

    def min_max_normalize(scores):
        scores_np = np.array(scores)
        min_val = scores_np.min()
        max_val = scores_np.max()
        if max_val - min_val > 0:
            return (scores_np - min_val) / (max_val - min_val)
        return scores_np - min_val # Avoid division by zero if all scores are the same

    scores_emb_norm = min_max_normalize(scores_emb)
    scores_stats_norm = min_max_normalize(scores_stats)
    
    final_scores = w_emb * scores_emb_norm + (1 - w_emb) * scores_stats_norm

    print("\n--- 분석 완료 ---")
    overall_auroc = roc_auc_score(ground_truth_labels, final_scores)
    print(f"전체 평균 성능 (Overall AUROC): {overall_auroc:.4f}\n")
    
 #   print("--- 오탐지/미탐지 분석 ---")
    fpr, tpr, thresholds = roc_curve(ground_truth_labels, final_scores)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
 #   print(f" ▷ 최적 임계값(Optimal Threshold): {optimal_threshold:.4f}")
    predictions = [1 if score >= optimal_threshold else 0 for score in final_scores]
    tn, fp, fn, tp = confusion_matrix(ground_truth_labels, predictions).ravel()
 #   print(f" ▶ 정상을 이상으로 잘못 판단 (오탐지, FP): {fp}개")
   # print(f" ▶ 이상을 정상으로 잘못 판단 (미탐지, FN): {fn}개")
  #  print(" --------------------------\n")

    print("--- 기계 타입별 성능 ---")
    unique_machine_types = sorted(list(set(machine_types)))
    labels = np.array(ground_truth_labels)
    types = np.array(machine_types)
    for machine_type in unique_machine_types:
        mask = (types == machine_type)
        if np.any(mask):
            type_scores = final_scores[mask]
            type_labels = labels[mask]
            if len(np.unique(type_labels)) > 1:
                type_auroc = roc_auc_score(type_labels, type_scores)
                print(f"  - {machine_type.ljust(10)}: {type_auroc:.4f}")
            else:
                print(f"  - {machine_type.ljust(10)}: (비정상 샘플 없음)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="앙상블(임베딩+통계) 기반 이상 탐지 평가 스크립트")
    parser.add_argument("-k", type=int, default=1, help="임베딩 점수 계산에 사용할 이웃(K)의 수")
    parser.add_argument("--w", type=float, default=1.0, help="최종 점수에서 임베딩 점수의 가중치 (0.0 ~ 1.0)")
    args = parser.parse_args()
    
    evaluate_ensemble(args.k, args.w)