import torch
from torch.utils.data import DataLoader
from dataset import DCASE_Dataset
from tqdm import tqdm
import os

from transformers import AutoModel
from peft import LoraConfig, get_peft_model

def extract_embeddings_and_stats():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("인코더 구조를 생성합니다...")
    encoder = AutoModel.from_pretrained(
        "worstchan/EAT-base_epoch30_pretrain",
        trust_remote_code=True
    )
    lora_config = LoraConfig(r=8, lora_alpha=32, target_modules=["qkv", "proj"])
    encoder = get_peft_model(encoder, lora_config)

    encoder_save_path = "best_encoder_model.pth"
    if not os.path.exists(encoder_save_path):
        print(f"'{encoder_save_path}' 파일을 찾을 수 없습니다.")
        return
        
    print(f"'{encoder_save_path}'에서 훈련된 인코더 가중치를 불러옵니다...")
    encoder.load_state_dict(torch.load(encoder_save_path))
    encoder.to(device)
    encoder.eval()

    print("'train' 폴더의 정상 데이터를 로딩합니다...")
    dataset = DCASE_Dataset("./dev_data", mode='train')
    
    normal_indices = [i for i, path in enumerate(dataset.file_list) if "normal" in os.path.basename(path)]
    normal_dataset = torch.utils.data.Subset(dataset, normal_indices)
    normal_loader = DataLoader(normal_dataset, batch_size=16, shuffle=False, num_workers=2)

    all_embeddings = []
    all_stats = []
    normal_filepaths = [dataset.file_list[i] for i in normal_indices]

    print("정상 데이터의 임베딩 및 통계 특성 추출을 시작합니다...")
    with torch.no_grad():
        for specs, stats, _, _, _ in tqdm(normal_loader, desc="Extracting"):
            specs = specs.to(device)
            
            outputs = encoder(specs)
            pooled_output = outputs.last_hidden_state[:, 0, :] if hasattr(outputs, "last_hidden_state") else outputs[:, 0, :]
            all_embeddings.append(pooled_output.cpu())
            all_stats.append(stats.cpu())

    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_stats = torch.cat(all_stats, dim=0)

    embedding_library = {'filepaths': normal_filepaths, 'embeddings': all_embeddings}
    emb_save_path = "normal_embeddings.pt"
    torch.save(embedding_library, emb_save_path)
    
    stats_save_path = "normal_stats.pt"
    torch.save(all_stats, stats_save_path)
    
    print("\n--- 추출 및 저장 완료 ---")
    print(f"  - 총 {len(normal_filepaths)}개의 정상 샘플이 처리되었습니다.")
    print(f"  - 임베딩이 '{emb_save_path}'에 저장되었습니다. (형태: {all_embeddings.shape})")
    print(f"  - 통계 특성이 '{stats_save_path}'에 저장되었습니다. (형태: {all_stats.shape})")

if __name__ == "__main__":
    extract_embeddings_and_stats()