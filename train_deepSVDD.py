import torch
from pyod.models.deep_svdd import DeepSVDD
import joblib
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import glob

# =======================================================
# ✨ 여기서 주요 하이퍼파라미터를 쉽게 변경하며 테스트하세요.
# =======================================================
LEARNING_RATE = 1e-5      # 1e-4, 1e-5, 1e-6 순서로 시도
OPTIMIZER = 'sgd'         # 'adam' 또는 'sgd'
PCA_COMPONENTS = 64       # 32, 64, 128 등
HIDDEN_NEURONS = [32, 16] # PCA_COMPONENTS의 절반 정도로 시작
EPOCHS = 200              # 에폭을 늘려서 충분히 학습
# =======================================================

def train_all_models():
    embedding_dir = "normal_embeddings_per_type"
    model_dir = "svdd_models_per_type"
    os.makedirs(model_dir, exist_ok=True)

    embedding_files = glob.glob(os.path.join(embedding_dir, "*.pt"))

    if not embedding_files:
        print(f"오류: '{embedding_dir}' 디렉토리에서 임베딩 파일(.pt)을 찾을 수 없습니다.")
        return

    for file_path in embedding_files:
        machine_type = os.path.basename(file_path).replace('embeddings_', '').replace('.pt', '')
        print(f"\n{'='*20} [{machine_type}] 모델 훈련 시작 {'='*20}")

        library = torch.load(file_path, weights_only=True)
        normal_embeddings = library['embeddings'].numpy()

        scaler = StandardScaler()
        scaled_embeddings = scaler.fit_transform(normal_embeddings)
        joblib.dump(scaler, os.path.join(model_dir, f"scaler_{machine_type}.joblib"))

        pca = PCA(n_components=PCA_COMPONENTS)
        pca_embeddings = pca.fit_transform(scaled_embeddings)
        joblib.dump(pca, os.path.join(model_dir, f"pca_{machine_type}.joblib"))
        
        n_samples, n_features = pca_embeddings.shape
        print(f"  - [{machine_type}] 총 {n_samples}개, {n_features}차원 데이터로 훈련")

        # PyOD 라이브러리 업데이트 후에는 'learning_rate' 파라미터 사용
        clf = DeepSVDD(
            n_features=n_features,
            hidden_neurons=HIDDEN_NEURONS,
            epochs=EPOCHS,
            
            optimizer=OPTIMIZER,
            verbose=1,
            random_state=42
        )
        clf.fit(pca_embeddings)
        
        joblib.dump(clf, os.path.join(model_dir, f"deep_svdd_{machine_type}.joblib"))
        print(f"  - [{machine_type}] 모델 저장 완료")

if __name__ == "__main__":
    train_all_models()