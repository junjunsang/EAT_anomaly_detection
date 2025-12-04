#  EAT-LoRA Anomaly Detection (DCASE Task 2)

이 프로젝트는 **DCASE Task 2 (Unsupervised Anomalous Sound Detection)** 를 해결하기 위한 딥러닝 모델링 파이프라인입니다.

**Efficient Audio Transformer (EAT)** 를 백본으로 사용하며, **LoRA (Low-Rank Adaptation)** 를 적용하여 파라미터 효율적인 튜닝을 수행합니다. 이상 탐지 단계에서는 **Deep SVDD** 방식과 **Ensemble (KNN + Statistics)** 방식을 모두 지원합니다.

## 📂 Project Structure

```bash
EAT-Anomaly-Detection/
├── dev_data/                   # (Not included in repo) Raw Dataset
│   ├── train/                  # Normal data for training
│   └── test/                   # Test data (Normal + Anomaly)
├── svdd_models_per_type/       # Saved SVDD models (Auto-generated)
├── model.py                    # EAT Classifier with LoRA
├── dataset.py                  # DCASE Dataset Loader
├── preprocessing.py            # Mel-Spectrogram transformation
├── train.py                    # Step 1: Encoder Representation Learning
├── extract_embeddings.py       # Step 2: Extract embeddings from normal data
├── train_deepSVDD.py           # Step 3-A: Train Deep SVDD models
├── evaluate_deepSVDD.py        # Step 3-A: Evaluate with Deep SVDD
├── evaluate.py                 # Step 3-B: Evaluate with Ensemble (KNN+Stats)
└── datashape.py                # Data integrity check
````

##  Prerequisites

필요한 라이브러리를 설치

```bash
pip install torch torchaudio transformers peft pyod scikit-learn tqdm numpy scipy joblib
```

##  Usage Pipeline

전체 파이프라인은 **[학습] -\> [특징 추출] -\> [이상 탐지 평가]** 순서로 진행됩니다.

### 0\. Data Preparation

DCASE 데이터셋을 `dev_data` 폴더에 위치시켜야 합니다.

  - `dev_data/train/`: 정상 데이터 (학습용)
  - `dev_data/test/`: 정상 및 이상 데이터 (테스트용)

데이터가 잘 로드되는지 확인하려면:

```bash
python datashape.py
```

### 1\. Encoder Training (Representation Learning)

기계의 속성(Attribute)을 분류하는 보조 과제(Auxiliary Task)를 통해 인코더를 학습시킵니다.

  - **Output:** `best_encoder_model.pth` (최고 성능 모델 가중치)

<!-- end list -->

```bash
python train.py
```

### 2\. Feature Extraction

학습된 인코더를 사용하여 `train` 폴더(정상 데이터)의 임베딩과 통계적 특징을 추출하고 라이브러리를 구축합니다.

  - **Output:** `normal_embeddings.pt`, `normal_stats.pt`

<!-- end list -->

```bash
python extract_embeddings.py
```

### 3\. Anomaly Detection & Evaluation

두 가지 방법 중 하나를 선택하거나 비교할 수 있습니다.

#### Option A: Deep SVDD (Recommended)

기계 타입(Machine Type)별로 독립된 SVDD 모델을 학습하고 평가합니다.

```bash
# SVDD 모델 학습 (svdd_models_per_type 폴더에 저장됨)
python train_deepSVDD.py

# 평가 수행 (AUROC 점수 출력)
python evaluate_deepSVDD.py
```

#### Option B: Ensemble (KNN + Statistics)

별도의 추가 학습 없이, 임베딩 거리(Cosine)와 통계적 거리(Mahalanobis)를 결합하여 평가합니다.

  - `-k`: KNN 탐색 시 고려할 이웃 수 (기본값: 1)
  - `--w`: 임베딩 점수 가중치 (0.0 \~ 1.0, 기본값: 1.0)

<!-- end list -->

```bash
python evaluate.py -k 1 --w 0.5
```


```

```

