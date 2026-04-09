from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. 데이터 로드 및 샘플링
dataset = load_dataset("heegyu/korean-petitions")
df = pd.DataFrame(dataset["train"])

df_working = df[['title', 'content', 'category']].sample(n=5000, random_state=42).reset_index(drop=True)

train_df, test_df = train_test_split(
    df_working, test_size=0.2, random_state=42, stratify=df_working["category"]
)

# 2. 라벨 인코딩 & 텍스트 결합
train_df["text"] = train_df["title"] + " " + train_df["content"]
test_df["text"] = test_df["title"] + " " + test_df["content"]

label_encoder = LabelEncoder()
train_df["label"] = label_encoder.fit_transform(train_df["category"])
test_df["label"] = label_encoder.transform(test_df["category"])

# 3. 문장 임베딩
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def embed_texts(texts, batch_size=64):
    return model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)

X_train = embed_texts(train_df["text"].tolist())
y_train = train_df["label"].values
X_test = embed_texts(test_df["text"].tolist())
y_test = test_df["label"].values

# 4. 카테고리 분류기 학습
clf = LogisticRegression(max_iter=2000, n_jobs=-1)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(f"\n✅ Test Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")

# 5. 예측 함수
def predict_category(petition_text: str, top_k: int = 3):
    emb = model.encode([petition_text], convert_to_numpy=True)
    probs = clf.predict_proba(emb)[0]
    top_indices = np.argsort(-probs)[:top_k]

    print("\n=======================================")
    print("📝 입력한 청원 내용:\n", petition_text)
    print("🔮 예측된 카테고리 (Top-k):")
    for idx in top_indices:
        cat_name = label_encoder.classes_[idx]
        p = probs[idx]
        print(f"- {cat_name} (확률 {p:.3f})")
    print("=======================================\n")

# 6. 테스트 실행
sample_texts = [
    "초등학생 아이들 방과 후 돌봄 교실을 확대해 주세요. 맞벌이 가정이 너무 힘듭니다.",
    "최저임금 인상에 따른 소상공인들의 부담을 줄이기 위한 지원 정책을 마련해주세요.",
    "노인 복지를 강화하고, 동시에 청년 일자리 창출을 위한 예산 편성을 늘려주세요.",
    "AI 교육을 의무화하여 미래 인재를 양성하고, 관련 스타트업에 대한 투자를 확대해주세요.",
    "길거리 흡연 단속을 강화하고, 금연 구역을 확대하여 시민 건강을 보호해주세요."
]

for t in sample_texts:
    predict_category(t)
