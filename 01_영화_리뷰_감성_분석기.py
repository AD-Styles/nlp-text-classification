from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. 데이터 로드
dataset = load_dataset("sepidmnorozy/Korean_sentiment")
train_ds = dataset["train"]
val_ds = dataset["validation"]
test_ds = dataset["test"]

# 2. 문장 임베딩 생성
st_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def embed_dataset(split):
    texts = [item["text"] for item in split]
    labels = [item["label"] for item in split]
    embeddings = st_model.encode(texts, show_progress_bar=True)
    return embeddings, np.array(labels)

X_train, y_train = embed_dataset(train_ds)
X_val, y_val = embed_dataset(val_ds)
X_test, y_test = embed_dataset(test_ds)

# 3. 로지스틱 회귀로 감성 분류기 학습
clf = LogisticRegression(max_iter=2000, n_jobs=-1)
clf.fit(X_train, y_train)

val_pred = clf.predict(X_val)
print("Validation accuracy:", accuracy_score(y_val, val_pred))

# 4. 테스트셋 평가 및 예측 함수
test_pred = clf.predict(X_test)
print("Test accuracy:", accuracy_score(y_test, test_pred))

label_map = {0: "부정 👎", 1: "긍정 👍"}

def predict_sentiment(text: str):
    emb = st_model.encode([text])
    pred = clf.predict(emb)[0]
    proba = clf.predict_proba(emb)[0][pred]
    return label_map[pred], float(proba)

# 5. 테스트 실행
examples = [
    "이 영화 정말 최고야! 다음주에 2회차 관람해야지!",
    "정말 돈 아까운 영화. 정말로 할거없는사람만 보세요.",
    "와!!! 이 영화를 돈주고 2시간이나 앉아서봤다니...내 인내심에 박수를 보낸다!"
]

for t in examples:
    s, p = predict_sentiment(t)
    print(f"문장: {t}\n → 예측: {s} (확률 {p:.3f})\n")
