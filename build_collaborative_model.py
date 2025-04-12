# build_collaborative_model.py
import pandas as pd
import os
from surprise import Dataset, Reader, SVD
from surprise import dump

# ===== Bước 1: Đọc dữ liệu gốc =====
rating_path = "data/Products_ThoiTrangNam_rating_clean.csv"
df = pd.read_csv(rating_path, sep="\t")
print(f"📊 Tổng số dòng dữ liệu: {len(df)}")

# ===== Bước 2: Lọc user có ít nhất 3 lượt đánh giá =====
user_counts = df['user_id'].value_counts()
valid_users = user_counts[user_counts >= 3].index
filtered_df = df[df['user_id'].isin(valid_users)].copy()
print(f"✅ Số user đủ điều kiện (≥3 đánh giá): {len(valid_users)}")
print(f"📂 Số dòng sau khi lọc: {len(filtered_df)}")

# ===== Bước 3: Huấn luyện mô hình collaborative filtering với Surprise =====
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(filtered_df[['user_id', 'product_id', 'rating']], reader)
trainset = data.build_full_trainset()

model = SVD()
print("🤖 Đang huấn luyện mô hình SVD...")
model.fit(trainset)
print("✅ Huấn luyện xong.")

# ===== Bước 4: Lưu mô hình bằng dump =====
os.makedirs("models", exist_ok=True)
model_path = "models/collaborative_model_svd.pkl"
dump.dump(model_path, algo=model)
print(f"💾 Mô hình đã lưu tại: {model_path} (dùng dump của surprise)")