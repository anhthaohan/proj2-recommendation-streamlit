# build_content_based_light_model.py
# python3 build_content_based_light_model.py

import pandas as pd
import joblib
import os
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ========== Cài đặt ==========
input_file = "data/Products_ThoiTrangNam_clean.csv"
output_pkl = "models/content_based_model_top1000.pkl"
NUM_PRODUCTS = 1000  # Giới hạn số sản phẩm để giảm dung lượng .pkl

# ========== Hàm tiền xử lý ==========
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', '', text)
    return text

female_keywords = [
    "nữ", "croptop", "váy", "đầm", "áo dây", "baby doll",
    "đồ bộ nữ", "xinh", "dễ thương", "form rộng", "áo đôi", "cặp đôi"
]

def is_suspect(text):
    text = str(text).lower()
    return any(word in text for word in female_keywords)

# ========== Đọc & lọc dữ liệu ==========
print("📦 Đang load dữ liệu...")
df = pd.read_csv(input_file)

# Xử lý mô tả
df["combined_text"] = (df["product_name"].fillna("") + " " + df["clean_description"].fillna("")).apply(preprocess_text)

# Gắn nhãn sản phẩm nghi ngờ là nữ
df["possibly_female"] = df["product_name"].apply(is_suspect)

# Loại bỏ sản phẩm không hợp lệ
df = df[df["combined_text"].notnull() & (df["price"] > 0) & (~df["possibly_female"])]

# Lọc ngẫu nhiên 1000 sản phẩm (đảm bảo reproducibility)
df_sample = df.sample(n=min(NUM_PRODUCTS, len(df)), random_state=42).reset_index(drop=True)

print(f"✅ Đã chọn {df_sample.shape[0]} sản phẩm hợp lệ cho mô hình.")

# ========== Xây dựng mô hình TF-IDF ==========
print("🔍 Đang vector hóa TF-IDF...")
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df_sample["combined_text"])

print("📊 Đang tính toán ma trận cosine similarity...")
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# ========== Lưu mô hình ==========
print("💾 Đang lưu mô hình .pkl ...")
model = {
    "product_df": df_sample,
    "tfidf_vectorizer": vectorizer,
    "cosine_similarity": cosine_sim
}

joblib.dump(model, output_pkl)
print(f"🎉 Mô hình đã lưu vào {output_pkl}")