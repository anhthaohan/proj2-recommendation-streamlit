# build_content_based_model.py
# python3 build_content_based_model.py
import pandas as pd
import joblib
import os
import re
from utils.content_based import ContentBasedRecommender

# Tạo thư mục lưu mô hình nếu chưa có
os.makedirs("models", exist_ok=True)

# Hàm tiền xử lý văn bản
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', '', text)
    return text

# Danh sách từ khoá nghi ngờ là sản phẩm nữ
female_keywords = [
    "nữ", "croptop", "váy", "đầm", "áo dây", "baby doll",
    "đồ bộ nữ", "xinh", "dễ thương", "form rộng", "áo đôi", "cặp đôi"
]

def is_suspect(text):
    text = str(text).lower()
    return any(word in text for word in female_keywords)

# Load dữ liệu
print("🔄 Đang load dữ liệu...")
df = pd.read_csv("Products_ThoiTrangNam_clean.csv")

# Tiền xử lý văn bản đầu vào để kết hợp TF-IDF
print("🔧 Đang tiền xử lý văn bản...")
df["combined_text"] = (df["product_name"].fillna("") + " " + df["clean_description"].fillna("")).apply(preprocess_text)

# Gắn nhãn sản phẩm nghi ngờ là nữ (chỉ dựa trên product_name để tránh over-filter)
df["possibly_female"] = df["product_name"].apply(is_suspect)

# Thống kê dữ liệu
print(f"✅ Dữ liệu sau xử lý: {df.shape[0]} sản phẩm | {df['possibly_female'].sum()} sản phẩm nghi là nữ")

# Xây dựng mô hình
print("🚀 Đang xây dựng mô hình Content-Based Filtering...")
model = ContentBasedRecommender(df)
model.build_model()

# Lưu mô hình ra file
joblib.dump(model, "models/content_based_model.pkl")
print("✅ Mô hình đã lưu vào models/content_based_model.pkl")
