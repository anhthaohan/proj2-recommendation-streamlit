import pandas as pd
import os
import requests

df = pd.read_csv("data/Products_ThoiTrangNam_clean.csv")
output_dir = "images/products"
os.makedirs(output_dir, exist_ok=True)

for _, row in df.iterrows():
    product_id = row['product_id']
    image_links = str(row.get("image", "")).split()
    if image_links:
        first_img = image_links[0]
        if first_img.startswith("http"):
            try:
                img_data = requests.get(first_img, timeout=5).content
                with open(f"{output_dir}/{product_id}.jpeg", "wb") as f:
                    f.write(img_data)
                print(f"✅ {product_id}")
            except:
                print(f"❌ {product_id} - lỗi tải ảnh")