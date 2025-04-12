# utils/content_based_top1000.py
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Hàm chính để gợi ý sản phẩm từ mô hình đã lưu dạng dict
def search_and_recommend_top10(model_dict, keyword, top_k=10):
    product_df = model_dict["product_df"]
    vectorizer = model_dict["tfidf_vectorizer"]
    cosine_sim = model_dict["cosine_similarity"]

    # Tiền xử lý từ khóa tìm kiếm
    keyword_vector = vectorizer.transform([keyword])
    similarities = cosine_similarity(keyword_vector, vectorizer.transform(product_df['combined_text'])).flatten()

    # Lấy top-k sản phẩm có similarity cao nhất
    top_indices = similarities.argsort()[::-1][:top_k * 5]
    top_similarities = similarities[top_indices]

    result = product_df.iloc[top_indices].copy()
    result['similarity'] = top_similarities

    # Lọc ra những sản phẩm thực sự liên quan
    result = result[result['similarity'] > 0]

    # Lọc bỏ sản phẩm nghi ngờ là nữ nếu có cột possibly_female
    if "possibly_female" in result.columns:
        result = result[result["possibly_female"] == False]

    # Mô tả ngắn nếu chưa có
    if "short_description" not in result.columns:
        result["short_description"] = result["description"].astype(str).str.slice(0, 200)

    # Đổi tên cột tiếng Việt
    rename_cols = {
        "product_id": "Mã SP",
        "product_name": "Tên sản phẩm",
        "sub_category": "Loại sản phẩm",
        "price": "Giá",
        "rating": "Đánh giá",
        "short_description": "Mô tả",
        "similarity": "Độ tương đồng"
    }
    result.rename(columns=rename_cols, inplace=True)

    return result[["Mã SP", "Tên sản phẩm", "Loại sản phẩm", "Giá", "Đánh giá", "Mô tả", "Độ tương đồng"]].head(top_k)

# Gợi ý theo product_id (tùy chọn nếu muốn dùng thêm)
def recommend_by_product_id_top10(model_dict, product_id, top_k=10):
    product_df = model_dict["product_df"]
    cosine_sim = model_dict["cosine_similarity"]

    if product_id not in product_df['product_id'].values:
        raise ValueError("Mã sản phẩm không tồn tại trong dữ liệu.")

    index = product_df.index[product_df['product_id'] == product_id][0]
    scores = list(enumerate(cosine_sim[index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top_indexes = [i[0] for i in scores if i[0] != index][:top_k * 5]

    result = product_df.iloc[top_indexes].copy()
    result["similarity"] = [scores[i][1] for i in range(1, len(top_indexes)+1)]

    result = result[result["similarity"] > 0]

    if "possibly_female" in result.columns:
        result = result[result["possibly_female"] == False]

    if "short_description" not in result.columns:
        result["short_description"] = result["description"].astype(str).str.slice(0, 200)

    rename_cols = {
        "product_id": "Mã SP",
        "product_name": "Tên sản phẩm",
        "sub_category": "Loại sản phẩm",
        "price": "Giá",
        "rating": "Đánh giá",
        "short_description": "Mô tả",
        "similarity": "Độ tương đồng"
    }
    result.rename(columns=rename_cols, inplace=True)

    return result[["Mã SP", "Tên sản phẩm", "Loại sản phẩm", "Giá", "Đánh giá", "Mô tả", "Độ tương đồng"]].head(top_k)
