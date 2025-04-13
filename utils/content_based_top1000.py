# utils/content_based_top1000.py
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def search_and_recommend_top10(model_dict, keyword, top_k=10):
    product_df = model_dict["product_df"]
    vectorizer = model_dict["tfidf_vectorizer"]
    cosine_sim = model_dict["cosine_similarity"]

    # Vector hóa từ khóa và tính similarity
    keyword_vector = vectorizer.transform([keyword])
    similarities = cosine_similarity(keyword_vector, vectorizer.transform(product_df['combined_text'])).flatten()

    # Lấy top-k sản phẩm có similarity cao nhất
    top_indices = similarities.argsort()[::-1][:top_k * 5]
    top_similarities = similarities[top_indices]

    result = product_df.iloc[top_indices].copy()
    result['similarity'] = top_similarities
    result = result[result['similarity'] > 0]

    if "possibly_female" in result.columns:
        result = result[result["possibly_female"] == False]

    # Xử lý mô tả ngắn an toàn
    result["short_description"] = result["description"].fillna("").astype(str).str.slice(0, 200)

    # Đổi tên cột để hiển thị tiếng Việt
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


def recommend_by_product_id_top10(model_dict, product_id, top_k=10):
    product_df = model_dict["product_df"]
    cosine_sim = model_dict["cosine_similarity"]

    if product_id not in product_df['product_id'].values:
        raise ValueError("❌ Mã sản phẩm không tồn tại trong dữ liệu.")

    index = product_df[product_df['product_id'] == product_id].index.values[0]

    if index >= cosine_sim.shape[0]:
        raise ValueError("❌ Index vượt quá phạm vi của ma trận cosine_similarity.")

    scores = list(enumerate(cosine_sim[index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top_indexes = [i[0] for i in scores if i[0] != index][:top_k * 5]

    result = product_df.iloc[top_indexes].copy()
    result["similarity"] = [scores[i][1] for i in range(1, len(top_indexes)+1)]

    result = result[result["similarity"] > 0]

    if "possibly_female" in result.columns:
        result = result[result["possibly_female"] == False]

    result["short_description"] = result["description"].fillna("").astype(str).str.slice(0, 200)

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