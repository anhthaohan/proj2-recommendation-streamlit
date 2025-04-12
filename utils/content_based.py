# utils/content_based.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedRecommender:
    def __init__(self, product_df):
        # Khởi tạo với dữ liệu sản phẩm
        self.product_df = product_df.copy()
        self.vectorizer = TfidfVectorizer(stop_words='english')  # Dùng TF-IDF để vector hóa văn bản
        self.similarity_matrix = None  # Ma trận cosine similarity

    def build_model(self):
        # Xây dựng mô hình TF-IDF và cosine similarity
        self._prepare_data()
        tfidf_matrix = self.vectorizer.fit_transform(self.product_df['combined_text'])
        self.similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    def _prepare_data(self):
        # Loại bỏ sản phẩm thiếu tên hoặc văn bản mô tả
        self.product_df.dropna(subset=["product_name", "combined_text"], inplace=True)
        self.product_df.reset_index(drop=True, inplace=True)

    def recommend_by_product_id(self, product_id, top_k=10):
        # Gợi ý sản phẩm dựa vào ID sản phẩm
        if product_id not in self.product_df['product_id'].values:
            raise ValueError("Mã sản phẩm không tồn tại trong dữ liệu.")

        # Tìm vị trí chỉ mục của sản phẩm cần gợi ý
        index = self.product_df.index[self.product_df['product_id'] == product_id][0]

        # Tính điểm tương đồng với tất cả sản phẩm khác
        similarity_scores = list(enumerate(self.similarity_matrix[index]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        similarity_scores = [x for x in similarity_scores if x[0] != index][:top_k * 5]  # lấy dư để còn lọc

        # Lấy thông tin sản phẩm tương ứng với top điểm cao nhất
        recommendations = self.product_df.iloc[[i[0] for i in similarity_scores]].copy()
        recommendations['similarity'] = [i[1] for i in similarity_scores]

        # Loại bỏ sản phẩm không liên quan (similarity = 0)
        recommendations = recommendations[recommendations['similarity'] > 0]

        # Lọc bỏ sản phẩm nghi ngờ là dành cho nữ (nếu có cột này)
        if "possibly_female" in recommendations.columns:
            recommendations = recommendations[recommendations["possibly_female"] == False]

        # Nếu chưa có short_description, tạo mô tả ngắn từ description
        if "short_description" not in recommendations.columns:
            recommendations["short_description"] = recommendations["description"].astype(str).str.slice(0, 200)

        # Đổi tên cột sang tiếng Việt cho người dùng
        recommendations = recommendations.rename(columns={
            'product_id': 'Mã SP',
            'product_name': 'Tên sản phẩm',
            'sub_category': 'Loại sản phẩm',
            'price': 'Giá',
            'rating': 'Đánh giá',
            'short_description': 'Mô tả',
            'similarity': 'Độ tương đồng'
        })

        return recommendations[['Mã SP', 'Tên sản phẩm', 'Loại sản phẩm', 'Giá', 'Đánh giá', 'Mô tả', 'Độ tương đồng']].head(top_k)

    def search_and_recommend(self, keyword, top_k=10):
        # Vector hóa từ khóa tìm kiếm
        query_vector = self.vectorizer.transform([keyword])

        # Tính similarity giữa từ khóa và tất cả sản phẩm
        similarities = cosine_similarity(query_vector, self.vectorizer.transform(self.product_df['combined_text'])).flatten()

        # Lấy top sản phẩm có similarity cao nhất
        top_indices = similarities.argsort()[::-1][:top_k * 5]
        top_similarities = similarities[top_indices]

        # Tạo DataFrame kết quả
        recommendations = self.product_df.iloc[top_indices].copy()
        recommendations['similarity'] = top_similarities

        # Lọc ra những sản phẩm thực sự liên quan
        recommendations = recommendations[recommendations['similarity'] > 0]

        if "possibly_female" in recommendations.columns:
            recommendations = recommendations[recommendations["possibly_female"] == False]

        if "short_description" not in recommendations.columns:
            recommendations["short_description"] = recommendations["description"].astype(str).str.slice(0, 200)

        # Đổi tên cột để hiển thị
        recommendations = recommendations.rename(columns={
            'product_id': 'Mã SP',
            'product_name': 'Tên sản phẩm',
            'sub_category': 'Loại sản phẩm',
            'price': 'Giá',
            'rating': 'Đánh giá',
            'short_description': 'Mô tả',
            'similarity': 'Độ tương đồng'
        })

        return recommendations[['Mã SP', 'Tên sản phẩm', 'Loại sản phẩm', 'Giá', 'Đánh giá', 'Mô tả', 'Độ tương đồng']].head(top_k)
