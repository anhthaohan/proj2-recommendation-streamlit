# utils/collaborative.py
import pandas as pd

def get_top_n_recommendations(user_id, model, product_df, ratings_df, n=5):
    """
    Trả về top-n sản phẩm được gợi ý cho user_id sử dụng collaborative filtering (Surprise).
    """
    # Lọc các sản phẩm mà user đã đánh giá
    rated_products = ratings_df[ratings_df['user_id'] == int(user_id)]['product_id'].unique()
    unrated_products = product_df[~product_df['product_id'].isin(rated_products)].copy()

    # Dự đoán rating cho từng sản phẩm chưa đánh giá
    predictions = []
    for pid in unrated_products['product_id']:
        try:
            pred = model.predict(str(user_id), str(pid))
            predictions.append((pid, pred.est))
        except:
            continue

    # Sắp xếp theo rating dự đoán giảm dần
    top_predictions = sorted(predictions, key=lambda x: x[1], reverse=True)[:n]
    top_ids = [pid for pid, _ in top_predictions]
    score_dict = dict(top_predictions)

    # Lấy thông tin sản phẩm tương ứng và giữ cả ảnh
    result_df = product_df[product_df['product_id'].isin(top_ids)].copy()
    result_df['Dự đoán'] = result_df['product_id'].map(score_dict)

    # Chuyển đổi cột thành tiếng Việt nếu chưa có
    rename_cols = {
        'product_id': 'Mã SP',
        'product_name': 'Tên sản phẩm',
        'sub_category': 'Loại sản phẩm',
        'price': 'Giá',
        'rating': 'Đánh giá',
        'description': 'Mô tả',
        'image': 'image'  # giữ nguyên không đổi tên
    }
    result_df.rename(columns={k: v for k, v in rename_cols.items() if k in result_df.columns}, inplace=True)

    # Loại bỏ sản phẩm không hợp lệ (giá = 0 hoặc mô tả trống)
    result_df = result_df[(result_df['Giá'] > 0) & (result_df['Mô tả'].notnull())]

    # Chỉ giữ các cột cần thiết
    final_cols = ['Mã SP', 'Tên sản phẩm', 'Loại sản phẩm', 'Giá', 'Đánh giá', 'Mô tả', 'image', 'Dự đoán']
    result_df = result_df[[col for col in final_cols if col in result_df.columns]]

    return result_df.reset_index(drop=True)