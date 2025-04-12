# pages/recommendation.py
import streamlit as st
import pandas as pd
import joblib
import os

# ====== Load mô hình & dữ liệu ======
@st.cache_resource
def load_cb_model():
    path = "models/content_based_model_top1000.pkl"
    if not os.path.exists(path):
        st.error("❌ Không tìm thấy file content_based_model.pkl")
        st.stop()
    return joblib.load(path)

@st.cache_resource
def load_cf_model():
    path = "models/collaborative_model_svd.pkl"
    if not os.path.exists(path):
        st.error("❌ Không tìm thấy file collaborative_model_svd.pkl")
        st.stop()
    return joblib.load(path)

@st.cache_data
def load_products():
    return pd.read_csv("data/Products_ThoiTrangNam_clean.csv")

@st.cache_data
def load_ratings():
    return pd.read_csv("data/Products_ThoiTrangNam_rating_clean.csv", sep="\t")

# ====== Hiển thị sản phẩm gợi ý ======
def display_recommendations(result_df, is_cb=True):
    if result_df.empty:
        st.warning("🙁 Không tìm thấy sản phẩm phù hợp.")
    else:
        for _, row in result_df.iterrows():
            with st.container():
                cols = st.columns([1, 4])
                with cols[0]:
                    img_path = f"images/products/{row['Mã SP']}.jpeg"
                    fallback_img = "images/no_image.jpg"
                    if os.path.exists(img_path):
                        st.image(img_path, width=120)
                    elif os.path.exists(fallback_img):
                        st.image(fallback_img, width=120)
                    else:
                        st.image("https://via.placeholder.com/100x100.png?text=No+Image", width=120)

                with cols[1]:
                    mota = str(row['Mô tả'])
                    short_desc = mota[:200] + "..." if len(mota) > 200 else mota

                    st.markdown(f"""
                    **🧢 Tên sản phẩm:** {row['Tên sản phẩm']}  
                    **📦 Loại sản phẩm:** {row['Loại sản phẩm']}  
                    **💸 Giá:** {int(row['Giá']):,}₫  
                    **⭐ Đánh giá:** {float(row['Đánh giá']):.1f}  
                    **📖 Mô tả:** {short_desc}
                    """)

                    if is_cb and 'Độ tương đồng' in row:
                        st.markdown(f"📊 **Độ tương đồng:** {float(row['Độ tương đồng']):.3f}")
                    elif not is_cb and 'Dự đoán' in row and float(row['Dự đoán']) > 0:
                        st.markdown(f"📊 **Dự đoán:** {float(row['Dự đoán']):.1f}")

                st.markdown("---")

# ====== Giao diện chính gợi ý ======
def product_recommendation():
    st.header("🎯 Hệ thống gợi ý sản phẩm")

    method = st.selectbox("🔍 Chọn phương pháp gợi ý:", ["Gợi ý theo nội dung", "Gợi ý theo người dùng"])
    products_df = load_products()

    if method == "Gợi ý theo nội dung":
        model_cb = load_cb_model()

        from utils.content_based_top1000 import search_and_recommend_top10, recommend_by_product_id_top10

        search_mode = st.radio("Chọn cách tìm kiếm:", ["Từ khóa", "Mã sản phẩm"])
        if search_mode == "Từ khóa":
            keyword = st.text_input("Nhập từ khóa (ví dụ: áo thun)")
            if st.button("Gợi ý", key="btn_cb_keyword"):
                result = search_and_recommend_top10(model_cb, keyword, top_k=10)
                display_recommendations(result, is_cb=True)

        elif search_mode == "Mã sản phẩm":
            unique_ids = products_df["product_id"].dropna().unique()
            product_id = st.selectbox("Chọn mã sản phẩm:", unique_ids)

            if st.button("Gợi ý", key="btn_cb_product"):
                try:
                    result = recommend_by_product_id_top10(model_cb, product_id, top_k=10)
                    display_recommendations(result, is_cb=True)
                except Exception as e:
                    st.error(f"Lỗi: {e}")

    elif method == "Gợi ý theo người dùng":
        from utils.collaborative import get_top_n_recommendations
        model_cf = load_cf_model()
        ratings_df = load_ratings()

        user_map_df = ratings_df[['user_id', 'user']].drop_duplicates()
        user_map = dict(zip(user_map_df['user_id'], user_map_df['user']))

        # Hiển thị bảng user + user_id
        st.subheader("👥 Danh sách người dùng và mã ID")
        st.dataframe(user_map_df.reset_index(drop=True), use_container_width=True)

        user_ids = sorted(user_map.keys())
        selected_user = st.selectbox("Chọn User ID:", user_ids)

        if selected_user:
            user_name = user_map.get(selected_user, "Không xác định")
            st.markdown(f"👤 **Tên người dùng:** `{user_name}`")

        # Hiển thị sản phẩm đã đánh giá
        st.subheader("🛍️ Sản phẩm đã đánh giá:")
        user_rated_df = ratings_df[ratings_df['user_id'] == selected_user]
        rated_products = products_df[products_df['product_id'].isin(user_rated_df['product_id'])].copy()
        rated_products.rename(columns={
            'product_id': 'Mã SP',
            'product_name': 'Tên sản phẩm',
            'sub_category': 'Loại sản phẩm',
            'price': 'Giá',
            'rating': 'Đánh giá',
            'description': 'Mô tả'
        }, inplace=True)
        display_recommendations(rated_products, is_cb=False)

        if st.button("Gợi ý", key="btn_cf_user"):
            try:
                result = get_top_n_recommendations(
                    user_id=selected_user,
                    model=model_cf,
                    product_df=products_df,
                    ratings_df=ratings_df,
                    n=10
                )
                st.subheader("🎁 Gợi ý sản phẩm dựa trên hành vi người dùng:")
                display_recommendations(result, is_cb=False)
            except Exception as e:
                st.error(f"Lỗi khi gợi ý: {e}")
