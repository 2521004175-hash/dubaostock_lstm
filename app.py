import streamlit as st
import datetime

# Cấu hình trang
st.set_page_config(
    page_title="Dự Báo Giá Cổ Phiếu bằng LSTM",
    page_icon="📈",
    layout="wide"
)

# Thanh điều hướng (Sidebar)
st.sidebar.title("⚙️ Cấu Hình Dự Báo")

ma_ck = st.sidebar.text_input("Mã chứng khoán", value="VNM.VN")
ngay_bd = st.sidebar.date_input("Ngày bắt đầu lấy dữ liệu", datetime.date(2018, 1, 1))
ngay_kt = st.sidebar.date_input("Ngày kết thúc", datetime.date(2026, 10, 14))

so_ngay_du_bao = st.sidebar.slider("Số ngày muốn dự báo tương lai", 1, 60, 10)
so_epochs = st.sidebar.slider("Số Epochs huấn luyện", 1, 100, 10)

nut_bat_dau = st.sidebar.button("🚀 Bắt Đầu Dự Báo", type="primary")

# Trang chính
st.title("📈 Ứng Dụng Dự Báo Giá Cổ Phiếu bằng Mô Hình LSTM")
st.write("Dự báo xu hướng giá cổ phiếu trong ngắn hạn và tính toán phần trăm **Lãi / Lỗ** kỳ vọng.")

if not nut_bat_dau:
    st.info("👈 Vui lòng bấm nút '🚀 Bắt Đầu Dự Báo' ở thanh bên trái để khởi chạy mô hình!")
else:
    st.success(f"Đã nhận yêu cầu dự báo cho mã {ma_ck}!")