import streamlit as st
import datetime
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 1. Cấu hình trang
st.set_page_config(page_title="Dự Báo Giá Cổ Phiếu bằng LSTM", page_icon="📈", layout="wide")

# 2. Thanh cấu hình Sidebar
st.sidebar.title("⚙️ Cấu Hình Dự Báo")
ma_ck = st.sidebar.text_input("Mã chứng khoán", value="VNM.VN")
ngay_bd = st.sidebar.date_input("Ngày bắt đầu lấy dữ liệu", datetime.date(2018, 1, 1))
ngay_kt = st.sidebar.date_input("Ngày kết thúc", datetime.date(2026, 10, 14))
so_ngay_du_bao = st.sidebar.slider("Số ngày muốn dự báo tương lai", 1, 60, 10)
so_epochs = st.sidebar.slider("Số Epochs huấn luyện", 1, 100, 10)

nut_bat_dau = st.sidebar.button("🚀 Bắt Đầu Dự Báo", type="primary")

# 3. Nội dung chính
st.title("📈 Ứng Dụng Dự Báo Giá Cổ Phiếu bằng Mô Hình LSTM")
st.write("Dự báo xu hướng giá cổ phiếu trong ngắn hạn và tính toán phần trăm **Lãi / Lỗ** kỳ vọng.")

if not nut_bat_dau:
    st.info("👈 Vui lòng bấm nút '🚀 Bắt Đầu Dự Báo' ở thanh bên trái để khởi chạy mô hình!")
else:
    with st.spinner("Đang tải dữ liệu và huấn luyện mô hình LSTM..."):
        # Tải dữ liệu từ Yahoo Finance
        df = yf.download(ma_ck, start=ngay_bd, end=ngay_kt)
        
        if df.empty:
            st.error(f"Không tìm thấy dữ liệu cho mã {ma_ck}. Vui lòng kiểm tra lại mã cổ phiếu!")
        else:
            st.success(f"Tải thành công dữ liệu mã {ma_ck}!")
            
            # Tiền xử lý dữ liệu
            data = df[['Close']].values
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data)
            
            time_step = 60
            X_train, y_train = [], []
            for i in range(time_step, len(scaled_data)):
                X_train.append(scaled_data[i-time_step:i, 0])
                y_train.append(scaled_data[i, 0])
                
            X_train, y_train = np.array(X_train), np.array(y_train)
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
            
            # Xây dựng mô hình LSTM
            model = Sequential([
                LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
                LSTM(units=50),
                Dense(units=1)
            ])
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(X_train, y_train, epochs=so_epochs, batch_size=32, verbose=0)
            
            # Dự báo tương lai
            temp_input = list(scaled_data[-time_step:, 0])
            lst_output = []
            for i in range(so_ngay_du_bao):
                x_input = np.array(temp_input[-time_step:]).reshape((1, time_step, 1))
                yhat = model.predict(x_input, verbose=0)
                temp_input.append(yhat[0][0])
                lst_output.append(yhat[0][0])
                
            predictions = scaler.inverse_transform(np.array(lst_output).reshape(-1, 1))
            
            # Tính phần trăm Lãi/Lỗ
            gia_hien_tai = data[-1][0]
            gia_du_bao_cuoi = predictions[-1][0]
            phan_tram = ((gia_du_bao_cuoi - gia_hien_tai) / gia_hien_tai) * 100
            
            # Hiển thị kết quả
            col1, col2 = st.columns(2)
            col1.metric("Giá hiện tại", f"{gia_hien_tai:,.0f} VND")
            col2.metric(f"Dự báo sau {so_ngay_du_bao} ngày", f"{gia_du_bao_cuoi:,.0f} VND", f"{phan_tram:+.2f}%")
            
            # Vẽ biểu đồ
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(df.index[-100:], data[-100:], label="Giá quá khứ")
            future_dates = [df.index[-1] + datetime.timedelta(days=i) for i in range(1, so_ngay_du_bao + 1)]
            ax.plot(future_dates, predictions, label="Giá dự báo", color='red', linestyle='--')
            ax.legend()
            st.pyplot(fig)
