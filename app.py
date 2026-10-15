import streamlit as st
import random
import datetime
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# 1. Cấu hình giao diện Streamlit
st.set_page_config(page_title="Ứng Dụng Dự Báo Giá Cổ Phiếu Bằng AI (LSTM)", page_icon="📈", layout="wide")

# Tiêu đề ứng dụng
st.title("📈 Ứng Dụng Dự Báo Giá Cổ Phiếu Bằng AI (LSTM)")

# Ô nhập mã cổ phiếu và nút bấm
ticker_input = st.text_input("Nhập mã cổ phiếu (VD: VNM.VN, FPT.VN, AAPL):", value="VNM.VN")
btn_run = st.button("Tải dữ liệu & Chạy dự báo", type="primary")

if btn_run:
    with st.spinner("Đang tải dữ liệu, huấn luyện mô hình LSTM và tính toán..."):
        # Cố định Seed để ổn định kết quả dự báo
        seed = 42
        np.random.seed(seed)
        tf.random.set_seed(seed)
        random.seed(seed)

        # 2. Tải dữ liệu từ Yahoo Finance (theo logic trong file Word)
        df = yf.download(ticker_input, start="2018-01-01", end="2026-10-14")
        
        if df.empty:
            st.error(f"Không tìm thấy dữ liệu cho mã '{ticker_input}'. Vui lòng kiểm tra lại!")
        else:
            data = df[['Close']].values
            
            # Chia dữ liệu Train/Test (80% Train, 20% Test)
            train_size = int(0.8 * len(data))
            train_data = data[:train_size]
            test_data = data[train_size:]

            # Chuẩn hóa dữ liệu bằng MinMaxScaler
            scaler = MinMaxScaler()
            train_scaled = scaler.fit_transform(train_data)
            test_scaled = scaler.transform(test_data)

            # Hàm tạo chuỗi dữ liệu (window_size=60)
            def create_sequences(data_seq, window_size=60):
                X, y = [], []
                for i in range(window_size, len(data_seq)):
                    X.append(data_seq[i-window_size:i])
                    y.append(data_seq[i])
                return np.array(X), np.array(y)

            X_train, y_train = create_sequences(train_scaled)
            test_combined = np.vstack((train_scaled[-60:], test_scaled))
            X_test, y_test = create_sequences(test_combined)

            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

            # Chia tập Validation (10% từ Train)
            val_size = int(0.1 * len(X_train))
            X_val = X_train[-val_size:]
            y_val = y_train[-val_size:]
            X_train_final = X_train[:-val_size]
            y_train_final = y_train[:-val_size]

            # 3. Khởi tạo & Huấn luyện mô hình LSTM
            model = Sequential()
            model.add(LSTM(50, return_sequences=True, input_shape=(60, 1)))
            model.add(LSTM(50))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mean_squared_error')

            early_stop = EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )

            history = model.fit(
                X_train_final, y_train_final,
                validation_data=(X_val, y_val),
                epochs=10,
                batch_size=32,
                shuffle=False,
                callbacks=[early_stop],
                verbose=0
            )

            # Dự báo trên tập Test
            test_pred = model.predict(X_test, verbose=0)
            test_pred = scaler.inverse_transform(test_pred)
            y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1))

            # 4. Tính toán các chỉ số đánh giá
            rmse = np.sqrt(mean_squared_error(y_test_real, test_pred))
            mae = mean_absolute_error(y_test_real, test_pred)
            r2 = r2_score(y_test_real, test_pred)

            # Hiển thị Kết quả đánh giá mô hình
            st.subheader("📊 Kết quả đánh giá mô hình")
            c1, c2, c3 = st.columns(3)
            c1.metric("RMSE", f"{rmse:.2f}")
            c2.metric("MAE", f"{mae:.2f}")
            c3.metric("R² Score", f"{r2:.2f}")

            # 5. Dự báo 10 ngày tiếp theo trong tương lai
            last_60_days = np.vstack((train_scaled, test_scaled))[-60:]
            last_60_days = last_60_days.reshape(1, 60, 1)

            future_predictions = []
            for i in range(10):
                next_pred = model.predict(last_60_days, verbose=0)
                future_predictions.append(next_pred[0, 0])
                last_60_days = np.append(
                    last_60_days[:, 1:, :],
                    next_pred.reshape(1, 1, 1),
                    axis=1
                )

            future_predictions = np.array(future_predictions).reshape(-1, 1)
            future_predictions = scaler.inverse_transform(future_predictions)

            # Hiển thị bảng Dự báo 10 ngày tiếp theo
            st.subheader(f"🚀 Dự báo giá 10 ngày tiếp theo cho {ticker_input}")
            df_future = pd.DataFrame({
                "Ngày": [f"Ngày {i+1}" for i in range(10)],
                "Giá dự báo": [f"{val[0]:,.2f}" for val in future_predictions]
            })
            st.dataframe(df_future.set_index("Ngày"), use_container_width=True)

            # 6. Biểu đồ giá thật và dự báo
            st.subheader("📉 Biểu đồ giá thật và dự báo")
            fig1, ax1 = plt.subplots(figsize=(12, 6))
            ax1.plot(y_test_real, label="Giá thật")
            ax1.plot(test_pred, label="Dự báo test")
            future_x = range(len(y_test_real), len(y_test_real) + 10)
            ax1.plot(future_x, future_predictions, marker='o', label="10 ngày tới", color='green')
            ax1.legend()
            st.pyplot(fig1)

            # 7. Biểu đồ Loss Train vs Validation
            st.subheader("📉 Biểu đồ Loss Train vs Validation")
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            ax2.plot(history.history['loss'], label='Train Loss')
            ax2.plot(history.history['val_loss'], label='Validation Loss')
            ax2.set_title("Loss Train vs Validation")
            ax2.legend()
            st.pyplot(fig2)
