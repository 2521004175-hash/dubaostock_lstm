import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Sequential

st.set_page_config(page_title="Dự báo giá cổ phiếu bằng LSTM", page_icon="📈", layout="wide")
st.title("📈 Dự báo giá cổ phiếu bằng LSTM")

WINDOW = 60  # số phiên nhìn lại

c_a, c_b, c_c = st.columns([2, 1, 1])
ticker = c_a.text_input("Mã cổ phiếu (VD: VNM.VN, FPT.VN, AAPL):", value="VNM.VN").strip().upper()
horizon = c_b.slider("Số ngày dự báo", 1, 10, 5)
epochs = c_c.slider("Số epoch tối đa", 10, 60, 30)


# ---------- Dữ liệu & đặc trưng ----------
@st.cache_data(ttl=3600, show_spinner=False)
def load(tk):
    # Không đặt end: luôn lấy đến phiên giao dịch mới nhất
    df = yf.download(tk, start="2018-01-01", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def make_features(close: pd.Series) -> pd.DataFrame:
    """Đặc trưng chỉ suy ra từ giá, để dự báo đệ quy được."""
    ret = np.log(close).diff()
    return pd.DataFrame({
        "ret": ret,                                   # lợi suất log trong ngày
        "ret5": ret.rolling(5).mean(),                # xu hướng ngắn hạn
        "vol10": ret.rolling(10).std(),               # biến động 10 phiên
        "ma20": close / close.rolling(20).mean() - 1,  # lệch so với MA20
    })


def set_seed(s=42):
    random.seed(s)
    np.random.seed(s)
    tf.random.set_seed(s)


def metrics(actual, pred):
    err = actual - pred
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "mape": float(np.mean(np.abs(err / actual)) * 100),
    }


# ---------- Huấn luyện & đánh giá (có cache, bấm lại không train lại) ----------
@st.cache_resource(show_spinner=False)
def train_and_eval(tk, max_epochs):
    set_seed()
    df = load(tk)
    if df.empty or "Close" not in df:
        raise ValueError(f"Không tìm thấy dữ liệu cho mã '{tk}'.")
    close = df["Close"].dropna().astype(float)
    if len(close) < 500:
        raise ValueError("Mã này có quá ít dữ liệu (cần ít nhất khoảng 500 phiên).")

    feats = make_features(close).dropna()
    close = close.loc[feats.index]
    F = feats.values
    n = len(F)

    # Mẫu t: dùng WINDOW phiên kết thúc tại t để đoán lợi suất phiên t+1
    idx = np.arange(WINDOW - 1, n - 1)
    cut = int(0.8 * len(idx))  # 80% đầu theo thời gian = train, 20% cuối = test

    # Scaler chỉ học từ dữ liệu train
    scaler = StandardScaler().fit(F[: idx[cut - 1] + 2])
    Fs = scaler.transform(F)
    X = np.stack([Fs[t - WINDOW + 1: t + 1] for t in idx])
    y = Fs[idx + 1, 0]

    X_tr, y_tr = X[:cut], y[:cut]
    v = int(0.1 * len(X_tr))
    X_val, y_val = X_tr[-v:], y_tr[-v:]
    # Chừa khoảng trống WINDOW mẫu giữa train và validation để các cửa sổ không chồng lấn
    X_fit, y_fit = X_tr[: -(v + WINDOW)], y_tr[: -(v + WINDOW)]

    model = Sequential([
        Input((WINDOW, F.shape[1])),
        LSTM(32, return_sequences=True),
        Dropout(0.2),
        LSTM(16),
        Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="huber")
    hist = model.fit(
        X_fit, y_fit,
        validation_data=(X_val, y_val),
        epochs=max_epochs, batch_size=32, shuffle=False, verbose=0,
        callbacks=[EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)],
    )

    # Đánh giá 1 ngày trên tập test (mô hình chưa từng thấy)
    te = idx[cut:]
    p = model.predict(X[cut:], verbose=0).ravel()
    pred_ret = p * scaler.scale_[0] + scaler.mean_[0]
    prev = close.values[te]
    actual = close.values[te + 1]
    pred = prev * np.exp(pred_ret)
    act_ret = np.log(actual / prev)

    return {
        "model": model, "scaler": scaler, "close": close, "hist": hist.history,
        "dates": close.index[te + 1], "actual": actual, "pred": pred, "prev": prev,
        "m_model": metrics(actual, pred),
        "m_naive": metrics(actual, prev),  # baseline: giá mai = giá hôm nay
        "dir_acc": float(np.mean(np.sign(pred_ret) == np.sign(act_ret))),
        "sigma": float(np.std(act_ret - pred_ret)),
        "n_test": len(te), "epochs_run": len(hist.history["loss"]),
    }


def forecast(res, h):
    s = res["close"].copy()
    model, scaler = res["model"], res["scaler"]
    prices = []
    for _ in range(h):
        f = make_features(s.iloc[-(WINDOW + 40):]).dropna().values
        x = scaler.transform(f[-WINDOW:])[None]
        r = model.predict(x, verbose=0)[0, 0] * scaler.scale_[0] + scaler.mean_[0]
        new_price = s.iloc[-1] * np.exp(r)
        s = pd.concat([s, pd.Series([new_price], index=[s.index[-1] + pd.offsets.BDay(1)])])
        prices.append(new_price)
    prices = np.array(prices)
    steps = np.sqrt(np.arange(1, h + 1))
    lo = prices * np.exp(-1.96 * res["sigma"] * steps)
    hi = prices * np.exp(1.96 * res["sigma"] * steps)
    return s.index[-h:], prices, lo, hi


# ---------- Giao diện ----------
if st.button("Tải dữ liệu & chạy dự báo", type="primary"):
    st.session_state["job"] = (ticker, epochs)

job = st.session_state.get("job")
if job:
    try:
        with st.spinner("Đang tải dữ liệu và huấn luyện mô hình (lần đầu có thể mất vài phút)..."):
            res = train_and_eval(*job)
    except Exception as e:
        st.error(str(e))
        st.stop()

    close = res["close"]

    # 1) Chất lượng mô hình, so với baseline
    m, b = res["m_model"], res["m_naive"]
    st.subheader("📊 Mô hình có đáng tin không?")
    k1, k2, k3 = st.columns(3)
    k1.metric("Sai số trung bình (MAPE)", f"{m['mape']:.2f}%",
              f"{m['mape'] - b['mape']:+.2f}% so với baseline", delta_color="inverse")
    k2.metric("RMSE", f"{m['rmse']:,.0f}",
              f"{m['rmse'] - b['rmse']:+,.0f} so với baseline", delta_color="inverse")
    k3.metric("Đoán đúng hướng tăng/giảm", f"{res['dir_acc'] * 100:.1f}%",
              f"{(res['dir_acc'] - 0.5) * 100:+.1f} điểm so với tung đồng xu")

    # 2) Dự đoán 1 ngày trên giai đoạn test
    k = 120
    fig1, ax1 = plt.subplots(figsize=(11, 4.5))
    ax1.plot(res["dates"][-k:], res["actual"][-k:], label="Giá thật")
    ax1.plot(res["dates"][-k:], res["pred"][-k:], label="LSTM (đoán 1 ngày)")
    ax1.plot(res["dates"][-k:], res["prev"][-k:], "--", alpha=0.6, label="Baseline: giá hôm qua")
    ax1.set_title(f"{k} phiên gần nhất trong tập test")
    ax1.legend()
    fig1.autofmt_xdate()
    st.pyplot(fig1)

    # 3) Dự báo tương lai kèm vùng bất định
    dates, prices, lo, hi = forecast(res, horizon)
    st.subheader(f"🚀 Kịch bản {horizon} phiên tới cho {job[0]}")
    last = close.iloc[-1]
    table = pd.DataFrame({
        "Giá dự báo": [f"{v:,.0f}" for v in prices],
        "Thấp (~95%)": [f"{v:,.0f}" for v in lo],
        "Cao (~95%)": [f"{v:,.0f}" for v in hi],
        "Thay đổi so với hôm nay": [f"{(v / last - 1) * 100:+.1f}%" for v in prices],
    }, index=dates.strftime("%d/%m/%Y"))
    st.dataframe(table, use_container_width=True)

    tail = close.iloc[-90:]
    fig2, ax2 = plt.subplots(figsize=(11, 4.5))
    ax2.plot(tail.index, tail.values, label="Giá thật")
    ax2.plot(dates, prices, "o-", color="green", label="Dự báo")
    ax2.fill_between(dates, lo, hi, color="green", alpha=0.15, label="Vùng bất định ~95%")
    ax2.legend()
    fig2.autofmt_xdate()
    st.pyplot(fig2)

    with st.expander("Chi tiết kỹ thuật"):
        fig3, ax3 = plt.subplots(figsize=(8, 3.5))
        ax3.plot(res["hist"]["loss"], label="Train loss")
        ax3.plot(res["hist"]["val_loss"], label="Validation loss")
        ax3.set_xlabel("Epoch")
        ax3.legend()
        st.pyplot(fig3)
