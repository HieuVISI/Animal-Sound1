import streamlit as st
import librosa
import numpy as np
import io
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

st.title("🐾 Phân loại âm thanh động vật")

# Upload âm thanh cho từng lớp
lion_files = st.file_uploader("Upload âm thanh Sư tử", type=["wav","mp3","ogg","flac"], accept_multiple_files=True)
bear_files = st.file_uploader("Upload âm thanh Gấu", type=["wav","mp3","ogg","flac"], accept_multiple_files=True)
dog_files = st.file_uploader("Upload âm thanh Chó", type=["wav","mp3","ogg","flac"], accept_multiple_files=True)
cat_files = st.file_uploader("Upload âm thanh Mèo", type=["wav","mp3","ogg","flac"], accept_multiple_files=True)
elephant_files = st.file_uploader("Upload âm thanh Voi", type=["wav","mp3","ogg","flac"], accept_multiple_files=True)

classify_file = st.file_uploader("Upload âm thanh để phân loại", type=["wav","mp3","ogg","flac"])

train_clicked = st.button("🚀 Train model")
classify_clicked = st.button("🔍 Phân loại")
reset_clicked = st.button("♻️ Reset")
export_clicked = st.button("📤 Xuất mô hình & thông số")

# Hàm trích xuất đặc trưng
def extract_features(file):
    y, sr = librosa.load(file, sr=None, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    return np.hstack([mfcc_mean, zcr, centroid, rolloff])

# Reset session
if reset_clicked:
    st.session_state.clear()
    st.success("Đã reset dữ liệu.")

# Train model
if train_clicked:
    X, y = [], []

    for f in lion_files:
        X.append(extract_features(f))
        y.append("Lion")
    for f in bear_files:
        X.append(extract_features(f))
        y.append("Bear")
    for f in dog_files:
        X.append(extract_features(f))
        y.append("Dog")
    for f in cat_files:
        X.append(extract_features(f))
        y.append("Cat")
    for f in elephant_files:
        X.append(extract_features(f))
        y.append("Elephant")

    if len(X) > 0:
        X = np.array(X)
        y = np.array(y)

        model = make_pipeline(StandardScaler(), SVC(probability=True))
        model.fit(X, y)

        st.session_state["model"] = model
        st.session_state["labels"] = list(set(y))
        st.session_state["X_train_size"] = len(X)

        preds = model.predict(X)
        rep = classification_report(y, preds)
        st.session_state["report"] = rep

        st.success("Đã train mô hình xong!")
        st.text(rep)

        cm = confusion_matrix(y, preds, labels=st.session_state["labels"])
        fig, ax = plt.subplots()
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(np.arange(len(st.session_state["labels"])))
        ax.set_yticks(np.arange(len(st.session_state["labels"])))
        ax.set_xticklabels(st.session_state["labels"])
        ax.set_yticklabels(st.session_state["labels"])
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        for i in range(len(st.session_state["labels"])):
            for j in range(len(st.session_state["labels"])):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="red")

        ax.set_title("Confusion Matrix")
        st.pyplot(fig)

        # Tải mô hình
        buf = io.BytesIO()
        joblib.dump(model, buf)
        buf.seek(0)
        st.download_button(
            label="💾 Tải về mô hình",
            data=buf,
            file_name="animal_sound_classifier.joblib",
            mime="application/octet-stream",
        )
    else:
        st.warning("Hãy upload ít nhất 1 file để train.")

# Classification
if classify_clicked:
    if "model" not in st.session_state:
        st.error("Chưa có mô hình, hãy train trước.")
    else:
        feats = extract_features(classify_file)
        pred = st.session_state["model"].predict([feats])[0]
        proba = st.session_state["model"].predict_proba([feats])[0]
        st.success(f"Âm thanh được phân loại là: **{pred}**")
        for label, p in zip(st.session_state["model"].classes_, proba):
            st.write(f"{label}: {p:.2f}")

# ---------- Export model info ----------
if export_clicked:
    if "model" not in st.session_state:
        st.error("Chưa có mô hình để xuất.")
    else:
        model = st.session_state["model"]
        labels = st.session_state.get("labels", [])
        n_train = st.session_state.get("X_train_size", 0)
        rep = st.session_state.get("report", "(chưa có)")

        st.subheader("📤 Thông tin mô hình")
        st.write(f"Số mẫu train: **{n_train}**")
        st.write(f"Nhãn: {labels}")
        st.code(rep, language="text")

        buf = io.BytesIO()
        joblib.dump(model, buf)
        buf.seek(0)
        st.download_button(
            label="💾 Tải về mô hình (.joblib)",
            data=buf,
            file_name="animal_sound_classifier.joblib",
            mime="application/octet-stream",
        )
