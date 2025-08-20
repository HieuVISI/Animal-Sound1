import streamlit as st
import numpy as np
import io
import tempfile
import soundfile as sf
import librosa
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import joblib

st.set_page_config(page_title="Lion vs Bear Sound Classifier", page_icon="🦁🐻", layout="centered")
st.title("🦁🐻 Phân loại âm thanh: Sư tử vs Gấu")
st.write("Tải âm thanh để **train** cho 2 lớp và **phân loại** một âm thanh mới. Hỗ trợ: wav, mp3, ogg, flac.")

# ---------- Helpers ----------
SR = 22050
N_MFCC = 40

def _load_audio(file) -> np.ndarray:
    """Load audio from an uploaded file-like object using a temporary file."""
    with tempfile.NamedTemporaryFile(delete=True, suffix=".audio") as tmp:
        tmp.write(file.read())
        tmp.flush()
        y, _ = librosa.load(tmp.name, sr=SR, mono=True)
    return y


def extract_features_from_signal(y: np.ndarray, sr: int = SR) -> np.ndarray:
    # Basic safety for empty audio
    if y is None or len(y) == 0:
        return np.zeros(N_MFCC * 2 + 5)

    # Trim leading/trailing silence
    y, _ = librosa.effects.trim(y, top_db=30)

    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)

    # Extra timbre features (means)
    zcr = librosa.feature.zero_crossing_rate(y).mean()
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
    rms = librosa.feature.rms(y=y).mean()

    feats = np.hstack([mfcc_mean, mfcc_std, [zcr, centroid, rolloff, bandwidth, rms]])
    return feats.astype(np.float32)


def extract_features(files):
    X = []
    for f in files:
        try:
            f.seek(0)
            y = _load_audio(f)
            X.append(extract_features_from_signal(y))
        except Exception as e:
            st.warning(f"Không đọc được file {getattr(f, 'name', 'unknown')}: {e}")
    if len(X) == 0:
        return np.empty((0, N_MFCC * 2 + 5), dtype=np.float32)
    return np.vstack(X)


# ---------- UI for uploads ----------
col1, col2 = st.columns(2)
with col1:
    lion_files = st.file_uploader(
        "📥 Upload âm thanh **Sư tử** (nhiều file)",
        type=["wav", "mp3", "ogg", "flac"],
        accept_multiple_files=True,
        key="lion_uploader",
    )
with col2:
    bear_files = st.file_uploader(
        "📥 Upload âm thanh **Gấu** (nhiều file)",
        type=["wav", "mp3", "ogg", "flac"],
        accept_multiple_files=True,
        key="bear_uploader",
    )

st.divider()

classify_file = st.file_uploader(
    "🎯 Upload 1 file âm thanh để **phân loại**",
    type=["wav", "mp3", "ogg", "flac"],
    accept_multiple_files=False,
    key="classify_uploader",
)

# Buttons
colA, colB, colC = st.columns([1, 1, 1])
with colA:
    train_clicked = st.button("🚀 Train model", use_container_width=True)
with colB:
    classify_clicked = st.button("🔎 Phân loại", use_container_width=True)
with colC:
    clear_clicked = st.button("🧹 Reset", use_container_width=True)

if clear_clicked:
    for k in ["model", "report", "labels", "X_train_size"]:
        if k in st.session_state:
            del st.session_state[k]
    st.success("Đã reset phiên làm việc.")

# ---------- Training ----------
if train_clicked:
    if not lion_files or not bear_files:
        st.error("Cần upload **cả hai** nhóm âm thanh Sư tử và Gấu để train.")
    else:
        with st.spinner("Đang trích chọn đặc trưng và huấn luyện..."):
            X_lion = extract_features(lion_files)
            X_bear = extract_features(bear_files)

            y_lion = np.array(["lion"] * len(X_lion))
            y_bear = np.array(["bear"] * len(X_bear))

            X = np.vstack([X_lion, X_bear]) if len(X_lion) and len(X_bear) else np.empty((0, X_lion.shape[1] if len(X_lion) else N_MFCC*2+5))
            y = np.hstack([y_lion, y_bear]) if len(X) else np.array([])

            if len(X) < 4:
                st.error("Dữ liệu quá ít. Hãy upload thêm mỗi lớp tối thiểu 2–3 file.")
            else:
                # Model: Standardize + SVM (RBF)
                model = make_pipeline(
                    StandardScaler(),
                    SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)
                )
                model.fit(X, y)
                st.session_state["model"] = model
                st.session_state["labels"] = sorted(list(set(y)))
                st.session_state["X_train_size"] = len(X)

                # Simple self-eval (resubstitution)
                y_pred = model.predict(X)
                rep = classification_report(y, y_pred, digits=3)
                cm = confusion_matrix(y, y_pred, labels=["lion", "bear"])
                st.session_state["report"] = rep

                st.success("✅ Đã train xong mô hình!")
                st.code(rep, language="text")
                st.write("Confusion matrix (lion, bear):")
                st.write(cm)

                # Allow download of model
                buf = io.BytesIO()
                joblib.dump(model, buf)
                buf.seek(0)
                st.download_button(
                    label="💾 Tải về mô hình (.joblib)",
                    data=buf,
                    file_name="lion_bear_classifier.joblib",
                    mime="application/octet-stream",
                )

# ---------- Classification ----------
if classify_clicked:
    if "model" not in st.session_state:
        st.error("Chưa có mô hình. Hãy train trước (hoặc tải mô hình có sẵn và nạp vào mã).")
    elif classify_file is None:
        st.error("Hãy upload 1 file âm thanh để phân loại.")
    else:
        with st.spinner("Đang phân tích và dự đoán..."):
            try:
                classify_file.seek(0)
                ysig = _load_audio(classify_file)
                feats = extract_features_from_signal(ysig).reshape(1, -1)
                proba = st.session_state["model"].predict_proba(feats)[0]
                classes = st.session_state["model"].classes_
                # Sort for display
                order = np.argsort(-proba)
                st.subheader("Kết quả dự đoán")
                for idx in order:
                    st.write(f"**{classes[idx]}**: {proba[idx]*100:.2f}%")
            except Exception as e:
                st.error(f"Không thể phân loại file: {e}")

# ---------- Sidebar: Tips & Requirements ----------
st.sidebar.header("Hướng dẫn nhanh")
st.sidebar.markdown(
    """
- Chuẩn bị mỗi lớp **>= 5** file âm thanh, càng đa dạng càng tốt (gầm, rống, khoảng cách, môi trường).
- File nên đủ dài (>= 1–2 giây), hạn chế tạp âm nếu có thể.
- Với tập dữ liệu lớn hơn, hãy cân nhắc tách train/test thật sự (sklearn.model_selection).
- Bạn có thể đổi SVC thành RandomForest/LogReg tùy nhu cầu.

**Cài đặt (local/Colab):**
```bash
pip install streamlit librosa soundfile scikit-learn joblib
streamlit run streamlit_audio_classifier.py
```
    """
)
