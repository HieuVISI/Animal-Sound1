import streamlit as st
model.fit(X, y)


st.session_state["model"] = model
st.session_state["labels"] = sorted(list(set(y)))
st.session_state["X_train_size"] = len(X)


y_pred = model.predict(X)
rep = classification_report(y, y_pred, digits=3)
cm = confusion_matrix(y, y_pred, labels=st.session_state["labels"])
st.session_state["report"] = rep


st.success("✅ Đã train xong mô hình!")
st.code(rep, language="text")
st.write("Confusion matrix:")
st.write(cm)


buf = io.BytesIO()
joblib.dump(model, buf)
buf.seek(0)
st.download_button(
label="💾 Tải về mô hình (.joblib)",
data=buf,
file_name="animal_sound_classifier.joblib",
mime="application/octet-stream",
)


# ---------- Classification ----------
if classify_clicked:
if "model" not in st.session_state:
st.error("Chưa có mô hình. Hãy train trước.")
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
order = np.argsort(-proba)
st.subheader("Kết quả dự đoán")
for idx in order:
st.write(f"**{classes[idx]}**: {proba[idx]*100:.2f}%")
except Exception as e:
st.error(f"Không thể phân loại file: {e}")


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


# ---------- Sidebar ----------
st.sidebar.header("Hướng dẫn nhanh")
st.sidebar.markdown(
"""
- Chuẩn bị mỗi lớp >= 5 file âm thanh, càng đa dạng càng tốt.
- File nên dài >= 1–2 giây, hạn chế tạp âm.
- Với dữ liệu lớn hơn, hãy tách train/test để đánh giá chuẩn.


**Cài đặt:**
```bash
pip install streamlit librosa scikit-learn joblib
streamlit run streamlit_app.py
```
"""
)
