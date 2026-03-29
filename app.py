import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os
from streamlit_paste_button import paste_image_button

st.set_page_config(page_title="Traffic Sign AI", layout="centered", page_icon="🚩")
st.title("Hệ Thống Nhận Diện Biển Báo 🗿🥀")
st.divider() # Vạch kẻ chia cắt cho nó gọn

@st.cache_resource
def load_model():
    model_path = os.path.join(os.getcwd(), 'best.pt')
    if not os.path.exists(model_path):
        st.error(f"Cooked! Không tìm thấy file model ở: {model_path}")
        st.stop()
    return YOLO(model_path)

model = load_model()

# Tạo 2 cột
col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Kéo thả ảnh vào đây...", type=['jpg', 'png', 'jpeg'])

with col2:
    st.write("Hoặc dán ảnh từ Clipboard (Ctrl + V):")
    pasted_image = paste_image_button("Click vào đây rồi Paste")

raw_image = None

if uploaded_file is not None:
    raw_image = Image.open(uploaded_file).convert('RGB')
elif pasted_image and pasted_image.data is not None:
    raw_image = pasted_image.data.convert('RGB')

if raw_image:
    st.image(raw_image, caption='Ảnh mày vừa đưa tao check...', use_container_width=True)
    st.divider()

    img_array_rgb = np.array(raw_image)
    img_array_bgr = cv2.cvtColor(img_array_rgb, cv2.COLOR_RGB2BGR)

    results = model(img_array_bgr, conf=0.05)
    
    st.write("Kết quả nhận diện:")
    
    boxes = results[0].boxes.data.tolist() # [x1, y1, x2, y2, conf, cls]
    
    if len(boxes) == 0:
         st.warning("Vẫn không thấy cái biển báo nào oni chan oi")
    else:
        boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
        
        for i, box in enumerate(boxes[:5]):
            conf = box[4]
            cls = int(box[5])
            name = model.names[cls]
            
            st.info(f"**Top {i+1}: {name}**")
            st.progress(float(conf), text=f"Độ tự tin: {conf:.1%}")

    st.divider()
    st.write("### Ảnh thực tế AI nó nhìn thấy (Đã vẽ khung):")

    res_plotted_bgr = results[0].plot()

    res_plotted_rgb = cv2.cvtColor(res_plotted_bgr, cv2.COLOR_BGR2RGB)
    
    st.image(res_plotted_rgb, caption='Đã đánh dấu biển báo', use_container_width=True)