import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os
from streamlit_paste_button import paste_image_button

# Config cái trang cho nó "bảnh"
st.set_page_config(page_title="Traffic Sign AI", layout="centered", page_icon="🚩")
st.title("Hệ Thống Nhận Diện Biển Báo 🗿🥀")
st.divider() # Vạch kẻ chia cắt cho nó gọn

# Load model (dùng os.path cho chuẩn bài, không lo lỗi đường dẫn trên Windows)
@st.cache_resource
def load_model():
    model_path = os.path.join(os.getcwd(), 'best.pt')
    if not os.path.exists(model_path):
        st.error(f"Cooked! Không tìm thấy file model ở: {model_path} 😭")
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

# Biến trung gian chứa ảnh gốc (chuẩn RGB của Pillow)
raw_image = None

if uploaded_file is not None:
    raw_image = Image.open(uploaded_file).convert('RGB')
elif pasted_image and pasted_image.data is not None:
    raw_image = pasted_image.data.convert('RGB')

# Xử lý khi có ảnh
if raw_image:
    st.image(raw_image, caption='Ảnh mày vừa đưa tao check...', use_container_width=True)
    st.divider()

    # --- ĐÂY LÀ CHỖ FIX LỖI MÀU SẮC RGB vs BGR ---
    # Chuyển PIL Image (RGB) sang Numpy Array (RGB)
    img_array_rgb = np.array(raw_image)
    # Chuyển từ RGB sang BGR cho chuẩn gu của thằng YOLO/OpenCV
    img_array_bgr = cv2.cvtColor(img_array_rgb, cv2.COLOR_RGB2BGR)

    # Đưa ảnh chuẩn BGR vào model phán (ép conf thấp xuống 0.05 để vớt vát mấy con ảnh mờ)
    results = model(img_array_bgr, conf=0.05)
    
    st.write("Kết quả nhận diện:")
    
    # Lấy thông tin từ results
    boxes = results[0].boxes.data.tolist() # [x1, y1, x2, y2, conf, cls]
    
    if len(boxes) == 0:
         st.warning("Vẫn không thấy cái biển báo nào, model tao chưa đủ khôn hoặc ảnh mày quá dị! 😭")
    else:
        # Sort theo độ tin cậy giảm dần
        boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
        
        # Hiển thị Top K (lấy tối đa 5 cái tốt nhất)
        for i, box in enumerate(boxes[:5]):
            conf = box[4]
            cls = int(box[5])
            name = model.names[cls]
            
            st.info(f"🚩 **Top {i+1}: {name}**")
            st.progress(float(conf), text=f"Độ tự tin: {conf:.1%}")

    st.divider()
    st.write("### Ảnh thực tế AI nó nhìn thấy (Đã vẽ khung):")
    # plot() nó sẽ vẽ lên cái ảnh BGR
    res_plotted_bgr = results[0].plot()
    # Chuyển ngược lại sang RGB để hiển thị trên Streamlit cho đúng màu
    res_plotted_rgb = cv2.cvtColor(res_plotted_bgr, cv2.COLOR_BGR2RGB)
    
    st.image(res_plotted_rgb, caption='Đã đánh dấu biển báo', use_container_width=True)