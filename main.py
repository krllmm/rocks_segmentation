import io
import streamlit as st
from PIL import Image
import numpy as np
from PIL import ImageDraw
from ultralytics import YOLO
from io import BytesIO

st.set_page_config(
    page_title="Rock Segmentation",
    page_icon="🗿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

@st.cache_data()
def load_model():
    return YOLO("best.pt")  

def load_image():
    uploaded_file = st.file_uploader(label='Выберите изображение для сегментации')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None

model = load_model()

with st.sidebar:
    st.title("Информация о команде")
    st.text("Главный разработчик: Я")
    st.text("Обучатель сети: не Я")

st.title("Сегментация изображения горных пород")

img = load_image()
result = st.button('Распознать изображение')
if result:
    results = model(img)

    st.divider()

    st.success('Your edited image was processes! 🎉')
    st.write('**Результаты распознавания:**')

    result = results[0]
    masks = result.masks
    for mask in masks:
        polygon = mask.xy[0]
        draw = ImageDraw.Draw(img)
        draw.polygon(polygon,outline=(0,255,0), width=5)

    segmented = draw._image

    st.image(segmented)
    
    buf = BytesIO()
    img.save(buf, format="JPEG")
    byte_im = buf.getvalue()

    st.download_button(
        label="Download Image",
        data=byte_im,
        file_name="segmented_rocks.png",
        mime="image/jpeg",
    )
