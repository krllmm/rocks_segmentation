import io
import streamlit as st
from PIL import Image, ImageDraw
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
    st.text("Главный разработчик: ")
    st.text("Обучатель сети: ")

st.title("Сегментация изображения горных пород")

img = load_image()
result = st.button('Сегментировать изображение')
if result:
    results = model(img)

    st.divider()

    st.success('Ваше изображение успешно обработано! 🎉')
    st.write('**Результаты сегментиования:**')

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
        label="Скачать изображение",
        data=byte_im,
        file_name="segmented_rocks.png",
        mime="image/jpeg",
    )
