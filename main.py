import io
import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.applications import EfficientNetB0 
from tensorflow.keras.preprocessing import image 
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions 


st.set_page_config(
    page_title="Rock Segmantation",
    page_icon="🗿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

@st.cache_data()
def load_model():
    return EfficientNetB0(weights='imagenet')


def preprocess_image(img):
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def load_image():
    uploaded_file = st.file_uploader(label='Выберите изображение для сегментации')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None


def print_predictions(preds):
    classes = decode_predictions(preds, top=3)[0]
    for cl in classes:
        st.write(cl[1], cl[2])
    


model = load_model()

with st.sidebar:
    st.title("Информация о команде")
    st.text("Главный разработчик: Я")
    st.text("Обучатель сети: не Я")

st.title("Сегментация изображения горных пород")

img = load_image()
result = st.button('Распознать изображение')
if result:
    x = preprocess_image(img)
    preds = model.predict(x)

    st.divider()

    st.success('Your edited image was processes! 🎉')
    st.write('**Результаты распознавания:**')
    print_predictions(preds)
    #st.download_button("Сохранить")


