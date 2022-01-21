import streamlit as st
import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image

### Function ###
def load_image(image_path):
    img = Image.open(image_path)
    img = img.convert('RGB')
    img = np.array(img)
    return img

def load_style_image(img, image_size=(256, 256)):
    img = img / 255.
    img = tf.image.resize(img, image_size)
    img = tf.expand_dims(img, axis=0)
    return img

def load_content_image(img):
    img = img / 255.
    # if img.shape[0] > 1080:
    #     img = tf.image.resize(img, (1080, 1080), preserve_aspect_ratio=True)
    # else:
    #     img = tf.convert_to_tensor(img, dtype=tf.float32)
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    img = tf.expand_dims(img, axis=0)
    return img

@st.cache
def load_model():
    # return hub.load('magenta_arbitrary-image-stylization-v1-256_2/')
    return hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
### Function ###

model = load_model()

### GUI ###
st.title('Art Work')

# Model
st.sidebar.selectbox("Model", ['TF-HUB', 'Model 1', 'Model 2'])

# Content Image
content = st.file_uploader('IMAGES TO PROCESS', type=['JPG', 'PNG'])
if content:
    file_name, in_format = content.name.split('.')
    content_img = load_image(content)
    st.image(content_img)
    content_img = load_content_image(content_img)

# Style Image
style_list_path = os.listdir('images/style/')
style_list = [s.split('.')[0].replace('_', ' ').title() for s in style_list_path]
style_list.append('Custom')
style_list.append('One For All')
style = st.sidebar.selectbox("Image Style: ", style_list)
all_styles = []
if style == 'Custom':
    style_custom = st.sidebar.file_uploader('Style image', type=['JPG', 'PNG'])
    if style_custom:
        style_img = load_image(style_custom)
        st.header("Style Image")
        st.image(style_img)
        style_img = load_style_image(style_img)
elif style == 'One For All':
    for style_index in range(len(style_list)-2):
        style_image_path = 'images/style/' + style_list_path[style_index]
        style_img = load_image(style_image_path)
        style_img = load_style_image(style_img)
        all_styles.append(style_img)
else:
    style_image_path = 'images/style/' + style_list_path[style_list.index(style)]
    style_img = load_image(style_image_path)
    st.header("Style Image")
    st.image(style_img)
    style_img = load_style_image(style_img)

# Resolution
resolution_type = ['Original', 'Custom', 'HD 16:9', 'Widescreen 16:10', 'Standard 4:3', 'Standard 5:4', 
                                        'Dual Monitors', 'Avatar|Profile Photo', 'Profile Covers', 'Mobile Resolutions',
                                        'Table Resolution', '2-in-1 Resolutions']
hd169 = [(1600, 900), (1920, 1080), (2048, 1152), (2560, 1440), (2880, 1620), (3360, 1890), (3840, 2160), (5120, 2880), (7680, 4320)]
wd1610 = [(1680, 1050), (1920, 1200), (2560, 1600), (2880, 1800), (3360, 2100), (3840, 2400), (5120, 3200), (8192, 5120)]
std43 = [(1600, 1200), (1920, 1440), (2048, 1536), (2560, 1920), (3200, 2400), (5120, 3840), (7680, 5760)]
std54 = [(1800, 1440), (2560, 2048), (3750, 3000), (5120, 4096)]
dm = [(2048, 768), (2560, 720), (2560, 1024), (3200, 1200), (3360, 1050), (3840, 1080), (3840, 1200), (5120, 2048)]
ava = [(150, 150), (165, 165), (180, 180), (256, 256), (300, 300), (400, 400), (512, 512)]
cv = [(851, 315), (1400, 425), (1500, 500), (2560, 1440)]
mb = [(480, 800), (540, 960), (640, 1136), (720, 1280), (750, 1334), (1080, 1920), (1125, 2436), (1440, 2560), (1440, 2960), (1440, 3040)]
tab = [(800, 1280), (1024, 1024), (1200, 1920), (1536, 2048), (1600, 2560), (1668, 2388), (2048, 2732)]
_2in1 = [(1920, 1080), (1920, 1200), (2160, 1440), (2736, 1824), (3200, 1800), (3240, 2160), (3840, 2160)]
resolution = dict(zip(resolution_type[2:], [hd169, wd1610, std43, std54, dm, ava, cv, mb, tab, _2in1]))
resolution_type = st.sidebar.selectbox("Resolution Type: ", resolution_type)
size = None
if resolution_type != 'Original':
    if resolution_type == 'Custom':
        width = st.sidebar.number_input("Width: ", min_value=0)
        heigth = st.sidebar.number_input("Height: ", min_value=0)
        if width and heigth:
            size = (width, heigth)
    else:
        size = st.sidebar.selectbox("Select resolution: ", resolution[resolution_type])

# Format
out_format = st.sidebar.selectbox("Save To Format: ", ['AUTO', 'JPG', 'PNG'])
if content and out_format == 'AUTO':
    out_format = in_format

# Button
if st.sidebar.button("Start Processing") and content:
    if len(all_styles) == 0:
        stylized_image = model(tf.constant(content_img), tf.constant(style_img))[0][0]
        if size:
            stylized_image = tf.image.resize(stylized_image, size[::-1])
#         tf.keras.preprocessing.image.save_img(f'images/stylized/{file_name}.{out_format}', stylized_image)
        show_img = tf.keras.preprocessing.image.array_to_img(stylized_image)
        st.header("Stylized Image")
        st.image(show_img)
    else:
        for i, style in enumerate(all_styles):
            stylized_image = model(tf.constant(content_img), tf.constant(style))[0][0]
            if size:
                stylized_image = tf.image.resize(stylized_image, size[::-1])
#             tf.keras.preprocessing.image.save_img(f'images/stylized/{file_name} - {style_list[i]}.{out_format.lower()}', stylized_image)
        show_img = tf.keras.preprocessing.image.array_to_img(stylized_image)
        st.header("Stylized Image")
        st.image(show_img)
