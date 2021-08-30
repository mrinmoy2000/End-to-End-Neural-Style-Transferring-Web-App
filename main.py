import streamlit as st
from PIL import Image
import stylize_image

st.title("Fast Neural Style Transfer Web-app.")
st.write("Choose an existing image from the sidebar or upload your own....")

img = st.sidebar.selectbox(
    'Select Image',
    ('Captain America',
     'Batman',
     'Deadpool',
     'Dragon',
     'Godzilla',
     'Iron Man',
     'One Punch Man',
     'Spider Man',
     'Superman',
     'Thor',
     'Wonder Woman')
)
img = img+'.jpg'

style_name = st.sidebar.selectbox(
    'Select Style',
    ('Candy', 'Mosaic', 'Rain Princess', 'Udnie')
)

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file:
    input_image = uploaded_file
else:
    input_image = "content_images/" + img

model = "models/" + style_name + ".pth"
output_image = "output_images/" + style_name + "-" + img

st.write("### Source image:")
image = Image.open(input_image)
st.image(image, width=400)

clicked = st.button('Stylize')

if clicked:
    model = stylize_image.load_model(model)
    stylize_image.stylize(model, input_image, output_image)

    st.write('#### Output image:')
    image = Image.open(output_image)
    st.image(image, width=400)
