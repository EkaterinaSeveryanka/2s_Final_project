from diffusers import StableDiffusionPipeline
import torch
import streamlit as st
import imageio
import accelerate

# model_id = "runwayml/stable-diffusion-v1-5"
# pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
# pipe = pipe.to("cuda")
#
# prompt = "a photo of an astronaut riding a horse on mars"
# image = pipe(prompt).images[0]
#
# image.save("astronaut_rides_horse.png")

# Загрузка модели
@st.cache(allow_output_mutation=True)
def load_model():
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    return pipe


pipe = load_model()

# Отображение формы загрузки пользовательской картинки
st.set_option('deprecation.showfileUploaderEncoding', False)
uploaded_file = st.file_uploader("Загрузите файл в формате PNG или JPG", type=["png", "jpg"])

# Получение предсказания при помощи модели
if uploaded_file is not None:
    image = imageio.imread(uploaded_file)
    st.image(output_image.astype('uint8'), caption='Обработанное изображение', use_column_width=True)

    with st.spinner("Идет обработка..."):
        input_image = torch.tensor([torch.tensor(image)])
        prompt = "a photo of an astronaut riding a horse on mars"
        output_image = pipe(prompt, init_image=input_image, num_images=1).images[0]

        st.image(output_image, caption='Обработанное изображение', use_column_width=True)
