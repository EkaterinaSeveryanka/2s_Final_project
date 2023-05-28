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
image_1 = model_id.text_to_image("photograph of an astronaut riding a banana with old dragon", batch_size=3)


# Получение предсказания при помощи модели
def plot_images(images):
    plt.figure(figsize=(20, 20))
    for i in range(len(images)):
        ax = plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i])
        plt.axis("off")
