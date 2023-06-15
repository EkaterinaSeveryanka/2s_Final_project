from fastapi import FastAPI
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

app = FastAPI()

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

@app.get("/")
async def root(prompt: str):
    image = pipe(prompt).images[0]
    image = Image.fromarray(image)

    return {"image": image}