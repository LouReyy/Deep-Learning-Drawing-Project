from os import remove
from fastapi import  FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image, ImageDraw
from uuid import uuid4
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse
from starlette.background import BackgroundTask


class ImageData(BaseModel):
    strokes: list
    box: list

app = FastAPI()

# Configurer CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/transform")
async def transform(image_data: ImageData):
    print(image_data.strokes)
    print(image_data.box)
    filepath = "./images/" + str(uuid4()) + ".png"
    print(filepath)  # Vérifiez que le chemin du fichier est correct
    img = transform_img(image_data.strokes, image_data.box)
    img.save(filepath)

    return FileResponse(filepath, background=BackgroundTask(remove, path=filepath))


app.mount("/", StaticFiles(directory="static", html=True), name="static")

def transform_img(strokes, box):
    # Calc cropped image size
    width = box[2] - box[0]
    height = box[3] - box[1]

    image = Image.new("RGB", (width, height), color=(255, 255, 255))
    image_draw = ImageDraw.Draw(image)

    for stroke in strokes:
        positions = []
        for i in range(0, len(stroke[0])):
            positions.append((stroke[0][i], stroke[1][i]))
        image_draw.line(positions, fill=(0, 0, 0), width=3)

    return image.resize(size=(28, 28))



# from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline
# import torch

# prj_path = "diffusion"
# model = "stabilityai/stable-diffusion-xl-base-1.0"
# pipe = DiffusionPipeline.from_pretrained(
#     model,
#     torch_dtype=torch.float16,
# )
# pipe.to("cuda")
# pipe.load_lora_weights(prj_path, weight_name="pytorch_lora_weights.safetensors")

# refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
#     "stabilityai/stable-diffusion-xl-refiner-1.0",
#     torch_dtype=torch.float16,
# )
# refiner.to("cuda")

# prompt = "photo of a sks dog in a bucket"

# seed = 42
# generator = torch.Generator("cuda").manual_seed(seed)
# image = pipe(prompt=prompt, generator=generator).images[0]
# image = refiner(prompt=prompt, generator=generator, image=image).images[0]
# image.save(f"generated_image.png")