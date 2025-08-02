import torch
from transformers.models.clip import CLIPModel, CLIPProcessor, CLIPTokenizer
from PIL import Image
import numpy as np


class CLIP:
    def __init__(self):
        self.__device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.__model_ID = "openai/clip-vit-large-patch14"
        self.__model: CLIPModel = CLIPModel.from_pretrained(self.__model_ID).to(self.__device) # type: ignore
        processor = CLIPProcessor.from_pretrained(self.__model_ID, use_fast=True)
        assert isinstance(processor, CLIPProcessor)
        self.__processor: CLIPProcessor = processor
        self.__tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(self.__model_ID)

    def get_image_embedding(self, my_image: Image.Image):
        image = self.__processor(
            text = None,
            images = my_image,
            return_tensors="pt"
        )["pixel_values"].to(self.__model.device) # type: ignore
        embedding = self.__model.get_image_features(image).detach().flatten().reshape(1,-1)[0]
        embedding_as_np = np.nan_to_num(embedding.cpu().detach().numpy(), nan=0.0)
        return embedding_as_np


if __name__ == "__main__":
    clip: CLIP = CLIP()

    image_embedding = clip.get_image_embedding(Image.open("./Ralsei_overworld.webp"))

    print(image_embedding.shape)
