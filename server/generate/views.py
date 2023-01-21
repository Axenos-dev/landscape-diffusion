from django.shortcuts import render
from django.http import HttpResponse, HttpRequest
from model import model as M
from model import diffusion

import json
import torch
import torchvision.transforms as T
import os


class GenerateView:
    num_images: int = 1
    random_seed: int = None

    model = M.UNET()

    def collect_images_data(self, request: HttpRequest):
        body = json.loads(request.body)

        try:
            self.num_images = int(body['num_images'])
            self.random_seed = int(body['random_seed'])
        except:
            pass


    def generate_images(self, request: HttpRequest) -> HttpResponse:
        self.collect_images_data(request=request)

        if self.random_seed: 
            torch.manual_seed(self.random_seed)

        try:
            self.initiate_model()

        except FileNotFoundError:
            return HttpResponse(json.dumps(
                {
                    "code": 500,
                    "description": "Failed to load model state dict",
                    "results": None
                }
            ), content_type="application/json")

        try:
            samples = self.create_samples(self.num_images, noise_steps=10)
        except:
            samples = None

            return HttpResponse(json.dumps(
                {
                    "code": 403,
                    "results": samples
                }
           ))

        transform = T.ToPILImage()
        # convert the tensor to PIL image using above transform
        img = transform(samples.squeeze())

        #TODO: Create image decoder to send decoded image binary as a successful response

        return HttpResponse(json.dumps(
            {
                "code": 200,
                "results": None
            }
        ))


    def initiate_model(self):
        self.model.load_state_dict(
            torch.load(
                f=os.path.join("model", "diffusion_model.pt"), 
                map_location=torch.device("cpu")
            )
        )


    def create_samples(self, n: int, noise_steps: int) -> torch.Tensor:
        # huge amount of noise steps and n(number of images) will increase time of generating image due weak cpu`s on web-hostings

        if n > 8: return None

        diff = diffusion.Diffusion(noise_steps=noise_steps)
        samples = diff.sample(self.model, n)

        return samples
