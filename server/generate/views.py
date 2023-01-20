from django.shortcuts import render
from django.http import HttpResponse, HttpRequest
from model import model as M

import json
import torch
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
            ))

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
