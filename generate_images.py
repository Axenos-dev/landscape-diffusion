import argparse
import os
import torch
from utils import utils
from model import model as M
from diffusion import diffusion as diff


def create_sample(num_of_images):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Generating image on {device} device -------------\n")
    
    model = M.UNET().to(device=device)
    model.load_state_dict(torch.load(os.path.join("model", "checkpoints", "6.9.ckpt.pt")))
    print(f"Model loaded\n")
    
    diffusion = diff.Diffusion(noise_steps=1000)
    
    sampled_images = diffusion.sample(model=model.to(device), n=num_of_images)
    
    utils.save_images(sampled_images, os.path.join("static", "generated_images", "sample.jpg"))
    print("\nImage generated at /static/generated_images/sample.jpg")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("num_of_images", type=int)
    
    args = parser.parse_args()
    
    create_sample(args.num_of_images)
