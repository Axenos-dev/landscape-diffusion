import argparse
import os
import torch
from utils import utils
from model import model as M
from diffusion import diffusion as diff


def generate_from_images():
    parser = argparse.ArgumentParser()
    parser.add_argument("noise_steps", type=int)
    parser.add_argument("batch_size", type=int)
    
    args = parser.parse_args()
    args.batch_size = args.batch_size
    args.image_size = 64
    args.dataset_path = "test_image"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Generating image on {device} device -------------\n")
    
    model = M.UNET().to(device)
    model.load_state_dict(torch.load(os.path.join("model", "checkpoints", "6.9.ckpt.pt")))
    print(f"Model loaded\n")
    
    dataloader = utils.get_data(args)
    images = next(iter(dataloader))[0]
    
    diffusion = diff.Diffusion(noise_steps=args.noise_steps)
    
    t = diffusion.sample_timesteps(images.shape[0]).to(device=device)
    noise_image, _ = diffusion.noise_images(images, t)
    
    results = diffusion.denoise_images(model=model, x=noise_image)
    utils.save_images(images=results, path=os.path.join("generated_images", "result.jpg"))
    
    print("\nImage generated at /generated_images/result.jpg")
    

if __name__ == "__main__":
    generate_from_images()