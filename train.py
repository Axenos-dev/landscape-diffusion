import argparse
import os
import torch
from tqdm import tqdm
from utils import utils
from torch import nn

from model import model as M
from diffusion import diffusion as diff


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device set to {device} -----------------")
    
    dataloader = utils.get_data(args)
    model = M.UNET().to(device)
    
    print(f"Loading from checkpoint {args.checkpoint_name} \n")
    
    model.load_state_dict(torch.load(os.path.join("model", "checkpoints", args.checkpoint_name)))
    
    print("Chackpoint loaded\n")
    
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    
    diffusion = diff.Diffusion()
    
    l = len(dataloader)
    
    for epoch in range(1, args.epochs):
        print(f"EPOCH: {epoch}")
        
        prog_bar = tqdm(dataloader)
        
        for i, (images, _) in enumerate(prog_bar):
            
            images = images.to(device)
            
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            
            predicted_noise = model(x_t, t)
            loss = mse(noise.to(device), predicted_noise.to(device))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            prog_bar.set_postfix(MSE=loss.item())
        
        if epoch % 3 == 0:
            print(f"Creating {epoch/100} checkpoint on {epoch} epoch------------")
            torch.save(model.state_dict(), os.path.join("model", "checkpoints", f"{epoch/100}.ckpt.pt"))
            
        if epoch % 10 == 0:
            sampled_images = diffusion.sample(model=model.to(device), n=10)
            utils.save_images(sampled_images, os.path.join("generated_images", f"{epoch}.jpg"))
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.epochs = 500
    args.batch_size = 3
    args.image_size = 64
    args.dataset_path = "landscape_dataset"
    args.lr = 3e-4
    args.checkpoint_name = "6.9.ckpt.pt"
    train(args)