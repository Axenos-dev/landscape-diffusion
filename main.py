import os
import torch
from tqdm import tqdm
from utils import get_data, save_images
import argparse
import sys
from torch import nn

from model import model as M
from diffusion import diffusion as diff

class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=64):
        self.device = "cuda" if torch.cuda.is_available else "cpu"
        
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        
        self.beta = self.prepare_noise_schedule()
        self.alpha = 1. - self.beta
        
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        
    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
    
    
    def noise_images(self, x, t):
        x = x.to('cpu')
        
        sqrt_aplha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        E = torch.randn_like(x)
        
        return sqrt_aplha_hat * x + sqrt_one_minus_alpha_hat * E, E
    
    
    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))
    
    
    def denoise_images(self, model, x):
        model = model.to(self.device)
        x = x.to(self.device)
        
        model.eval()
        
        with torch.no_grad():
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(x.shape[0]) * i).long().to(self.device)
                predicted_noise = model(x, t)
                
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                
                if i > 1:
                    noise = torch.randn_like(x).to(self.device)
                    
                else:
                    noise = torch.zeros_like(x).to(self.device)
                    
                x = 1 / torch.sqrt(alpha.to(self.device)) * (x - ((1 - alpha.to(self.device)) / (torch.sqrt(1 - alpha_hat.to(self.device)).to(self.device))) * predicted_noise.to(self.device)) + torch.sqrt(beta.to(self.device)) * noise
                
        model.train()
        
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x
    
    
    def sample(self, model, n):
        model.eval()
        
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                
                if i > 1:
                    noise = torch.randn_like(x).to(self.device)
                    
                else:
                    noise = torch.zeros_like(x).to(self.device)
                    
                x = 1 / torch.sqrt(alpha.to(self.device)) * (x - ((1 - alpha.to(self.device)) / (torch.sqrt(1 - alpha_hat.to(self.device)).to(self.device))) * predicted_noise.to(self.device)) + torch.sqrt(beta.to(self.device)) * noise
                
        model.train()
        
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device set to {device} -----------------")
    
    dataloader = get_data(args)
    model = M.UNET().to(device)
    
    print(f"Loading from checkpoint {args.checkpoint_name} \n")
    
    model.load_state_dict(torch.load(os.path.join("models", args.checkpoint_name)))
    
    print("Chackpoint loaded\n")
    
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    
    diffusion = Diffusion()
    
    l = len(dataloader)
    
    for epoch in range(421, args.epochs):
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
            save_images(sampled_images, os.path.join("generated_images", f"{epoch}.jpg"))
            
            
def launch_training():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.epochs = 500
    args.batch_size = 3
    args.image_size = 64
    args.dataset_path = "landscape_dataset"
    args.lr = 3e-4
    args.checkpoint_name = "4.2.ckpt.pt"
    train(args)
    
    
def create_sample():
    parser = argparse.ArgumentParser()
    parser.add_argument("num_of_images", type=int)
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = M.UNET().to(device=device)
    model.load_state_dict(torch.load(os.path.join("model", "checkpoints", "6.9.ckpt.pt")))
    
    diffusion = diff.Diffusion()
    
    sampled_images = diffusion.sample(model=model.to(device), n=args.num_of_images)
    
    print(sampled_images.shape)
    save_images(sampled_images, os.path.join("generated_images", "sample.jpg"))
    

def generate_from_image():
    parser = argparse.ArgumentParser()
    parser.add_argument("noise_steps", type=int)
    
    args = parser.parse_args()
    args.batch_size = 4
    args.image_size = 64
    args.dataset_path = "test_image"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = M.UNET().to(device)
    model.load_state_dict(torch.load(os.path.join("model", "checkpoints", "6.9.ckpt.pt")))
    
    dataloader = get_data(args)
    images = next(iter(dataloader))[0]
    
    diffusion = diff.Diffusion(noise_steps=args.noise_steps)
    
    t = diffusion.sample_timesteps(images.shape[0]).to(device=device)
    noise_image, _ = diffusion.noise_images(images, t)
    
    results = diffusion.denoise_images(model=model, x=noise_image)
    save_images(images=results, path=os.path.join("generated_images", "res.jpg"))
    
if __name__ == "__main__":
    generate_from_image()