import torch
from tqdm import tqdm

class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=64):
        self.device = "cpu"
        
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        
        self.beta = self.prepare_noise_schedule()
        self.alpha = 1. - self.beta
        
        self.alpha_hat = torch.cumprod(self.alpha, dim=0).to(self.device)
        
    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps).to(self.device)
    
    def noise_images(self, x, t):
        x = x.to(self.device)
        
        sqrt_aplha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None].to(self.device)
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None].to(self.device)
        E = torch.randn_like(x).to(self.device)
        
        return sqrt_aplha_hat * x + sqrt_one_minus_alpha_hat * E, E
    
    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,)).to(self.device)
    
    def sample(self, model, n):
        model.eval()
        
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                model = model.to(self.device)
                
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