from tqdm import tqdm
import torch

class Diffusion:
    def __init__(self, noise_steps: int=1000, beta_start: float=1e-4, beta_end: float=0.02, img_size: int=64):
        self.device = "cuda" if torch.cuda.is_available else "cpu"
        
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        
        self.beta = self.prepare_noise_schedule()
        self.alpha = 1. - self.beta
        
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        
    def prepare_noise_schedule(self) -> torch.Tensor:
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
    
    
    def noise_images(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.to('cpu')
        
        sqrt_aplha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        E = torch.randn_like(x)
        
        return sqrt_aplha_hat * x + sqrt_one_minus_alpha_hat * E, E
    
    
    def sample_timesteps(self, n: int) -> torch.Tensor:
        return torch.randint(low=1, high=self.noise_steps, size=(n,))
    
    
    def denoise_images(self, model, x: torch.Tensor) -> torch.Tensor:
        model = model.to(self.device)
        x = x.to(self.device)
        
        model.eval()
        
        with torch.no_grad():
            for i in tqdm(reversed(range(1, self.noise_steps))):
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
    
    
    def sample(self, model, n: int) -> torch.Tensor:
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