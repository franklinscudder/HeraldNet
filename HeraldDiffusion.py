import torch.nn as nn
import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from UNet2 import UNet
import torch.nn.functional as F

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

IMAGE_X = 128
IMAGE_Y = 128

transform_comp = transforms.Compose([
    transforms.Resize([IMAGE_X, IMAGE_Y], interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ColorJitter(brightness=.3, hue=.1),
    transforms.ToTensor(),
])


"""
chan 0: mean = 0.6880784630775452, var = 0.10271414369344711
chan 1: mean = 0.6260817050933838, var = 0.11727848649024963
chan 2: mean = 0.5751838088035583, var = 0.14141134917736053
"""

def norm_img_batch(batch):
    batch[:, 0, :, :] -= 0.6880784630775452
    batch[:, 0, :, :] /= 0.10271414369344711

    batch[:, 1, :, :] -= 0.6260817050933838
    batch[:, 1, :, :] /= 0.11727848649024963

    batch[:, 2, :, :] -= 0.5751838088035583
    batch[:, 2, :, :] /= 0.14141134917736053

    return batch   

def inv_norm_batch(batch):
    batch[:, 0, :, :] *= 0.10271414369344711
    batch[:, 0, :, :] += 0.6880784630775452
 
    batch[:, 1, :, :] *= 0.11727848649024963
    batch[:, 1, :, :] += 0.6260817050933838

    batch[:, 2, :, :] *= 0.14141134917736053
    batch[:, 2, :, :] += 0.5751838088035583

    return batch

def gather(consts, t):
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1, 1)

class Diffusion:
    def __init__(self, T):
        self.T = T
        self.unet = UNet()
        self.unet_loss_fn = nn.MSELoss()
        self.unet_optimizer = torch.optim.Adam(self.unet.parameters(), lr=0.0005)

        self.beta = torch.linspace(0.0001, 0.02, T).to(dev)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.sigma2 = self.beta

    def q_xt_x0(self, x0, t):
        mean = gather(self.alpha_bar, t) ** 0.5 * x0
        var = 1 - gather(self.alpha_bar, t)
        return mean, var

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)

        mean, var = self.q_xt_x0(x0, t)

        return mean + (var ** 0.5) * noise

    def p_sample(self, xt, t):
        unet_noise = self.unet(xt, t)
        alpha_bar = gather(self.alpha_bar, t)
        alpha = gather(self.alpha, t)
        eps_coef = (1 - alpha) / (1 - alpha_bar) ** .5
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * unet_noise)
        var = gather(self.sigma2, t)
        eps = torch.randn(xt.shape, device=xt.device)
        return mean + (var ** .5) * eps

    def loss(self, x0, noise=None):
        batch_size = x0.shape[0]
        t = torch.randint(0, self.T, (batch_size,), device=x0.device, dtype=torch.long)
        if noise is None:
            noise = torch.randn_like(x0)

        xt = self.q_sample(x0, t, noise=noise)
        eps_theta = self.unet(xt, t)
        return F.mse_loss(noise, eps_theta)

    def train(self, x): 
        self.unet_optimizer.zero_grad()
        loss = self.loss(x)
            
        print(f"e = {epoch} - b = {i} - loss = {round(loss.item(), 6)}")

        loss.backward()
        self.unet_optimizer.step()

    def generate(self):
        with torch.no_grad():
            x = torch.randn([self.T, 3, IMAGE_X, IMAGE_Y],
                            device=dev)
            for t_ in range(self.T):
                print(t_)
                t = self.T - t_ - 1
                x = self.p_sample(x, x.new_full((self.T,), t, dtype=torch.long))

            return x


if __name__ == '__main__':
    plt.ion()

    has_cuda = True if torch.cuda.is_available() else False
    dev = "cuda:0" if has_cuda else "cpu"

    print("Using device:", dev)

    data_dir = r"C:\Users\Tom\Documents\GitHub\HeraldryData\data\data"
    batch_size = 4
    num_epochs = 3000
    data = ImageFolder(data_dir, transform_comp)
    data_loader = DataLoader(data, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=4, pin_memory=True)

    diffusion = Diffusion(T=30)
    diffusion.unet.to(dev)

    for epoch in range(num_epochs):
        for i, (x, _) in enumerate(data_loader):
            x = norm_img_batch(x)
            x = x.to(dev)
            diffusion.train(x)

    example = inv_norm_batch(diffusion.generate()[0, :, :, :].cpu())
    example = example.permute(1, 2, 0).detach().clamp(0, 1).numpy()
    plt.imsave(f"diffusion\\_Diff_out_{epoch}_{i}.png", example)

    torch.save(diffusion.unet.state_dict(), f"unet_epoch_{epoch}.pt")
