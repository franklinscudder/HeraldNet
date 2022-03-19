from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.cuda as cuda

from torch.utils.data import Subset

from autoconvo.convo import make_convolutions

from matplotlib import pyplot as plt

import time

IMAGE_X = 60
IMAGE_Y = 80

def lambd(x):           # weird workaround cause apparently lambdas aint cool no more 
    return x[:, :IMAGE_Y, :]

transform_comp = transforms.Compose([
    transforms.Resize([int(1.1 * IMAGE_Y), IMAGE_X], interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Lambda(lambd)
])

class ReluAndUpsample(nn.Module):
    def __init__(self, out_size):
        super(ReluAndUpsample, self).__init__()
        self.relu = nn.LeakyReLU()
        self.out_size = out_size
    
    def forward(self, inp):
        x = self.relu(inp)
        x = F.interpolate(x, size=self.out_size, mode='bilinear', align_corners=True)
        return x
        
class BypassDeconv2d(nn.Module):
    def __init__(self, in_features, out_features, kernel, stride=1, padding=0, bias=True):
        super(BypassDeconv2d, self).__init__()
        self.conv = nn.ConvTranspose2d(in_features, out_features, kernel, stride=stride, padding=padding, bias=bias)
        self.fac = nn.Parameter(Tensor([0.5]))
        
    def forward(self, inp):
        
        self.fac = self.fac.clamp(0, 1) ### ?
        
        x1 = self.conv(inp)
        x2 = F.interpolate(inp, size=x1.shape(), mode='bilinear', align_corners=True)
        y = self.fac * x1 + (1 - self.fac) * x2
        return y
        
        

class LinearAndPermute(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearAndPermute, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, inp):
        x = inp.permute(0, 2, 3, 1)
        x = self.linear(x)
        x = x.permute(0, 3, 1, 2)
        return x


class G(nn.Module):
    def __init__(self, H, W):
        super(G, self).__init__()
        
        self.H = H
        self.W = W
        
        self.layers = nn.Sequential(
                    LinearAndPermute(1000, 125),
                    nn.LazyConvTranspose2d(125, 3, stride=1, padding=1, bias=False),
                    nn.ReLU(),
                    nn.BatchNorm2d(125),
                    nn.LazyConvTranspose2d(75, 6, stride=3, padding=1, bias=False),
                    nn.ReLU(),
                    nn.BatchNorm2d(75),
                    nn.LazyConvTranspose2d(10, 4, stride=[3,2], padding=1, bias=False),
                    nn.ReLU(),
                    nn.BatchNorm2d(10),
                    #nn.LazyConvTranspose2d(10, 5, stride=[3,4], padding=0, bias=False),
                    #nn.ReLU(),
                    #nn.BatchNorm2d(10),
                    # nn.LazyConvTranspose2d(10, 6, stride=2, padding=1, bias=False),
                    # nn.ReLU(),
                    # nn.BatchNorm2d(10),
                    # nn.LazyConvTranspose2d(3, 3, stride=1, padding=1, bias=False),
                    # nn.ReLU(),
                    nn.Conv2d(10, 3, 3, stride=1, padding=1, bias=False),
                    nn.Sigmoid()
                    )
                    
    def forward(self, B, dev): 
        latent = torch.randn((B, 1000, 12, 12), device=dev)
        out = self.layers(latent)
        #print(out.shape)
        offH = (out.shape[2] - self.H) // 2
        offW = (out.shape[3] - self.W) // 2
        out = out[:,:,offH:self.H+offH, offW:self.W+offW]
        #print(list(out.shape))
        assert list(out.shape)[2:] == [self.H, self.W]
        
        return torch.clamp(out, 0, 1)
        
class D(nn.Module):
    def __init__(self, H, W):
        super(D, self).__init__()
        
        self.H = H
        self.W = W
        
        #self.denoise = nn.Conv2d(3, 3, 3, padding=1, stride=1)
        #self.conv = make_convolutions((3, H, W), (250, 1, 1), 3)
        self.layers = nn.Sequential(
                    #self.denoise,
                    
                    nn.Conv2d(3, 10, 3, stride=1, padding=1, bias=False),
                    nn.ReLU(),
                    nn.BatchNorm2d(10),
                    nn.Conv2d(10, 75, 4, stride=[3,2], padding=1, bias=False),
                    nn.ReLU(),
                    nn.BatchNorm2d(75),
                    nn.Conv2d(75, 125, 6, stride=3, padding=1, bias=False),
                    nn.ReLU(),
                    nn.BatchNorm2d(125),
                    nn.Conv2d(125, 250, 4, stride=2, padding=1, bias=False),
                    nn.ReLU(),
                    nn.BatchNorm2d(250),
                    nn.Flatten(),
                    nn.Linear(4000,2),
                    nn.LeakyReLU(),
                    # nn.Linear(50, 2),
                    # nn.LeakyReLU(),
                    nn.Softmax(dim=1)
                    )
        #print(self.conv)
         
    def forward(self, image):
        return self.layers(image)
        
def to_showable(img):
    return img.data.to("cpu").permute(1,2,0).numpy().astype(float)
                    
        
if __name__ == "__main__":
    #plt.imshow(to_showable(data[0][0]))
    #plt.show()
    #quit()
    
    has_cuda = True if torch.cuda.is_available() else False
    dev = "cuda:0" if has_cuda else "cpu"

    data_dir =  r"C:\Users\Tom\Documents\GitHub\HeraldryData\data\data"
    
    batch_size = 100
    num_epochs = 3000
    
    #data = Subset(ImageFolder(data_dir, transform_comp), [2,3])
    data = ImageFolder(data_dir, transform_comp)
    data_loader = DataLoader(data, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=4, pin_memory=True)
    
    real_labels = torch.tensor([[1.0, 0.0]]*batch_size).to(dev)
    fake_labels = torch.tensor([[0.0, 1.0]]*batch_size).to(dev)
    double_real_labels = torch.tensor([[1.0, 0.0]]*(2*batch_size)).to(dev)

    d = D(IMAGE_Y, IMAGE_X)
    g = G(IMAGE_Y, IMAGE_X)

    if has_cuda:
        print("CUDA available")
        d.cuda()
        g.cuda()

    d_loss = nn.BCELoss().to(dev)
    g_loss = nn.BCELoss().to(dev)

    d_opt = optim.AdamW(d.parameters(), 0.000001)
    g_opt = optim.AdamW(g.parameters(), 0.000001)
    
    d_scheduler = optim.lr_scheduler.MultiStepLR(d_opt, [500], gamma=0.1, verbose=True)
    g_scheduler = optim.lr_scheduler.MultiStepLR(g_opt, [500], gamma=0.1, verbose=True)

    plt.ion()

    g_losses = []
    d_losses = []
    
    t0 = time.time()
    fig = False
    for E in range(1, num_epochs + 1):
        #train discrim
        
        b = 0
        
        for reals, _ in data_loader:
            b += 1
            print(f"Processing batch {b} of epoch {E}")
            g.eval()
            d.train()
            fakes = g(batch_size, dev)
            reals = reals.to(dev)
            
            real_discrims = d(reals)
            fake_discrims = d(fakes)
            
            loss = (d_loss(real_discrims, real_labels) + d_loss(fake_discrims, fake_labels))/2
            #print(real_discrims, real_labels)
            #print(fake_discrims, fake_labels)
            loss.backward()
            d_losses.append(loss.item())
            d_opt.step()
            
            d_opt.zero_grad()
            d.zero_grad()
            
            #train gen
            g.train()
            d.eval()
            
            fakes = g(batch_size * 2, dev)
            
            if b == 1 and E%1 == 0:
                if fig:
                    for ax in ax0, ax1, ax2, ax3:
                        ax.clear()
                
                plt.pause(0.001)
                if not fig:
                    fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4)
                    fig.set_size_inches(14, 6, forward=True)
                    ax3.set_ylim([0, 3])
                    ax2.set_ylim([0, 3])
                    
                plt.imsave(r"C:/Users/Tom/Documents/GitHub/HeraldNet/outputs/out_" + str(E) + ".png", to_showable(fakes[0, :, :, :]))
                ax1.imshow(to_showable(fakes[0, :, :, :]))

                ax0.imshow(to_showable(reals[0, :, :, :]))
                
                #ax2.set_yscale('log')
                
                ax2.plot(g_losses)
                ax2.title.set_text("Gen")
                
                #ax3.set_yscale('log')
                
                ax3.plot(d_losses)
                ax3.title.set_text("Dis")
                fig.canvas.draw()
            
            if fig:
                fig.canvas.flush_events()   
                plt.pause(0.001)
              
            if b % 1 == 0:       ## imbalance training here
                fake_discrims = d(fakes)
                loss = g_loss(fake_discrims, double_real_labels)
                loss.backward()
                g_losses.append(loss.item())
                g_opt.step()
            
            g_opt.zero_grad()
            g.zero_grad()
            
            t = time.time()
            batches_done = b + (E-1)*batch_size
            tpb = (t-t0) / batches_done
            print(f"Time per batch is {tpb} s. \nTime per sample is: {tpb/batch_size} s.")
            
        g_scheduler.step()
        d_scheduler.step()