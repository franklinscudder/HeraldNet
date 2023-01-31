import torch.nn as nn
import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision.utils as vutils
import torch.nn.init as init

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

IMAGE_X = 64
IMAGE_Y = 64


class MinibatchDiscrimination(nn.Module):
    def __init__(self, in_features, out_features, kernel_dims, mean=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_dims = kernel_dims
        self.mean = mean
        self.T = nn.Parameter(torch.Tensor(in_features, out_features, kernel_dims))
        init.normal_(self.T, 0, 1)

    def forward(self, x):
        # x is NxA
        # T is AxBxC
        matrices = x.mm(self.T.view(self.in_features, -1))
        matrices = matrices.view(-1, self.out_features, self.kernel_dims)

        M = matrices.unsqueeze(0)  # 1xNxBxC
        M_T = M.permute(1, 0, 2, 3)  # Nx1xBxC
        norm = torch.abs(M - M_T).sum(3)  # NxNxB
        expnorm = torch.exp(-norm)
        o_b = (expnorm.sum(0) - 1)   # NxB, subtract self distance
        if self.mean:
            o_b /= x.size(0) - 1

        x = torch.cat([x, o_b], 1)  # Nx(A+B)
        return x


transform_comp = transforms.Compose([
    transforms.Resize([IMAGE_X, IMAGE_Y], interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ColorJitter(brightness=.3, hue=.1),
    transforms.ToTensor(),
])


class ShapePrinter(nn.Module):
    def __init__(self, name="ShapePrinter", once=True):
        super(ShapePrinter, self).__init__()

        if once:
            self.done = False

        self.once = once
        self.name = name

    def forward(self, x):
        if not self.done:
            print(f"Shape printer '{self.name}' -> {list(x.shape)}")
        if self.once:
            self.done = True

        return x


# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.nc = 3
        self.ndf = 16

        self.main = nn.Sequential(
            # input nc * 64 * 64
            nn.Conv2d(self.nc, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # second ndf * 32 * 32
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # third 2ndf * 16 * 16
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # fourth 4ndf * 8 * 8
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # output 8ndf * 4 * 4
            nn.Conv2d(self.ndf * 8, self.ndf * 16, 4, 2, 1, bias=False),

            ShapePrinter(name="Before Flatten"),

            # Batch discrimination 16ndf * 2 * 2
            nn.Flatten(),
            MinibatchDiscrimination(16 * 2 * 2 * self.ndf, self.ndf * 2 * 2, self.ndf * 2 * 2, mean=True),
            ShapePrinter(name="After Batch Discrim"),
            nn.BatchNorm1d(1088),
            nn.Linear(1088, 64),
            nn.Sigmoid(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


# Generator Network
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.nz = 1000
        self.ngf = 10
        self.nc = 3
        self.main = nn.Sequential(
            # input Z; First
            nn.ConvTranspose2d(self.nz, self.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),

            # second
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),

            # third
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),

            # fourth
            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),

            # output nc * 64 * 64
            nn.ConvTranspose2d(self.ngf, self.nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        out = self.main(input)
        # scale to (+1, -1) ish
        out = 2 * out - 1
        # add some noise
        out = out + 0.05 * torch.randn_like(out)
        
        return out


if __name__ == '__main__':
    plt.ion()

    has_cuda = True if torch.cuda.is_available() else False
    dev = "cuda:0" if has_cuda else "cpu"

    print("Using device:", dev)

    data_dir = r"C:\Users\Tom\Documents\GitHub\HeraldryData\data\data"
    batch_size = 128
    num_epochs = 3000
    data = ImageFolder(data_dir, transform_comp)
    data_loader = DataLoader(data, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=4, pin_memory=True)

    netG = Generator().to(dev)
    netD = Discriminator().to(dev)

    G_losses = []
    D_losses = []

    img_list = []

    real_label = 0.95
    fake_label = 0.00

    nz = 1000

    fixed_noise = torch.randn(64, nz, 1, 1, device=dev)

    criterion = nn.BCELoss()
    optimizerG = optim.Adam(netG.parameters(), lr=0.0001)
    optimizerD = optim.Adam(netD.parameters(), lr=0.0001)

    for e in range(num_epochs):
        for i, (x, _) in enumerate(data_loader):
            netD.zero_grad()
            # train with real data
            real_data = x.to(dev)
            # scale to (+1, -1)
            real_data = 2 * real_data - 1
            # add some noise
            real_data = real_data + 0.05 * torch.randn_like(real_data)
            # make labels
            batch_size = real_data.size(0)
            labels = torch.full((batch_size,), real_label, device=dev)
            # forward pass real data through D
            real_outputD = netD(real_data).view(-1)
            # calc error on real data
            errD_real = criterion(real_outputD, labels)
            # calc grad
            errD_real.backward()
            D_x = real_outputD.mean().item()
            # train with fake data
            noise = torch.randn(batch_size, nz, 1, 1, device=dev)
            fake_data = netG(noise)
            labels.fill_(fake_label)
            # classify fake
            fake_outputD = netD(fake_data.detach()).view(-1)
            # calc error on fake data
            errD_fake = criterion(fake_outputD, labels)
            # calc grad
            errD_fake.backward()
            D_G_z1 = fake_outputD.mean().item()
            # add all grad and update D
            errD = errD_real + errD_fake
            optimizerD.step()

            ########################################
            ########## Training Generator ##########
            netG.zero_grad()
            # since aim is fooling the netD, labels should be flipped
            labels.fill_(real_label)
            # forward pass with updated netD
            fake_outputD = netD(fake_data).view(-1)
            # calc error
            errG = criterion(fake_outputD, labels)
            # calc grad
            errG.backward()
            D_G_z2 = fake_outputD.mean().item()
            # update G
            optimizerG.step()

            ########################################
            # output training stats
            if (i + 1) % 500 == 0:
                print(f'[{e+1}][{i+1}] Loss_D:{errD.item():.4f} Loss_G:{errG.item():.4f} D(x):{D_x:.4f} D(G(z)):{D_G_z1:.4f}/{D_G_z2:.4f}')
            # for later plot 
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            # generate fake image on fixed noise for comparison
            if (i % 500 == 0):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                    img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                    plt.imshow(img_list[-1].permute(1, 2, 0))
                    plt.imsave("outputs\\oup_{e}_{i}.png", img_list[-1].permute(1, 2, 0).numpy())
                    plt.draw()
                    torch.save(netD.state_dict(), "netD.pt")
                    torch.save(netG.state_dict(), "netG.pt")
            plt.pause(0.001)
