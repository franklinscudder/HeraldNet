from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import Tensor, cat

IMAGE_X = 128
IMAGE_Y = 128

if __name__ == "__main__":
    transform_comp = transforms.Compose([transforms.ToTensor(), transforms.Resize([IMAGE_X, IMAGE_Y], interpolation=transforms.InterpolationMode.BILINEAR)])

    data_dir = r"C:\Users\Tom\Documents\GitHub\HeraldryData\data\data"
    data = ImageFolder(data_dir, transform_comp)
    data_loader = DataLoader(data, batch_size=100, num_workers=4, pin_memory=True)

    chans = [Tensor([]), Tensor([]), Tensor([])]

    I = len(data_loader)

    for i, (x, _) in enumerate(data_loader):
        for chan in 0, 1, 2:
            print(round(i / I, 3))
            chans[chan] = cat((chans[chan], (x[:, chan, :, :].flatten())))

    for chan in 0, 1, 2:
        print(f"chan {chan}: mean = {chans[chan].mean()}, var = {chans[chan].var()}")
