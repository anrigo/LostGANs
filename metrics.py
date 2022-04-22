import pathlib
import numpy as np
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from PIL import Image
import torchvision.transforms as TF
from tqdm import tqdm


IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}


class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        
        img = torch.from_numpy(np.array(img))
        img = torch.permute(img, (2,0,1))

        return img


def get_files(path):
    path = pathlib.Path(path)
    files = sorted([file for ext in IMAGE_EXTENSIONS
                       for file in path.glob('*.{}'.format(ext))])
    
    return files


def get_loader(path, batch_size, num_workers):
    files = get_files(path)

    dataset = ImagePathDataset(files, transforms=TF.Resize((299,299))) # resize to 299x299 as in original paper
    dataloader = torch.utils.data.DataLoader(dataset,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                drop_last=False,
                                                num_workers=num_workers)
    
    return dataloader


device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')

path_real = 'datasets/coco/images/val2017/'
path_fake = 'samples/coco128-30/'

batch_size = 50
num_workers = 10

fid = FrechetInceptionDistance(feature=2048).to(device)
inception = InceptionScore().to(device)


# real images
dataloader = get_loader(path_real, batch_size, num_workers)

for batch in tqdm(dataloader):
    batch = batch.to(device)

    fid.update(batch, real=True)


# fake images
dataloader = get_loader(path_fake, batch_size, num_workers)

for batch in tqdm(dataloader):
    batch = batch.to(device)

    fid.update(batch, real=False)
    inception.update(batch)

print(f'FID: {fid.compute()}') # 92.0442 COCO 128 30 eps
print(f'IS: {inception.compute()[0]}') # 11.9291 COCO 128 30 eps