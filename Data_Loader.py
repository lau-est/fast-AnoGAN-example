from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import glob
import os

class My_Data(Dataset):
    def __init__(self, root, transforms_ = None, mode = 'training'):
        self.transform = transforms.Compose(transforms_)
        
        if mode == "training":
            self.files = root

        else:
            self.files     = sorted(glob.glob(os.path.join(root, '%s' % mode) + '/*.*'))


    def imread(self, path):
        img  = Image.open(path)
        img1 = ImageOps.grayscale(img)
        return img1

    def __getitem__(self, index):
        item = self.transform(  self.imread( self.files[ index % len(self.files) ] )  )
        return item

    def __len__(self):
        return (len(self.files))