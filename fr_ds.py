import random
import os
from PIL import Image
from torch.utils.data import Dataset
from collections import defaultdict

class Fr_Ds(Dataset):
    def __init__(self, trf, path) -> None:
        super().__init__()
        self.path = path
        self.names_folder_path = os.listdir(self.path)
        self.trf = trf

        self.imgs = []
        self.map_cls = defaultdict(list)
        self.labels = []
        for i in self.names_folder_path:
            for j in os.listdir(self.path + i):
                self.imgs.append(self.path + i + '/' + j)
                self.labels.append(i)
                self.map_cls[i].append(self.path + i + '/' + j)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        temp = self.imgs[index]
        label = self.labels[index]
        anchor_path = temp

        if 'train' in temp:
            key = ''
            for i in self.map_cls: 
                if i in temp: 
                    key = i
                    break
            
            positve_path = random.choice(self.map_cls[key])
            while positve_path == anchor_path: positve_path = random.choice(self.map_cls[key])

            negatve_path = random.choice(self.imgs)
            while key in negatve_path : negatve_path = random.choice(self.imgs)

        return self.trf(Image.open(anchor_path).convert('RGB')), self.trf(Image.open(positve_path).convert('RGB')) if 'train' in temp else False, self.trf(Image.open(negatve_path).convert('RGB')) if 'train' in temp else False, label