import torch
from torch import nn, optim
import torch.nn.functional as F
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import numpy as np

class ImageDatasetV2(Dataset):
    def __init__(self, 
                 acts_txt, image_dir, 
                 obs_txt=None, 
                 transform=transforms.ToTensor(), index=None):
        """
        Args:
            image_dir (str): Path to the directory containing the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.curr_image = sorted([os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith('_curr.png')])
        self.next_image = sorted([os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith('_next.png')])
        self.obs = None
        self.prev = len([fname for fname in self.curr_image if fname.split("/")[-1].startswith('00000_')])
        if obs_txt is not None:
            self.obs = torch.from_numpy(np.loadtxt(obs_txt))
        self.acts = torch.from_numpy(np.loadtxt(acts_txt))
        self.act_min = torch.min(self.acts, axis=0).values
        self.act_max = torch.max(self.acts, axis=0).values
        self.acts = (self.acts - self.act_min) / (self.act_max - self.act_min)  # Normalize to [0, 1] for consistency
        if index is not None:
            self.next_image = [self.next_image[i] for i in index]
            if self.obs is not None:
                self.obs = self.obs[index]
            self.acts = [self.acts[i] for i in index] # [self.prev*index[0]: min(len(self.acts), self.prev*(index[-1]+1))]
            index = np.arange(self.prev*index[0], self.prev*(index[-1]+1))
            self.curr_image = [self.curr_image[i] for i in index]
        self.transform = transform

    def __len__(self):
        # Returns the number of images in the dataset
        # return len(self.curr_image) // self.prev
        return len(self.next_image)

    def __getitem__(self, idx):
        # Loads an image and applies transformations
        curr_fname = self.curr_image[idx*self.prev: (idx+1)*self.prev]
        next_fname = [self.next_image[idx]]
        imgs_curr = [Image.open(fname).convert('RGB') for fname in curr_fname]
        imgs_next = [Image.open(fname).convert('RGB') for fname in next_fname]

        if self.transform:
            imgs_curr = [self.transform(img) for img in imgs_curr]  # Apply any transformations (e.g., resize, normalize)
            imgs_next = [self.transform(img) for img in imgs_next]  # Apply any transformations (e.g., resize, normalize)
            # acts = self.transform(np.array([self.acts[idx]])[:, None])
            acts = torch.tensor(self.acts[idx:(idx+1)], dtype=torch.float32).unsqueeze(-1)
        
        sample = {'imgs_curr': torch.stack(imgs_curr, dim=0), 
                  'imgs_next': torch.stack(imgs_next, dim=0), 
                  'acts': acts,
                  'fnames': curr_fname + next_fname,
                  'idx': idx}
        if self.obs is not None:
            sample['obs'] = self.obs[idx]
        return sample

if __name__ == "__main__":
    dataset = ImageDatasetV2(image_dir=os.getcwd() + "/data/cartpole-v1")
    print("len dataset", len(dataset))
    print("dataset[0]", dataset[0].shape)