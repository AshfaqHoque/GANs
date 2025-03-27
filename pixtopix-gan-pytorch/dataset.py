import torch
import os
import numpy as np
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt 
import config
from PIL import Image
import torchvision.transforms as transforms


class MapDataset(Dataset):
  def __init__(self, root_dir):
    self.high_dir = os.path.join(root_dir, "high")
    self.low_dir = os.path.join(root_dir, "low")
    self.high_images = sorted(os.listdir(self.high_dir))
    self.low_images = sorted(os.listdir(self.low_dir))

  def __len__(self):
    return len(os.listdir(self.high_dir))

  def __getitem__(self, index):
    high_path = os.path.join(self.high_dir, self.high_images[index])
    low_path = os.path.join(self.low_dir, self.low_images[index])
    print(f"High image path: {high_path}")
    print(f"Low image path: {low_path}")
    high_image = np.array(Image.open(high_path))
    low_image = np.array(Image.open(low_path))

    augmentations = config.both_transform(image=high_image, image0=low_image)
    high_image = augmentations["image"]
    low_image = augmentations["image0"]

    high_image = config.transform_only_input(image=high_image)["image"]
    low_image = config.transform_only_mask(image=low_image)["image"]
    high_image = high_image.cuda() if torch.cuda.is_available() else high_image
    low_image = low_image.cuda() if torch.cuda.is_available() else low_image

    return high_image, low_image
  

# path = kagglehub.dataset_download("tanvirnwu/loli-street-low-light-image-enhancement-of-street")
# #print("Dataset Path:", path)
# #print("Contents of the dataset root folder:", os.listdir(path))
# dataset_path = r"C:\Users\Md. Ashfaq Bin Hoque\.cache\kagglehub\datasets\tanvirnwu\loli-street-low-light-image-enhancement-of-street\versions\1"
# loli_dataset_folder = os.path.join(dataset_path, "LoLI-Street Dataset")
# print("Contents of LoLI-Street Dataset:", os.listdir(loli_dataset_folder))
train_path = r"D:\Notebooks\gan\pixtopix-gan-pytorch\LoLI-Street Dataset\Train"
#print(valid_path)
# high_path = os.path.join(train_path, "high")
# print(high_path)

if __name__ == "__main__":
    dataset = MapDataset(train_path)
    loader = DataLoader(dataset, batch_size=5)

    #Convert tensors back to PIL images
    to_pil = transforms.ToPILImage()


    # Visualize the entire batch of images
    for x, y in loader:
        # Denormalize (convert back to range [0, 1] for visualization)
        x = x * 0.5 + 0.5
        y = y * 0.5 + 0.5

        # Number of images in the batch
        batch_size = x.shape[0]

        # Create a subplot with enough space for all images in the batch
        fig, axes = plt.subplots(batch_size, 2, figsize=(10, batch_size * 5))

        # Loop through each image in the batch
        for i in range(batch_size):
            # Convert PyTorch tensors to numpy for Matplotlib
            x_np = x[i].permute(1, 2, 0).cpu().numpy()
            y_np = y[i].permute(1, 2, 0).cpu().numpy()

            # Plot input and target images
            axes[i, 0].imshow(x_np)
            axes[i, 0].set_title(f"Input Image {i+1}")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(y_np)
            axes[i, 1].set_title(f"Target Image {i+1}")
            axes[i, 1].axis("off")

        plt.tight_layout()
        plt.show()
        break  #only 1 batch


