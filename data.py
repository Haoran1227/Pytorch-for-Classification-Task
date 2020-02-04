from torch.utils.data import Dataset
import torch
import pandas as pd
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
from sklearn.model_selection import train_test_split
import torchvision as tv

# For rgb
train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    def __init__(self, mode, csv_file, split_parameter, transform=tv.transforms.Compose([tv.transforms.ToTensor()])):
        """
        Args:
             mode (string): Mode flag which can be either 'val' or 'train'.
             csv_file (string): Path to the csv file.
             split_parameter (float): 0< param < 1. Parameter which controls the ratio of validation data in all data.
             transform (torchvision.transforms.Compose): defines transform methods to convert image to tensor and do resize.
        """
        data_frame = pd.read_csv(csv_file, sep=';')         # read csv file
        img_name = list(data_frame.iloc[:, 0])              # The first column denotes image name
        self.all_label = np.array(data_frame.iloc[:, 2:])   # read label and convert it to ndarray. '2' is because '1' is poly_wafer label, it's ignored in this project
        # Separate dataset into training dataset and validation dataset randomly.
        if mode == 'train':
            # img_train, img_test, label_train, label_test = train_test_split(img_name, label, test_size, random_state)
            self.img_name, _, self.label, _ = train_test_split(img_name, self.all_label, test_size=split_parameter,
                                                               random_state=10)     # random_state means random seed
        if mode == 'val':
            _, self.img_name, _, self.label = train_test_split(img_name, self.all_label, test_size=split_parameter,
                                                               random_state=10)

        self.transform = transform

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()  # If item is a tensor, convert it to list

        # read image
        image = imread('./' + self.img_name[item])  # imread() return type of ndarray
        image = gray2rgb(image)  # convert the grayscale image to rgb
        image = self.transform(image)  # implement transform, return type of tensor

        # read label ('crack', 'inactive')
        label = torch.from_numpy(self.label[item]).float()  # convert ndarray to tensor with float dtype

        # construct sample as a tuple type, it contains two tensor type elements (image and label)
        sample = (image, label)
        return sample

    def pos_weight(self):
        # read labels. "crack" and "inactive"
        crack_label = self.all_label[:, 0]
        inactive_label = self.all_label[:, 1]

        # calculate the ratio of negative to positive samples
        crack_weight = np.sum(1 - crack_label) / np.sum(crack_label)
        inactive_weight = np.sum(1 - inactive_label) / np.sum(inactive_label)

        # return a torch.tensor of weights
        weight_tensor = torch.tensor([crack_weight, inactive_weight])
        return weight_tensor


# global parameter
csv_path = './train.csv'
split_num = 0.25  # You can modify the split of dataset by changing this parameter


def get_train_dataset():
    transform = tv.transforms.Compose([tv.transforms.ToPILImage(),
                                       tv.transforms.ToTensor(),
                                       tv.transforms.Normalize(mean=train_mean, std=train_std)])
    train_dataset = ChallengeDataset('train', csv_path, split_num, transform)
    return train_dataset


# this needs to return a dataset *without* data augmentation!
def get_validation_dataset():
    transform = tv.transforms.Compose([tv.transforms.ToPILImage(),
                                       tv.transforms.ToTensor(),
                                       tv.transforms.Normalize(mean=train_mean, std=train_std)])
    val_dataset = ChallengeDataset('val', csv_path, split_num, transform)
    return val_dataset
