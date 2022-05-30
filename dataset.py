from torchvision import datasets
import torch
import os
from PIL import Image
from torch.utils.data.dataset import Dataset
import random
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# transform = transforms.Compose(transforms.Resize([105,105]))

def get_random_partition(seed=42):
    digit_indices = [x for x in range(200)]
    random.seed(seed)
    random.shuffle(digit_indices)
    train = digit_indices[:]
    validate = digit_indices[:]
    test = digit_indices[:]
    return train, validate, test


def get_loaders(args):
    train, validate, test = get_random_partition(args.seed)
    transform1 = transforms.Resize(105)
    # transform2 = transforms.Compose(
    #     [transforms.Resize(105),
    #      transforms.RandomCrop(105,padding = 2 ,pad_if_needed = True,fill = 0,padding_mode ='constant'),
    #      transforms.RandomHorizontalFlip(p=0.5) # 表示进行左右的翻转
    #      ])

    dset_train = OmniglotPair("data_train_val", train, transform=transform1, train=True)
    if args.cuda:
        train_loader = DataLoader(dset_train,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=1,
                                  pin_memory=True)
    else:
        train_loader = DataLoader(dset_train,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=4,
                                  pin_memory=False)

    dset_validate = OmniglotPair("data_test", validate, transform=transform1, train=False)
    if args.cuda:
        validate_loader = DataLoader(dset_validate,
                                     batch_size=args.test_batch_size,
                                     shuffle=True,
                                     num_workers=1,
                                     pin_memory=True)
    else:
        validate_loader = DataLoader(dset_validate,
                                     batch_size=args.test_batch_size,
                                     shuffle=True,
                                     num_workers=4,
                                     pin_memory=False)

    dset_test = Omniglot("data_test", test, transform=transform1)

    return train_loader, validate_loader, dset_test


class Omniglot(object):
    def __init__(self, path, choosen_classes=None, transform=None):
        self.path = path
        self.data = []
        self.transform = transform
        class_num = 0
        folders = [f for f in os.listdir(path) if not f[0] == '.']
        for directory in folders:
            folders2 = [f for f in os.listdir(
                os.path.join(path, directory)) if not f[0] == '.']
            for subdirectory in folders2:
                if (choosen_classes is None) or class_num in choosen_classes:
                    folders3 = [f for f in os.listdir(os.path.join(
                        path, directory, subdirectory)) if not f[0] == '.']
                    for file in folders3:
                        self.data.append(
                            (os.path.join(path, directory, subdirectory, file), class_num))
                class_num += 1

    def __getitem__(self, index):
        label = self.data[index][0]
        img = Image.open(self.data[index][0])
        img = img.convert('L')
        if self.transform is not None:
            img = self.transform(img)
        img_array = np.array(img)
        img_array = img_array / 255.0
        img_array = img_array.reshape(
            (1, 1, img_array.shape[0], img_array.shape[1]))

        return torch.from_numpy(img_array).float(), label

    def __len__(self):
        return len(self.data)


class OmniglotPair(object):
    def __init__(self, path, choosen_classes=None, transform=None, train=True):
        random.seed(42)
        self.path = path
        self.data = []
        self.transform = transform
        self.train = train
        class_num = 0
        folders = [f for f in os.listdir(path) if not f[0] == '.']
        for directory in folders:
            folders2 = [f for f in os.listdir(
                os.path.join(path, directory)) if not f[0] == '.']
            for subdirectory in folders2:
                # character is in the choosen classes
                if (choosen_classes is None) or class_num in choosen_classes:
                    folders3 = [f for f in os.listdir(os.path.join(
                        path, directory, subdirectory)) if not f[0] == '.']
                    temp_array = []
                    for file in folders3:
                        temp_array.append(os.path.join(
                            path, directory, subdirectory, file))
                    self.data.append(temp_array)
                class_num += 1

        self.sub_index = []
        if(self.train):
            for i in range(14):
                for j in range(i, 15):
                    self.sub_index.append((i, j))
        else:
            for i in range(4):
                for j in range(i, 5):
                    self.sub_index.append((i, j))

    def __getitem__(self, index):
        # if(index >= 14 * 15 * 200):
        #     return
        if(self.train):
            class_num = index // 210
            img_number = index % 210
            if (img_number % 2 == 0):
                img1 = self.load_img(
                    self.data[class_num][self.sub_index[img_number // 2][0]])
                img2 = self.load_img(
                    self.data[class_num][self.sub_index[img_number // 2][1]])
                return img1, img2, torch.from_numpy(np.array([1])).float()
            else:
                img1 = self.load_img(
                self.data[class_num][self.sub_index[img_number // 2][0]])
                inc = random.randrange(1, len(self.data))
                dn = (class_num + inc) % len(self.data)
                di = random.randrange(0, len(self.data[dn]))
                img2 = self.load_img(self.data[dn][di])
                return img1, img2, torch.from_numpy(np.array([-1])).float()
        else:
            class_num = index // 20
            img_number = index % 20
            if (img_number % 2 == 0):
                img1 = self.load_img(
                    self.data[class_num][self.sub_index[img_number // 2][0]])
                img2 = self.load_img(
                    self.data[class_num][self.sub_index[img_number // 2][1]])
                return img1, img2, torch.from_numpy(np.array([1])).float()
            else:
                img1 = self.load_img(
                self.data[class_num][self.sub_index[img_number // 2][0]])
                inc = random.randrange(1, len(self.data))
                dn = (class_num + inc) % len(self.data)
                di = random.randrange(0, len(self.data[dn]))
                img2 = self.load_img(self.data[dn][di])
                return img1, img2, torch.from_numpy(np.array([-1])).float()

    def load_img(self, path):
        img = Image.open(path)
        img = img.convert('L')
        if self.transform is not None:
            img = self.transform(img)
        img_array = np.array(img)
        img_array = img_array / 255.0
        img_array = img_array.reshape(
            (1, img_array.shape[0], img_array.shape[1]))
        return torch.from_numpy(img_array).float()

    def __len__(self):
        if(self.train):
            return 15 * 14 * len(self.data)
        else:
            return 5 * 4 * len(self.data)