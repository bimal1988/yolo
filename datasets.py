import matplotlib.pyplot as plt
import os
import torch
from torch.utils.data import Dataset
# from PIL import Image
import numpy as np
# from skimage.transform import resize
import cv2


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
        Image should be converted to torch.*Tensor before calling this method.
        Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels,
        this transform will normalize each channel of the input Tensor i.e.
        ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """
    def __init__(self, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        tensor, labels = sample['image'], sample['labels']
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
        return {'image': tensor, 'labels': labels}


class Resize(object):
    """Resizes the image to a squared image(width=height).

    Args:
        output_size (int): Desired output size. If input image is not.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, int)
        self.output_size = output_size

    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']

        h, w = image.shape[:2]
        delta = np.abs(h - w)
        pad1, pad2 = delta // 2, delta - delta // 2
        (pad_left, pad_right) = (pad1, pad2) if h > w else (0, 0)
        (pad_top, pad_bottom) = (pad1, pad2) if w > h else (0, 0)
        padding = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
        image = np.pad(image, padding, 'constant', constant_values=0)
        image = cv2.resize(image, (self.output_size, self.output_size))

        # Get new height and width of image
        h_new, w_new = image.shape[:2]
        # Extract coordinates for unpadded + unscaled image
        x_min = (labels[:, 1] - labels[:, 3]/2) * w
        y_min = (labels[:, 2] - labels[:, 4]/2) * h
        x_max = (labels[:, 1] + labels[:, 3]/2) * w
        y_max = (labels[:, 2] + labels[:, 4]/2) * h

        # Adjust for added padding
        x_min += pad_left
        y_min += pad_top
        x_max += pad_left
        y_max += pad_top

        # Calculate ratios from new height and width
        labels[:, 1] = ((x_min + x_max) / 2) / w_new
        labels[:, 2] = ((y_min + y_max) / 2) / h_new
        labels[:, 3] *= w / w_new
        labels[:, 4] *= h / h_new

        return {'image': image, 'labels': labels}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']

        # Swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'labels': torch.from_numpy(labels)}


class Yolo(Dataset):
    """
    Yolo Dataset Structure
    ----------------------
    data----|--obj.data (metadata)
    (root)  |--obj.names (name of classes)
            |--obj--|--image1.jpg
            |       |--labels1.txt
            |       |--....
            |--train.txt (list of images for training)
            |--test.txt (list of images for testing)
    """

    def __init__(self, root, train=True, transform=None):
        self.root = os.path.expanduser(root)
        self.train = train  # Train / Test Dataset
        self.transform = transform
        datafile_name = 'train.txt' if train else 'test.txt'
        datafile_path = os.path.join(self.root, datafile_name)
        self.data = []
        if os.path.exists(datafile_path):
            with open(datafile_path) as datafile:
                self.data = datafile.readlines()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = self.data[index].strip()
        image = cv2.imread(img_path)
        # cv2 images are BGR, convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        anno_path = img_path.replace('.jpg', '.txt')
        labels = np.loadtxt(anno_path).reshape(-1, 5)

        item = {'image': image,
                'labels': labels}

        if self.transform:
            item = self.transform(item)

        return item


yolo = Yolo('data')
item = yolo.__getitem__(150)
tsfm = Resize(450)
transformed_sample = tsfm(item)
img = transformed_sample['image']
im = plt.imshow(img)
im.figure.show()


# class YOLODataset(Dataset):
#     def __init__(self, path, img_size=416):
#         with open(path) as f:
#             self.img_files = f.readlines()
#         self.label_files = [path.replace('images', 'labels').replace(
#             '.png', '.txt').replace('.jpg', '.txt') for path in self.img_files]
#         self.img_shape = (img_size, img_size)
#         self.max_objects = 50

#     def __getitem__(self, index):
#         img_path = self.img_files[index].strip()
#         img = np.array(Image.open(img_path))

#         # Handles images with less than three channels
#         while len(img.shape) != 3:
#             index += 1
#             img_path = self.img_files[index].strip()
#             img = np.array(Image.open(img_path))

#         h, w, _ = img.shape
#         dim_diff = np.abs(h - w)
#         # Upper (left) and lower (right) padding
#         pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
#         # Determine padding
#         pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else (
#             (0, 0), (pad1, pad2), (0, 0))
#         # Add padding
#         input_img = np.pad(img, pad, 'constant', constant_values=128) / 255.
#         padded_h, padded_w, _ = input_img.shape
#         # Resize and normalize
#         input_img = resize(input_img, (*self.img_shape, 3), mode='reflect')
#         # Channels-first
#         input_img = np.transpose(input_img, (2, 0, 1))
#         # As pytorch tensor
#         input_img = torch.from_numpy(input_img).float()

#         label_path = self.label_files[index].strip()
#         labels = None
#         if os.path.exists(label_path):
#             labels = np.loadtxt(label_path).reshape(-1, 5)
#             # Extract coordinates for unpadded + unscaled image
#             x1 = w * (labels[:, 1] - labels[:, 3]/2)
#             y1 = h * (labels[:, 2] - labels[:, 4]/2)
#             x2 = w * (labels[:, 1] + labels[:, 3]/2)
#             y2 = h * (labels[:, 2] + labels[:, 4]/2)
#             # Adjust for added padding
#             x1 += pad[1][0]
#             y1 += pad[0][0]
#             x2 += pad[1][0]
#             y2 += pad[0][0]
#             # Calculate ratios from coordinates
#             labels[:, 1] = ((x1 + x2) / 2) / padded_w
#             labels[:, 2] = ((y1 + y2) / 2) / padded_h
#             labels[:, 3] *= w / padded_w
#             labels[:, 4] *= h / padded_h
#         # Fill matrix
#         filled_labels = np.zeros((self.max_objects, 5))
#         if labels is not None:
#             filled_labels[range(len(labels))[:self.max_objects]
#                           ] = labels[:self.max_objects]
#         filled_labels = torch.from_numpy(filled_labels)

#         return img_path, input_img, filled_labels

#     def __len__(self):
#         return len(self.img_files)
