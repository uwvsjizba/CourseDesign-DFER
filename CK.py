from __future__ import print_function
from PIL import Image
import numpy as np
import h5py
import torch.utils.data as data


class CK(data.Dataset):
    """`CK+ Dataset.

    Args:
        train（bool，可选）：如果为 True，则从训练集创建数据集，否则
            从测试集创建。
        transform（可调用，可选）：接受 PIL 图像的函数/转换
            并返回转换后的版本。例如，“转换。随机裁剪''

        数据中有 135,177,75,207,84,249,54 图像
        数据中有 135,177,75,207,84,249,54 图像
        我们选择 123,159,66,186,75,225,48 张图像进行训练
        我们选择 12、8、9、21、9、24、6 张图像进行测试
        拆分按折叠数排序
    """

    def __init__(self, split='Training', fold = 1, transform=None):
        self.transform = transform
        self.split = split  # training set or test set
        self.fold = fold # the k-fold cross validation
        self.data = h5py.File('./data/CK+/CK_data.h5', 'r', driver='core')

        number = len(self.data['data_label']) #981
        sum_number = [0,135,312,387,594,678,927,981] # the sum of class number
        test_number = [12,18,9,21,9,24,6] # the number of each class

        test_index = []
        train_index = []

        for j in range(len(test_number)):
            for k in range(test_number[j]):
                if self.fold != 10: #the last fold start from the last element
                    test_index.append(sum_number[j]+(self.fold-1)*test_number[j]+k)
                else:
                    test_index.append(sum_number[j+1]-1-k)

        for i in range(number):
            if i not in test_index:
                train_index.append(i)

        print(len(train_index),len(test_index))

        # now load the picked numpy arrays
        if self.split == 'Training':
            self.train_data = []
            self.train_labels = []
            for ind in range(len(train_index)):
                self.train_data.append(self.data['data_pixel'][train_index[ind]])
                self.train_labels.append(self.data['data_label'][train_index[ind]])

        elif self.split == 'Testing':
            self.test_data = []
            self.test_labels = []
            for ind in range(len(test_index)):
                self.test_data.append(self.data['data_pixel'][test_index[ind]])
                self.test_labels.append(self.data['data_label'][test_index[ind]])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.split == 'Training':
            img, target = self.train_data[index], self.train_labels[index]
        elif self.split == 'Testing':
            img, target = self.test_data[index], self.test_labels[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = img[:, :, np.newaxis]
        img = np.concatenate((img, img, img), axis=2)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        if self.split == 'Training':
            return len(self.train_data)
        elif self.split == 'Testing':
            return len(self.test_data)

