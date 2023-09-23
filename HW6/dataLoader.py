import skimage
from skimage.color import rgba2rgb
import cv2
from os import listdir
from os.path import join, isfile
import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm

class DataLoader():
    """
     This class is a constructor for the dataset. It can be used for
     loading images with labels.
      
     :param path='dataset/': path to the dataset directory
     :param train_split=0.95: Percentile for the train-test split.
    """
    def __init__(self, path='dataset/', train_split=0.95):
        self.path = path
        self.images_path = join(path, 'images')
        self.labels_path = join(path, 'labels')
        image_list = listdir(self.images_path)
        print(f"There are {len(image_list)} images in dataset")
        self.train_image_list = image_list[:int(len(image_list) * train_split)]
        self.test_image_list = image_list[int(len(image_list) * train_split):]
        print(f"{len(self.train_image_list)} images for train set and {len(self.test_image_list)} for test set")
        
        with open(join(self.path, 'classes.txt')) as file:
            lines = [line.rstrip() for line in file]
        self.classes = {i+1: lines[i] for i in range(len(lines))}
        self.classes[0] = 'background'
        print(f"Dataset classes {self.classes}")
    
    """
     Visualize the image and boxes of the given data_dict.
     
     :param data_dict: Dictionary of data to be visualized.
    """
    def visualize(self, data_dict):
        image = data_dict['image']
        classes = data_dict['classes']
        bboxes = data_dict['bboxes']
        for i in range(len(classes)):
            x1,y1,x2,y2 = bboxes[i,:]
            image = cv2.rectangle(image, (x1, y1), (x2, y2), (36,255,12), 1)
            cv2.putText(image, self.classes[int(classes[i])], (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        plt.imshow(image)
        plt.show()

    """
     Returns a generator of all training data.
    """
    def getAllTrainData(self):
        for i in range(len(self.train_image_list)):
            data_dict = self.loadTrainImageAndLabel(i)
            yield data_dict

    """
     Returns a generator of all the images in the test set.
    """
    def getAllTestData(self):
        for i in range(len(self.test_image_list)):
            image_name = self.test_image_list[i]
            yield rgba2rgb(skimage.io.imread(join(self.images_path, image_name)))

    """
     Load the image and labels of a training image and returns a dictionary of images and labels.
     
     :param idx: Index of the training image.
     :param vis=False: If True the data_dict will be visoulized.
    """
    def loadTrainImageAndLabel(self, idx, vis=False):
        image_name = self.train_image_list[idx]
        image = rgba2rgb(skimage.io.imread(join(self.images_path, image_name)))
        h, w = image.shape[:2]
        with open(join(self.labels_path,image_name.replace('png','txt'))) as file:
            classAndBbox = np.array([line.rstrip().split(" ") for line in file])
        classes, bboxes = classAndBbox[:,0], classAndBbox[:,1:].astype(float)
        """
        Important note
        This dataset is in the 'Yolo dataset' standard format. In this format horizental axis is
        X, and the vertical axis is Y (contrary to NumPy array indexing).
        The origin of coordinates is the top-left corner of the image(just like NumPy array indexing)
        """
        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * w
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * h
        bboxes[:, 0] = bboxes[:, 0] - (bboxes[:, 2] / 2)
        bboxes[:, 1] = bboxes[:, 1] - (bboxes[:, 3] / 2)
        bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
        bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]
        
        data_dict = {"image": image, "classes": classes.astype(int)+1, "bboxes": bboxes.astype(int),\
                        "image_path": join(self.images_path, image_name)}
        if vis:
            self.visualize(data_dict)
        return data_dict
    




if __name__ == "__main__":
    "A simple example of how to use this class."
    dl = DataLoader()
    dict_ = dl.loadTrainImageAndLabel(9)
    dl.visualize(dict_)