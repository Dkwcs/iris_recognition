
import glob
import os
import cv2
import numpy as np

from segmentation import segment


def extractFeature(img_filename):
    im = cv2.imread(img_filename, 0)
    ### Pass needed algorithm
    segmented_iris, cirpupil, ciriris = segment(im)
    segmented_iris = cv2.equalizeHist(segmented_iris)
    segmented_iris = cv2.cvtColor(segmented_iris, cv2.COLOR_BGR2RGB)
    cv2.imshow('origin iris', im)
    cv2.imshow('segmented iris', segmented_iris)
    cv2.waitKey(0)

#extractFeature("F:/repo/iris_recognition-1/CASIA-Iris-Syn/005/S6005S01.jpg")

def extract_label_from_img(img_path):
    parts = img_path.split('\\')
    return int(parts[1])

def load_dataset(dataset_path):
    imgs = []
    labels = []

    for root, dirs, files in os.walk(dataset_path):
        num_in_folder=0
        for file in files:
            if file.endswith(".jpg"):
                img_path = os.path.join(root, file)
                print(img_path)
                img =  extractFeature(img_path)
                label = extract_label_from_img(img_path) - 1
            
                imgs.append(img)
                labels.append(label)
                print('../../normalized_dataset/'+str(int(label))+'.'+str(int(num_in_folder))+'.jpg')
                cv2.imwrite('../../normalized_dataset/'+str(label)+'.'+str(num_in_folder)+'.jpg', img)
                num_in_folder += 1
    print(len(imgs))
    print(len(np.unique(labels)))
    return imgs, labels

def load_casia_dataset(dataset_path):
    imgs = []
    labels = []
    label=0

    for filepath in glob.iglob(dataset_path):
        print(filepath)
        num_in_folder=0
        for filefilepath in glob.iglob(filepath+'/*'):
            if filefilepath[-1] == 'g':
                    img =  extractFeature(filefilepath)
                    print('../../normalized_casia_224x224_segmented_clahe/'+str(int(label))+'.'+str(int(num_in_folder))+'.jpg')
                    cv2.imwrite('../../normalized_casia_224x224_segmented_clahe/'+str(label)+'.'+str(num_in_folder)+'.jpg', img)
                    imgs.append(img)
                    labels.append(label)
                    num_in_folder = num_in_folder+1
        
        label=label+1

    print(len(imgs))
    print(len(np.unique(labels)))
    return imgs, labels

### Uncomment for prepare preprocessed datasets
#load_dataset("F:/repo/iris_recognition-1/CASIA1")
#load_casia_dataset("F:/repo/iris_recognition-1/CASIA-Iris-Syn/*")