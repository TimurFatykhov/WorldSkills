import numpy as np
import torch
import PIL
import pickle
import cv2 

TO_KNN = './knn.pickle'

class KNNClassifier():
    def __init__(self):
        global TO_KNN
        self.model = None
        try:
            with open(TO_KNN, 'rb') as inp:
                self.model = pickle.load(inp)
        except:
            self.model = None

    
    def predict(self, img_path='tmp.png'):
        if self.model is None:
            return -1, -1

        img = PIL.Image.open(img_path)
        arr = np.asarray(img)
        if len(arr.shape) > 2:
            arr = arr[..., 0] # rgb to gray
        arr = cv2.resize(arr, (28, 28), interpolation=cv2.INTER_AREA)
        arr = arr.reshape(-1, 28*28)

        proba = self.model.predict_proba(arr)[0]
        return proba.argmax(), proba.max()


class NNClassifier():
    def __init__(self, path='./cnn_model.pt'):
        try:
            self.model = torch.load(path)
        except:
            self.model = None

    def predict(self, img_path='tmp.png', device='cpu'):
        if self.model is None:
            return -1, -1

        img = PIL.Image.open(img_path)
        arr = np.asarray(img)
        if len(arr.shape) > 2:
            arr = arr[..., 0] # rgb to gray
        arr = arr / arr.max()
        arr = cv2.resize(arr, (28, 28), interpolation=cv2.INTER_AREA)

        t = torch.Tensor(arr.reshape((1, 1) + arr.shape))
        
        model = self.model.to(device)
        model.eval()
        score = None
        with torch.no_grad():
            score = model(t.to(device))
            
        score = score.to('cpu').numpy()[0]
        predict = score.argmax()
        proba = score[predict]
        return predict, proba