import torch
import numpy as np
from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt

class NNClassifier():
    def __init__(self, model, lr=1e-3, optimizer=None):
        """
        If optimizer is passed, then lr will be ignored
        """
        self.model = model
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        else:
            self.optimizer = optimizer
            
        self.train_history = []
        self.valid_history = []

    
    def predict_proba(self, X, batch_size=128):
        """
        Parameters:
        -----------
        - X: numpy.array
        - y: numpy.array
        - batch_size: int
        """
        self.model.to(self.device)
        X = torch.FloatTensor(X).to(self.device)
        N = len(X)
        
        proba = []
        with torch.no_grad():
            for i in range(0, N, batch_size):
                X_batch = X[i : min(i + batch_size, N)]

                proba.append(self.model(X_batch))
            
        proba = torch.cat(proba).to('cpu').numpy()
        return proba
    
    
    def predict(self, X, batch_size=128):
        """
        Parameters:
        -----------
        - X: numpy.array
        - y: numpy.array
        - batch_size: int
        """
        proba = self.predict_proba(X, batch_size)
        predict = proba.argmax(1)
        return predict
    
    
    def evaluate_score(self, X, y, batch_size=128):
        """
        Parameters:
        -----------
        - X: numpy.array
        - y: numpy.array
        - batch_size: int
        """
        predict = self.predict(X, batch_size)
        return (predict == y).mean()
    
    
    def loss(self, X, y, batch_size=128):
        """
        Parameters:
        -----------
        - X: numpy.array
        - y: numpy.array
        - batch_size: int
        """
        proba = self.predict_proba(X, batch_size)
        proba = torch.FloatTensor(proba).to(self.device)
        y = torch.LongTensor(y).to(self.device)
        loss = torch.nn.functional.cross_entropy(proba, y).item()
        return loss
    
    
    def show_history(self, hide_left=0):
        if self.valid_history is not None:
            N = len(self.train_history)
            plt.plot(np.arange(hide_left, N), self.train_history[hide_left:], color='blue', label='train')
        else: 
            print('Сначала обучите нейросеть!')
            return

        if self.valid_history is not None:
            plt.plot(np.arange(hide_left, N), self.valid_history[hide_left:], color='green', label='val')

        plt.legend()
        plt.grid()
        plt.show()
        
    
    def show_predict_grid(self, X, y, size=5, figsize=(15, 15)):
        pred = self.predict(X)
        
        fig, ax = plt.subplots(size, size, figsize=figsize)
        ax = np.ravel(ax)
        
        for i, img in enumerate(X[:size*size]):
            color = 'green' if y[i] == pred[i] else 'red'
            
            ax[i].imshow(np.transpose(X[i], (1, 2, 0)))
            ax[i].axis('off')
            ax[i].set_title('%d (%d)' % (pred[i], y[i]), color=color)
        
        plt.show()
            
            
            

    def fit(self, X, y, epochs, batch_size, valid_data=None, log_every_epoch=None):
        """
        Parameters:
        -----------
        - X: numpy.array
        
        - y: numpy.array
        
        - batch_size: int
        
        - valid_data: tuple (numpy.array, numpy.array)
            (X_valid, y_valid)
            
        - log_every_epoch: int
        """
        self.model.to(self.device)
        X = torch.FloatTensor(X).to(self.device)
        y = torch.LongTensor(y).to(self.device)

        N = len(X)
        
        bar = tqdm(range(1, epochs+1)) # progress bar
        for epoch in bar:
            cum_loss_train = 0
            part = 0
            for i in range(0, N, batch_size):
                part += 1
                X_batch = X[i : min(i + batch_size, N)]
                y_batch = y[i : min(i + batch_size, N)]

                proba_batch = self.model(X_batch)

                loss = torch.nn.functional.cross_entropy(proba_batch, y_batch)
                cum_loss_train += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            self.train_history.append(cum_loss_train / part)
                
            if valid_data is not None:
                valid_loss = self.loss(valid_data[0], valid_data[1], batch_size)
                self.valid_history.append(valid_loss)
                    
            if log_every_epoch is not None and epoch % log_every_epoch == 0:
                descr = None
                t_loss = self.train_history[-1]
                descr = ('t_loss: %5.3f' % t_loss)
                
                if valid_data is not None:
                    v_loss = self.valid_history[-1]
                    descr += ('v_loss: %5.3f' % v_loss)
                    
                bar.set_description(descr)
                    
                    
class Flatten(torch.nn.Module):
    def forward(self, x):
        N = x.shape[0]
        return x.view(N, -1)

    
class Softmax_layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        e = torch.exp(x - x.max(1, True)[0] )
        summ = e.sum(1, True)[0]
        return e / summ