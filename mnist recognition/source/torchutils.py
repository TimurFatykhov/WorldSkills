from tqdm import tqdm_notebook as tqdm
import numpy as np
import torch

class ModelWithAPI():
    def __init__(self, model, optim, criterion, device='cuda'):
        self.model = model
        self.optim = optim
        self.device = device
        self.criterion = criterion
        self.lr = self.optim.param_groups[0]['lr']
        
    def fit_loader(self, train_loader, epochs, val_loader=None, verbose=False, lr_decay_every=None):
        loss_story_train = []
        loss_story_val = []
        
        model = self.model
        optim = self.optim
        
        model.to(self.device)
        
        bar = tqdm(range(1, epochs+1))
        for epoch in bar:
            epoch_loss, counter = 0, 0
            for x, y in train_loader:
                model.train()
                x = x.to(self.device)
                y = y.to(self.device)
                
                score = model(x)
                
                loss = self.criterion(score, y)
                
                optim.zero_grad()
                loss.backward()
                optim.step()
                
                epoch_loss += loss.item()
                counter += 1
            
            loss_story_train.append(epoch_loss / counter)
            
            # update process-bar
            descr = None
            if val_loader is None:
                descr = 'loss: %8.5f' % (loss_story_train[-1])
            else:
                val_loss = self.evaluate_loss_loader(val_loader)
                loss_story_val.append(val_loss)
                descr = 'train loss: %8.5f | valid loss: %8.5f' % (loss_story_train[-1], val_loss)
            bar.set_description(descr)
            
            # adjust learning rate
            if lr_decay_every is not None and epoch % lr_decay_every == 0:
                lr = self.lr * (0.1 ** (epoch // lr_decay_every))
                print('New lr = %e' % lr)
                for param_group in self.optim.param_groups:
                    param_group['lr'] = lr
                
        
        return loss_story_train, loss_story_val
    
    
    def predict_score_loader(self, loader):
        score = []
        
        with torch.no_grad():
            self.model.to(self.device)
            self.model.eval()
            
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)
                
                score.append(self.model(x))
                
        return torch.cat(score)
    
    
    def evaluate_loss_loader(self, loader):
        with torch.no_grad():
            self.model.eval()
            cum_loss, counter = 0, 0
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)
                
                score = self.model(x)
                
                loss = self.criterion(score, y)
                cum_loss += loss.item()
                counter += 1
        
        return cum_loss / counter
    
    
    def evaluate_metrics_loader(self, loader, metrics=None):
        if metrics is None:
            metrics = ['accuracy']
            
        scores = self.predict_score_loader(loader)
        pred = torch.max(scores, dim=1)[1]
        return pred.to('cpu').numpy()
    
    def predict_numpy_sample(self, array):
        if len(array.shape) == 2:
            array = array.reshape((1, 1) + (array.shape))
        
        t = torch.Tensor(array)
        
        with torch.no_grad():
            self.model.to(self.device)
            self.model.eval()
            
            pred = self.model(t)
            
        return pred.to('cpu').numpy()
            

class Softmax(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        e = torch.exp(x - x.max(1, True)[0] )
        summ = e.sum(1, True)[0]
        return e / summ
            

class Flatten(torch.nn.Module):
    def forward(self, x):
        N = x.shape[0]
        return x.view(N, -1)
            