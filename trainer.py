import torch as t
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm
import numpy as np

from sklearn.metrics import f1_score

        
class EarlyStopping():
    def __init__(self,patience):
        self._patience = patience
        self._best = t.inf
        self._epochs = 0
    
    def check_early_stop(self,val_loss):
        if val_loss < self._best:
            self._best = val_loss
            self._epochs=0
        else:
            if self._epochs >= self._patience:
                return True
            
            else:
                self._epochs+=1
                return False

class Trainer:

    def __init__(self,
                 model,                        # Model to be trained.
                 crit,                         # Loss function
                 optim=None,                   # Optimizer
                 train_dl=None,                # Training data set
                 val_test_dl=None,             # Validation (or test) data set
                 cuda=True,                    # Whether to use the GPU
                 early_stopping_patience=-1):  # The patience for early stopping
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda
        self._best_metric = 0

        self._early_stopping_patience = early_stopping_patience
        self._EarlyStopper = EarlyStopping(self._early_stopping_patience)

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()
            
    def save_checkpoint(self, epoch):
        if epoch%5==0:
            t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))
    
    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])
    def save_bestmodel(self,val_metric):
        if self._best_metric<val_metric:
            self._best_metric = val_metric
            t.save({'state_dict': self._model.state_dict()}, 'checkpoints/best_model.ckp')
            print("New best model saved")



        
    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,                 # model being run
              x,                         # model input (or a tuple for multiple inputs)
              fn,                        # where to save the model (can be a file or file-like object)
              export_params=True,        # store the trained parameter weights inside the model file
              opset_version=10,          # the ONNX version to export the model to
              do_constant_folding=True,  # whether to execute constant folding for optimization
              input_names = ['input'],   # the model's input names
              output_names = ['output'], # the model's output names
              dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                            'output' : {0 : 'batch_size'}})
            
    def train_step(self, x, y):
        # perform following steps:
        # -reset the gradients. By default, PyTorch accumulates (sums up) gradients when backward() is called. This behavior is not required here, so you need to ensure that all the gradients are zero before calling the backward.
        self._optim.zero_grad()

        # -propagate through the network
        outputs = self._model(x)

        # -calculate the loss
        loss = self._crit(outputs,y)

        # -compute gradient by backward propagation
        loss.backward()

        # -update weights
        self._optim.step()


        # -return the loss
        return loss,outputs
        
        
    
    def val_test_step(self, x, y):
        
        # predict
        outputs = self._model(x)
        # propagate through the network and calculate the loss and predictions
        loss = self._crit(outputs,y)
        # return the loss and the predictions
        return loss,outputs
        
    def train_epoch(self):
        # set training mode
        self._model.train(True)

        predictions = []
        all_labels = []
        running_loss = 0
        # iterate through the training set
        for i,data in tqdm(enumerate(self._train_dl)):
            images,labels = data
            # transfer the batch to "cuda()" -> the gpu if a gpu is given
            if self._cuda:
                images = images.cuda()
                labels = labels.cuda()
            # perform a training step
            train_loss, prediction = self.train_step(images,labels)
            running_loss+=train_loss.cpu().item()

            #Save labels and predictions
            predictions.append(prediction.detach().cpu())
            all_labels.append(labels.detach().cpu())


            
        # calculate the average loss for the epoch and return it
        avg_loss = running_loss/i
        predictions = t.cat(predictions).numpy()
        all_labels = t.cat(all_labels).numpy()
        metric = self.calculate_metric(predictions,all_labels)

        
        return avg_loss,metric
    def calculate_metric(self,predictions,labels,threshold=0.7):

        predictions = predictions>threshold

        #Calculate accuracy
        acc = np.mean((labels ==predictions),1)
        acc = np.mean(acc)
        #Calculate F1 score:
        F1 = f1_score(labels, predictions, average='macro')

        return F1

    @t.no_grad() #Activate no-grad mode
    def val_test(self):
        # set eval mode. Some layers have different behaviors during training and testing (for example: Dropout, BatchNorm, etc.). To handle those properly, you'd want to call model.eval()
        self._model.eval()
        # disable gradient computation. Since you don't need to update the weights during testing, gradients aren't required anymore. 

        running_loss=0
        predictions = []
        all_labels = []
        # iterate through the validation set
        for i, data in enumerate(self._val_test_dl,1):
            images,labels = data
            # transfer the batch to "cuda()" -> the gpu if a gpu is given
            if self._cuda:
                images = images.cuda()
                labels = labels.cuda()
            # perform a validation step
            loss,prediction = self.val_test_step(images,labels)
            running_loss+=loss.detach().cpu().item()
            #save the predictions and the labels for each batch #NEEDED because it can happen that F! score acts strangly
            predictions.append(prediction.detach().cpu())
            all_labels.append(labels.detach().cpu())


        # calculate the average loss and average metrics of your choice. You might want to calculate these metrics in designated functions
        avg_loss = running_loss/i

        predictions = t.cat(predictions).numpy()
        all_labels = t.cat(all_labels).numpy()
        metric = self.calculate_metric(predictions,all_labels)
        # return the loss and print the calculated metrics
        return avg_loss,metric
    
    
    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0
        # create a list for the train and validation losses, and create a counter for the epoch 
        train_losses = []
        val_losses = []
        val_metrics = []
        train_metrics =[]
        epoch = 0
        while True:

 
            # stop by epoch number
            if epoch>epochs: break
            # train for a epoch and then calculate the loss and metrics on the validation set
            train_loss,train_metric = self.train_epoch()
            val_loss,val_metric = self.val_test()
            # append the losses to the respective lists
            train_losses.append(train_loss)
            train_metrics.append(train_metric)
            val_losses.append(val_loss)
            val_metrics.append(val_metric)
    

            # use the save_checkpoint function to save the model (can be restricted to epochs with improvement)
            self.save_checkpoint(epoch)
            self.save_bestmodel(val_metric)
            # check whether early stopping should be performed using the early stopping criterion and stop if so
            terminate = self._EarlyStopper.check_early_stop(val_loss)
            if terminate:
                print("Early stopping!")
                break
            else:
                epoch+=1
            print(f"Epoch:{epoch}, Training loss: {train_loss:.2f}, Validation loss: {val_loss:.2f}, F1: {val_metric:.2f}")
            
        # return the losses for both training and validation
        results = {"train_losses":train_losses,
                   "train_metrics": train_metrics,
                   "val_losses": val_losses,
                   "val_metrics": val_metrics}

        return results
                    




        
        
