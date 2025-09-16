#%%
import torch  as t
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import model
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import os
import time
from sklearn.metrics import multilabel_confusion_matrix

os.chdir('/home/cip/nf2025/fo87pyzu/Test/exercise4/src_to_implement')
print(os.getcwd())  # Shows current working directory
print(os.path.exists("data.csv"))  # Should be True if file is here

import torch
import torch.nn as nn

class WeightedBCELoss(nn.Module):
    def __init__(self, weights: torch.Tensor):
        """
        Args:
            pos_weight: Tensor of shape [num_classes], weighting positives more heavily
        """
        super().__init__()
        self.weights = weights

    def forward(self, probs: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            probs: predicted probabilities, shape [batch_size, num_classes]
            targets: ground truth labels (0 or 1), shape [batch_size, num_classes]
        Returns:
            Weighted binary cross-entropy loss (scalar)
        """
        eps = 1e-8  # numerical stability
        loss = -(
            self.weights * targets * torch.log(probs + eps) +
            (1 - targets) * torch.log(1 - probs + eps)
        )
        return loss.mean()

#Params



print("CUDA available:", torch.cuda.is_available())

# load the data from the csv file and perform a train-test-split
# this can be accomplished using the already imported pandas and sklearn.model_selection modules
data = pd.read_csv('./data.csv',sep=';')
train_data, test_data = train_test_split(data,train_size=0.8,test_size=0.2)
train_data, val_data = train_test_split(train_data,train_size=0.8,test_size=0.2)

train_dataset = ChallengeDataset(train_data,'train')
val_dataset = ChallengeDataset(val_data,'val') 
test_dataset = ChallengeDataset(test_data,'test')

print(f"Train data size: {len(train_data)}")
print(f"Validation data size: {len(val_dataset)}")
print(f"Test data size: {len(test_data)}")


#%%
# set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
batch_size=32
trainLoader = DataLoader(train_dataset,batch_size,shuffle=True)
valLoader = DataLoader(val_dataset,batch_size,shuffle=True)
testLoader = DataLoader(test_dataset,batch_size,shuffle=True)



#Test 
img,label = next(iter(trainLoader))
fig,axs = plt.subplots(3,3)
axs = axs.flatten()

# for i in range(9):
#     axs[i].imshow(img[i][0],cmap='gray')
#     axs[i].axis('off')
#     axs[i].set_title(f"Label:{label[i].numpy()}")

#%%
cuda = True
# create an instance of our ResNet model
model1 = model.ResNet()
# set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
loss_criterion = WeightedBCELoss(weights = torch.tensor([4,8]).cuda())  #Balancing dataset through loss function
# set up the optimizer (see t.optim)
optim = t.optim.Adam(model1.parameters(),lr=0.001)
# create an object of type Trainer and set its early stopping criterion



Trainer_1 = Trainer(model1,loss_criterion,optim,trainLoader,valLoader,early_stopping_patience=np.inf,cuda=cuda)


# go, go, go... call fit on trainer
start = time.perf_counter()  # high-resolution timer
res =Trainer_1.fit(epochs=50)
end = time.perf_counter()
elapsed = end - start
print(f"Execution time: {elapsed:.4f} seconds")
 

# plot the results
plt.figure()
plt.plot(np.arange(len(res["train_losses"]))+1, res["train_losses"], label='train loss')
plt.plot(np.arange(len(res["val_losses"]))+1, res["val_losses"], label='val loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
#plt.yscale('log')
plt.legend()
plt.savefig('losses.png')

plt.figure()
plt.plot(np.arange(len(res["train_metrics"]))+1, res["train_metrics"], label='train metric')
plt.plot(np.arange(len(res["val_metrics"]))+1, res["val_metrics"], label='val metric')
plt.xlabel("Epochs")
plt.ylabel("F1 score")
plt.legend()
plt.savefig('F1_metrics.png')

#Test

best_model_ckp = t.load('checkpoints/best_model.ckp', 'cuda' if Trainer_1._cuda else None)
best_model = model.ResNet()
best_model.load_state_dict(best_model_ckp["state_dict"])
if cuda:
    best_model.cuda()

with torch.no_grad():
    all_predictions = []
    all_labels  = []
    for data in testLoader:
        images,labels = data
        if cuda:
            images = images.cuda()
            labels = labels
        prediction = best_model(images)
        all_predictions.append(prediction.detach().cpu())
        all_labels.append(labels)

    all_predictions= t.cat(all_predictions).numpy()
    all_labels = t.cat(all_labels).numpy()
    metric = Trainer_1.calculate_metric(all_predictions,all_labels)
print(f"Testing F1 score:{metric:.2f}")

# Compute confusion matrices per class
all_predictions = all_predictions>0.7
conf_matrices = multilabel_confusion_matrix(all_labels, all_predictions)

for i, cm in enumerate(conf_matrices):
    print(f"Confusion matrix for class {i}:\n{cm}\n")

    

        


#Is it good to set random seed?

