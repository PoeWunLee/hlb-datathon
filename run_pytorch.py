import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
import torch.optim as optim
from torch.autograd import Variable
import csv
import sys
sys.path.append(sys.path[0] + "/..")
from model.mlp import MLP
import pandas as pd 
os.environ['CUDA_LAUNCH_BLOCKING']='1'
import pickle

class HLBDataset(Dataset):

    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.data = []
        
        xDF = pd.read_csv(os.path.join(csv_path,"cleaned_data_encoded.csv"))
        yDF = pd.read_csv(os.path.join(csv_path,"y_DF_categorical_label.csv"))					

        xDF.drop(columns=['Submission_Mth', 'MG_OMV', 'MG_SnP', 'highest_MG_SnP_OMV', 
                          'Free_Lease_Hold_Ind', 'Status_Of_Completion', 'Stage_Of_Completion','Title_Type',
                          'Build_Up_Area', 'Land_Area',
                          'Postcode', 'Property_State'],inplace=True)
        
        xcolList = xDF.columns
        ycolList = yDF.columns

        X = xDF[xcolList[0:5]]
        y = yDF[ycolList[-1]]

        X = X.values
        print(X)
        y = y.values - 1

        line_count = 0
        for i in range(len(y)):
            self.data.append({'X':X[i],'y':y[i]})
            line_count += 1


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        label = self.data[idx]['y']
        X = self.data[idx]['X']
  
        parameters = np.array(X)
        label = int(label)

        data = {"parameters": parameters, "class": label}

        return data

def calculate_accuracy(fx, y):
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = correct.float()/preds.shape[0]
    return acc

"""
TRAINING LOOP
"""
def do_train(model, device, trainloader, criterion, optimizer):
    model.train()

    train_loss = 0.0
    train_acc = 0.0

    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a dict of [inputs, class]
        inputs, target = data['parameters'], data['class']
        inputs = Variable(inputs).float().cuda()
        # target = Variable(target).float().cuda()
        target = torch.tensor(target, dtype=torch.long).cuda()
        # print(inputs)

        # forward + backward + optimize
        outputs = model(inputs)
        # print(outputs)

        loss = criterion(outputs, target)

        acc = calculate_accuracy(outputs, target)
       
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print statistics
        train_loss += loss.item()
        train_acc += acc.item()

    return train_loss/len(trainloader), train_acc/len(trainloader)


"""
TESTING LOOP
"""
def do_test(model, device, testloader, criterion):
    # model.load_state_dict(torch.load(weights_pth))
    model.eval()

    test_acc = 0.0
    test_loss = 0.0
    with torch.no_grad():
        for data in testloader:
            inputs, target = data['parameters'].to(device), data['class'].to(device)
            inputs = Variable(inputs).float().cuda()
            target = torch.tensor(target, dtype=torch.long).cuda()

            outputs = model(inputs)
            loss = criterion(outputs, target)

            preds = outputs.max(1, keepdim=True)[1]
            
            # print(preds.cpu().numpy().flatten())
            # print(target)
            correct = preds.eq(target.view_as(preds)).sum()

            acc = correct.float()/preds.shape[0]
            
            test_loss += loss.item()
            test_acc += acc.item()
            
    return test_loss/len(testloader), test_acc/len(testloader)

def save_data(train_loss_array, train_acc_array, test_loss_array, test_acc_array):
    with open('train_loss', 'wb') as f:
        pickle.dump(train_loss_array, f)
    with open('train_acc_array', 'wb') as f:
        pickle.dump(train_acc_array, f)
    with open('test_loss_array', 'wb') as f:
        pickle.dump(test_loss_array, f)
    with open('test_acc_array', 'wb') as f:
        pickle.dump(test_acc_array, f)

def plot_graph(train_loss_array, train_acc_array, test_loss_array, test_acc_array):
    plt.figure()
    plt.subplot(121)
    plt.title('Loss')
    plt.plot(train_loss_array)
    plt.plot(test_loss_array)
    plt.grid(color='k', linestyle='-', linewidth=0.1)
    plt.legend(['train_loss','test_loss'])

    plt.subplot(122)
    plt.title('Accuracy')
    plt.plot(train_acc_array)
    plt.plot(test_acc_array)
    plt.grid(color='k', linestyle='-', linewidth=0.1)
    plt.legend(['train_acc','test_acc'])
    plt.show()

def main(args):
    input_size = 5
    output_size = 5
    epochs = 100

    # Make the training and testing set to be less bias
    hlb_dataset = HLBDataset(args.dataset_path[0])

    hlb_train, hlb_test = random_split(hlb_dataset, (round(0.9*len(hlb_dataset)), round(0.1*len(hlb_dataset))))
    
    print(args.weights_path[0])
    print(f'Number of training examples: {len(hlb_train)}')
    print(f'Number of testing examples: {len(hlb_test)}')

    trainloader = DataLoader(hlb_train, batch_size=2048, shuffle=True, num_workers=2)
    testloader = DataLoader(hlb_test, batch_size=1024, shuffle=False, num_workers=2)

    save_weights_pth = args.weights_path[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(input_size=input_size, output_size=output_size)
    print(model)

    model.to(device)

    weights = torch.tensor([1.0, 0.25, 0.88, 0.9, 0.9])
    
    criterion = nn.CrossEntropyLoss(weight=weights).cuda()
    learning_rate = 1e-5
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_acc_array = []
    train_loss_array = []
    test_acc_array = []
    test_loss_array = []
    for i in range(epochs):  # loop over the dataset multiple times
        train_loss, train_acc = do_train(model, device, trainloader, criterion, optimizer)
        
        print('Epoch {} Train loss: {} Train acc: {}'.format(i, train_loss, train_acc))
        train_acc_array.append(train_acc)
        train_loss_array.append(train_loss)

        test_loss, test_acc = do_test(model, device, testloader, criterion)
        test_acc_array.append(test_acc)
        test_loss_array.append(test_loss)

        print('Test acc: {} Test loss: {}'.format(test_acc, test_loss))

        if i % 50 == 0:
            learning_rate = learning_rate * 0.99
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
    
    torch.save(model.state_dict(), args.weights_path[0])
    save_data(train_loss_array, train_acc_array, test_loss_array, test_acc_array)
    plot_graph(train_loss_array, train_acc_array, test_loss_array, test_acc_array)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Code to train feature vector network"
    )
    parser.add_argument(
        "--dataset-path",
        default="./data/",
        nargs="+",
        metavar="csv",
        help="Path to the csv file",
        type=str
    )
    parser.add_argument(
        "--weights-path",
        default="./my_weights.pth",
        nargs="+",
        metavar="WEIGHTS_PATH",
        help="Path to save the weights file",
        type=str
    )
    parser.add_argument(
        "--eval-only",
        help="set model to evaluate only",
        action='store_true'
    )

    args = parser.parse_args()

    main(args)
