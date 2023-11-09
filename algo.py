import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np
from PIL import Image
import torch.optim as optim
import torchsummary as summary
import matplotlib.pyplot as plt


# 數據隨機水平翻轉、剪裁、顏色失真、旋轉
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
])

# CIFAR-10 加載
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

vgg19_bn=torchvision.models.vgg19_bn(weights=False, num_classes=10)
if torch.cuda.is_available():
    device = torch.device("cuda")
vgg19_bn.to(device)
#print(vgg19_bn)
print('edit version12')
with open('training_records.txt', 'w') as file:
    file.write("Epoch\tTrain Loss\tTrain Accuracy\tValidation Loss\tValidation Accuracy\n")

#損失函數、優化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(vgg19_bn.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
scheduler = StepLR(optimizer, step_size=20, gamma=0.1)

train_accuracy = []
train_loss = []
val_accuracy = []
val_loss = []
#train
num_epochs = 100
best_val_loss = float("inf")
patience = 5
cur_patience = 0
for epoch in range(num_epochs):
    print(f"Epoch {epoch +1}/{num_epochs}")
    run_loss=0.0
    correct_train=0
    total_train=0
    vgg19_bn.train()
    for i,data in enumerate(trainloader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = vgg19_bn(inputs)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        
        run_loss += loss.item()

        _, predicted = torch.max(outputs.data,1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
    
    scheduler.step()
    train_accuracy.append(100 * correct_train / total_train)
    train_loss.append(run_loss / len(trainloader))

    print(f"Train Loss: {train_loss[-1]:.4f} Train Accuracy: {train_accuracy[-1]:.2f}%")
    
    #跑完1 epoch -> validate
    vgg19_bn.eval()
    correct_val = 0
    total_val = 0
    run_loss_val = 0.0
    
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = vgg19_bn(inputs)
            
            loss = criterion(outputs, labels)
            run_loss_val += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
        
    val_accuracy.append(100 * correct_val / total_val)
    val_loss.append(run_loss_val / len(testloader))
    
    print(f"Validation Loss: {val_loss[-1]:.4f} Validation Accuracy: {val_accuracy[-1]:.2f}%")
    
    #紀錄訓練過程
    #with open('training_records.txt', 'a') as file:
    #    file.write(f"{epoch}\t{train_loss[-1]:.4f}\t{train_accuracy[-1]:.2f}%\t{val_loss[-1]:.4f}\t{val_accuracy[-1]:.2f}%\n")
    #紀錄權重檔案 怕執行到一半斷線
    #if epoch % 15 == 14:
    #    torch.save(vgg19_bn, f'Wei_Vgg19-bn_weights_{epoch + 1}.pth')
    
    #early stop 防止overfitting
    if (val_loss[-1] < best_val_loss):
        best_val_loss = val_loss[-1]
        cur_patience = 0
    else:
        cur_patience += 1
        if cur_patience >= patience and epoch >= 49:
            print(f'Early stop after {epoch + 1} epoch')
            break
    
    

print('Done')
print('Save the files')        
np.savetxt('train_loss.txt', train_loss)
np.savetxt('val_loss.txt', val_loss)

np.savetxt('train_accuracy.txt', train_accuracy)
np.savetxt('val_accuracy.txt', val_accuracy)

torch.save(vgg19_bn, 'Wei_Vgg19-bn_weights.pth')