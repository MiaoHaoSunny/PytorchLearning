import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.utils.data as Data

# 1. Basic autograd example1
x = torch.tensor(1., requires_grad=True)
w = torch.tensor(2., requires_grad=True)
b = torch.tensor(3., requires_grad=True)

y = w * x + b
y.backward()
print(x.grad)
print(w.grad)
print(b.grad)
# 2. Basic autograd example2
x = torch.randn(10, 3)
y = torch.randn(10, 2)
linear = nn.Linear(3, 2)
print(linear)
print('W:', linear.weight)
print('b:', linear.bias)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)

pred = linear(x)
loss = criterion(pred, y)
print('loss:', loss.item())
loss.backward()
print('dL/dW: ', linear.weight.grad)
print('dL/db: ', linear.bias.grad)
optimizer.step()
pred = linear(x)
loss = criterion(pred, y)
print('loss after 1 step optimization: ', loss.item())

# 3. loading data from numpy
x = np.array([[1, 2], [3, 4]])
y = torch.from_numpy(x)
z = y.numpy()
train_dataset = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,
    transform=transforms.ToTensor(),
    download=True
)
image, label = train_dataset[0]
print(image.size())
print(label)
print(train_dataset.train_data.size())
plt.imshow(train_dataset.train_data[0].numpy())
plt.show()
train_loader = Data.DataLoader(
    dataset=train_dataset,
    batch_size=64,
    shuffle=True
)
data_iter = iter(train_loader)
images, labels = data_iter.next()
print(images.size())
# 预处理模型
resnet = torchvision.models.resnet18(pretrained=True)
for param in resnet.parameters():
    param.requires_grad = False

resnet.fc = nn.Linear(resnet.fc.in_features, 100)
images = torch.randn(64, 3, 224, 224)
outputs = resnet(images)
print(outputs.size())
