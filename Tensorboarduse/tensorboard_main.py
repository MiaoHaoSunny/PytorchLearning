import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
from torchvision import transforms
from Advanced.Tensorboard.logger import Logger


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = torchvision.datasets.MNIST(root='E:\\CodingDocument\\Pycharm\\TorchTutorials\\Basics\\',
                                     train=True,
                                     transform=transforms.ToTensor(),
                                     download=False)

data_loader = Data.DataLoader(dataset=dataset,
                              batch_size=100,
                              shuffle=True)


class NeuralNet(nn.Module):
    def __init__(self, input_size=784, hidden_size=500, num_classes=10):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


model = NeuralNet().to(device)

logger = Logger('./logs')

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

data_iter = iter(data_loader)
iter_per_epoch = len(data_loader)
total_step = 50000

# Start training
for step in range(total_step):

    # Reset the data_iter
    if (step + 1) % iter_per_epoch == 0:
        data_iter = iter(data_loader)

    # Fetch images and labels
    images, labels = next(data_iter)
    images, labels = images.view(images.size(0), -1).to(device), labels.to(device)

    # Forward pass
    outputs = model(images)
    loss = criterion(outputs, labels)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Compute accuracy
    _, argmax = torch.max(outputs, 1)
    accuracy = (labels == argmax.squeeze()).float().mean()

    if (step + 1) % 100 == 0:
        print('Step [{}/{}], Loss: {:.4f}, Acc: {:.2f}'
              .format(step + 1, total_step, loss.item(), accuracy.item()))

        info = {'loss': loss.item(), 'accuracy': accuracy.item()}

        for tag, value in info.items():
            logger.scalar_summary(tag, value, step+1)

        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            logger.histo_summary(tag, value.data.cpu().numpy(), step+1)
            logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), step+1)

        info = {'images': images.view(-1, 28, 28)[:10].cpu().numpy()}

        for tag, images in info.items():
            logger.image_summary(tag, images, step+1)