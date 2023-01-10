# %% Import stuff
import torch
from torch import nn
from torchvision import transforms
import torchvision
import matplotlib.pyplot as plt


#torch.set_default_tensor_type('torch.cuda.FloatTensor')
tensor_transform = transforms.ToTensor()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device:',device)


        
        
# %% train data
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

train_dataset = torchvision.datasets.MNIST(
    root="~/torch_datasets", train=True, transform=transform, download=True
)

test_dataset = torchvision.datasets.MNIST(
    root="~/torch_datasets", train=False, transform=transform, download=True
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=128, shuffle=True, num_workers=8, pin_memory=True
)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=32, shuffle=False, num_workers=8
)


example_loade = trorch.utils.data.DataLoader(
    test_dataset, batch_size=1, shuffle=true, num_workers=4
)

# %% plot helper func

def plot():
    f, axarr = plt.subplots(2,2)

    for i, item in enumerate(image):

    # Reshape the array for plotting
        item = item.reshape(-1, 28, 28)
        axarr[0,0].imshow(item[0].cpu())

    for i, item in enumerate(reconstructed):
        item = item.reshape(-1, 28, 28).cpu()
        item = item.detach().numpy()
        axarr[0,1].imshow(item[0])
        
def getImage():
     for image, _ in example_loader:
        reconstructed = model(image)
        
        f, axarr = plt.subplots(2)
        item = item.reshape(-1, 28, 28)
        axarr[0].imshow(item[0].cpu())
        
        item = item.reshape(-1, 28, 28).cpu()
        item = item.detach().numpy()
        axarr[1].imshow(item[0])
        



# %% autoencoder class 
input_size = 28*28 #784
hidden_size = 128
code_size = 32


class autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        #Encoder 
        self.encoder = nn.Sequential(
            nn.Linear(input_size,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,code_size)
        )
        
        #Decoder
        self.decoder = nn.Sequential(
            nn.Linear(code_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()
        )
        self.model = nn.Sequential(
        
        )
        
        
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
# %% Define model, loss, optimzer

# Model Initialization
model = autoencoder()
model.to(device)
 
# Validation using MSE Loss function
loss_function = nn.MSELoss()

#Adam Optimizer with lr = 0.1
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
# %% Train model

epochs = 20
losses = 0

for epoch in range(epochs):
    for image, _ in train_loader:
        image = image.view(-1,28*28).to(device)
        optimizer.zero_grad()
        
        reconstructed = model(image)
        loss = loss_function(reconstructed , image)
        loss.backward()
        optimizer.step()
        losses += loss.item()
    losses = losses / len(train_loader)
    print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))
    plot()

# %% Output images

