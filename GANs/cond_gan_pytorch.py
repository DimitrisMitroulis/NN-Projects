"""
Created on Tue Jan 17 13:24:23 2023

@author: DIMITRIS

Description: Simple GAN network, meant for educational purposes
It's working but it isn't training so well
"""


#%% Import and stuff
import torch
from torch import nn
from torchvision import transforms
import torchvision
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
import time
import torchvision.utils as vutils


NUM_EPOCHS = 500
LR = 0.0002
LATENT_DIM = 100
IMG_SIZE = 28
CHANNELS = 1
B1 = 0.5
B2 = 0.999

GEN_STATE_DICT = "gen_state_dict"
DISC_STATE_DICT = "disc_state_dict"
GEN_OPTIMIZER = "gen_optimizer"
DISC_OPTIMIZER = "disc_optimizer" 


SHUFFLE = True
PIN_MEMORY = True
NUM_WORKERS = 0
BATCH_SIZE = 200


img_shape = (CHANNELS, IMG_SIZE, IMG_SIZE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device:{}'.format(device))


#%% helper funcitons
def save_checkpoint(state, filename="cond_gan_pytorch.pth.tar"):
    print("=> Saving chekpoint")
    torch.save(state, filename)    

def load_checkpoint(checkpoint):
    generator.load_state_dict(checkpoint[GEN_STATE_DICT])
    optimizer_G.load_state_dict(checkpoint[GEN_OPTIMIZER])
    discriminator.load_state_dict(checkpoint[DISC_STATE_DICT])
    optimizer_D.load_state_dict(checkpoint[DISC_OPTIMIZER])
       
#takes input tensor and return a tensor of same size but every element has different value
def build_fake_labels(old_list):
   
    new_list = []
    
    for i,x in enumerate(old_list):
        
        if (i%10) != x:
            new_list.append(i%10)
        else:
           new_list.append((x.item()+1)%10)
        
    return torch.tensor(new_list,dtype=torch.int64).to(device)
    

#%%train data
#transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])
    ])


train_dataset = torchvision.datasets.MNIST(
    root="~/torch_datasets", train=True, transform=transform, download=True
)

test_dataset = torchvision.datasets.MNIST(
    root="~/torch_datasets", train=False, transform=transform, download=True
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=32, shuffle=True, num_workers=0
)

example_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=1, shuffle=True, num_workers=0,drop_last=True,
    )


#%% # Detective: fake or no fake -> 1 output [0, 1]
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(2, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 1)
        
        self.emb = nn.Embedding(10,50) 
        self.emb_fc = nn.Linear(50, 784)
        
        
        self.nconv1 = nn.Conv2d(2, 64, kernel_size=5)
        self.nconv2 = nn.Conv2d(64, 128, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.nfc1 = nn.Linear(1152, 164)
        self.nfc2 = nn.Linear(164, 1)
    
        
  #oldWay flag to select between 2 train methods, not sure which is best yet
    def forward(self, x, c ,oldWay=False):
        
        
        c = self.emb(c)
        c = self.emb_fc(c)
        c= c.view(-1,1,28,28)
        x = torch.cat((c,x),1) #concat image[1,28,28] with text [1,28,28]
            
        if oldWay: 
        
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
            # Flatten the tensor so it can be fed into the FC layers
            x = x.view(-1, 320)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = self.fc2(x)
    
        
        
        else:
            x = F.relu(self.nconv1(x))
            x = F.relu(self.nconv2(x))
            x = self.pool(x)
            x = self.pool2(x)
            x = x.view(-1, 1152)
            x = F.relu(self.nfc1(x))
            x = F.dropout(x, training=self.training)
            x = self.nfc2(x)
            
        
        x = torch.sigmoid(x)
        return x
            
            
        
        
    
#%% Generate Fake Data: output like real data [1, 28, 28] and values -1, 1
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(LATENT_DIM, 7*7*63)  # [n,100]->[n,3087]
        self.ct1 = nn.ConvTranspose2d(64, 32, 4, stride=2) # [n, 64, 16, 16] [32,..,..]
        self.ct2 = nn.ConvTranspose2d(32, 16, 4, stride=2) # [n, 32, , ]->[n, 16, 34, 34]
        self.conv = nn.Conv2d(16, 1, kernel_size=7)  # [n, 16, 34, 34]-> [n, 1, 28, 28]
        
        self.emb = nn.Embedding(10,50) 
        self.label_lin = nn.Linear(50,49)
        self.conv_x_c = nn.ConvTranspose2d(65,64,4,stride=2) # upsample [65,7,7] -> [64,14,14]
    

    def forward(self, x ,c ):
        # Pass latent space input into linear layer and reshape
        x = self.lin1(x) #(n,100) -> (n,3136)
        x = F.relu(x)
        x = x.view(-1, 63, 7, 7) # (n,3136) -> (64,7,7)
        
        #Encode label
       
        
        c = self.emb(c) #(n,) -> (n,50)
        c = self.label_lin(c) #(n,50) -> (n,49)
        c = c.view(-1,1,7,7) #(n,49) -> (n,1,7,7)
        
        
        x = torch.cat((c,x),1) #concat image[64,7,7] with text [1,7,7]
        #x = self.conv_x_c(x) #[65,7,7] -> [64,16,16]
        #x = F.relu(x)
            
        
        x = self.ct1(x) #[n, 64, 16, 16] [32,34,34]
        x = F.relu(x)
        
        
        # Upsample to 34x34 (16 feature maps)
        x = self.ct2(x)
        x = F.relu(x)
        
        # Convolution to 28x28 (1 feature map)
        return self.conv(x)    
    
    
#%% #loss function 
loss_func = nn.BCELoss()

# Initialize generator and discriminator
generator = Generator().to(device)
discriminator = Discriminator().to(device)

if torch.cuda.is_available():
    generator.cuda()
    discriminator.cuda()
    loss_func.cuda()
    
optimizer_G = torch.optim.Adam(generator.parameters(), lr=LR,betas=(B1 ,B2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=LR,betas=(B1 ,B2))
   
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor   



#%% Train both models
if torch.cuda.is_available():
    generator.to(device)
    discriminator.to(device)
    loss_func.to(device)
    
img_list = []
G_losses = []
D_losses = []
iters = 0
for epoch in range(NUM_EPOCHS):
    st = time.time()
    for i, (imgs,labels) in enumerate(train_loader):
          
        
        # Adversarial ground truths
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)
        
        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0],LATENT_DIM))))

        # transform to tensor [256,1,28,28]
        real_imgs = Variable(imgs.type(Tensor))
        fake_labels = build_fake_labels(labels.to(device))
        
        labels = labels.to(device)
        
        optimizer_D.zero_grad()
        #Forward pass through Discriminator
        s_r = discriminator(real_imgs,labels)
        sr_loss = loss_func(s_r, valid)
        sr_loss.backward()
        D_x = s_r.mean().item()



        # Generate a batch of images
        gen_imgs = generator(z,labels)
        
        # Calculate D's loss on the all-fake batch
        #s_w = discriminator(real_imgs,fake_labels)
        s_f = discriminator(gen_imgs.detach(),labels.detach())
        sf_loss = loss_func(s_f, fake)
        sf_loss.backward()
        D_G_z1 = s_f.mean().item()
        
        d_loss = s_f + s_r
        optimizer_D.step()
        
       
        # Measure discriminator's ability to classify real from generated samples
        #sr_loss = loss_func(s_r, valid)
        #sw_loss = loss_func(s_w, fake)
        #sf_loss = loss_func(s_f, fake)
        
        #d_loss = sr_loss + ((sw_loss+sf_loss) / 2)

        
        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()
        #Loss measures generator's ability to fool the discriminator
        s_f = discriminator(gen_imgs,labels)
        g_loss = loss_func(s_f, valid)
        g_loss.backward()
        D_G_z2 = s_f.mean().item()
        optimizer_G.step()
      

        
       # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f tD(x): %.4f tD(G(z)): %.4f / %.4f'
                  % (epoch, NUM_EPOCHS, i, len(train_loader),d_loss[0].item(), g_loss.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(g_loss.item())
        D_losses.append(d_loss[0].item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == NUM_EPOCHS-1) and (i == len(train_loader)-1)):
            with torch.no_grad():
                fake = generator(z,labels).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1
    

    
    
#%% gen image 
generator.to('cpu')
discriminator.to('cpu')

rand_latent = torch.rand_like(torch.Tensor(1,100))
caption = 0
with torch.no_grad():
    for image,_ in example_loader:
        f, axarr = plt.subplots(1)
        

        caption = torch.tensor(caption, dtype=torch.int64)
        fake_image = generator(rand_latent,caption)  
       
        fake_image = fake_image[0].reshape(-1, 28, 28)
        axarr.imshow(fake_image[0].cpu())    
        break
    
    
#%%


    
#%% Discriminate image
generator.to('cpu')
discriminator.to('cpu')


with torch.no_grad():
    for  i, (imgs, labels) in enumerate(example_loader):
        int = 0#random.randint(0, 1)
        f, axarr = plt.subplots(1)
        
        fake_labels = build_fake_labels(labels.to(device))
        labels = labels.to('cpu')
        z = Variable(Tensor(np.random.normal(0, 1, (1,LATENT_DIM)))).cpu()
        caption = torch.tensor(1, dtype=torch.int64)
        
        fake_image = generator(z,caption)#.detach().numpy()
        
        
        #imgs = imgs[0]
        #print(fake_image.shape)
        
        #feed discriminator fake image, expect "0" output
        if int == 0:
            axarr.imshow(fake_image[0].reshape(-1, 28, 28)[0])
            pred = discriminator(fake_image,caption)
            print("Discriminator Prediction: {},Should be: {}".format(pred,"0"))
        #feed discriminator real image, expect "1" output
        else:
            axarr.imshow(imgs[0].reshape(-1, 28, 28)[0])
            pred = discriminator(imgs,labels[0])
            print("Discriminator Prediction: {},Should be: {}, label= {}".format(pred,"1",labels[0]+1))
        
        
        
        
        
        break    
#%%
train_disc(1)
    
#%%Train disriminator for x epochs 

def train_disc(epochs=100):
    if torch.cuda.is_available():
        generator.to(device)
        discriminator.to(device)
        loss_func.to(device)
        
    for epoch in range(epochs):
        st = time.time()
        for i, (imgs,labels) in enumerate(train_loader):
              
            
            # Adversarial ground truths
            valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)
            
            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0],LATENT_DIM))))
    
            # transform to tensor [256,1,28,28]
            real_imgs = Variable(imgs.type(Tensor))
            fake_labels = build_fake_labels(labels.to(device))
            
            labels = labels.to(device)
            # -----------------
            #  Train Generator
            # -----------------
    
    
            # Generate a batch of images
            gen_imgs = generator(z,labels)
    
    
            #Pass fake and real images through discriminator        
            s_r = discriminator(gen_imgs,labels)
            s_w = discriminator(real_imgs,fake_labels)
            s_f = discriminator(real_imgs,labels)
            
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            # Measure discriminator's ability to classify real from generated samples
            optimizer_D.zero_grad()
            sr_loss = loss_func(s_r, valid)
            sw_loss = loss_func(s_w, fake)
            sf_loss = loss_func(s_f, fake)
            
            d_loss = (sr_loss + ((sw_loss+sf_loss) / 2))/2
    
            d_loss.backward()
            optimizer_D.step()
          
            batches_done = epoch * len(train_loader) + i
            
           
        
        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] " 
            % (epoch,NUM_EPOCHS, i, len(train_loader), d_loss.item()))  
        
        et = time.time()
        print(et-st)
       
    
    
    
    
#%%
#TODO: add a check if prediction ~0.5 warn not to save checkpoint
checkpoint = {GEN_STATE_DICT : generator.state_dict(), 
              GEN_OPTIMIZER : optimizer_G.state_dict(),
              DISC_STATE_DICT : discriminator.state_dict(),
              DISC_OPTIMIZER : optimizer_D.state_dict()}
save_checkpoint(checkpoint)

#%% 
load_checkpoint(torch.load("cond_gan_pytorch.pth.tar",map_location=(device)))



#%% For test, 
#what[1] = what the number is supposed to be
#what[0] = tensor with image


for i, (imgs, what) in enumerate(train_loader):
    f, axarr = plt.subplots(1)
    img = imgs.reshape(-1, 28, 28)
    axarr.imshow(img[0])
    print(what[0].item())
    break


#%%

with torch.no_grad():
        
    for i, (imgs, what) in enumerate(train_loader):
        
        #conv_concat = nn.ConvTranspose2d(65,64,4,stride=2)
        #z = Variable(Tensor(np.random.normal(0, 1, (200,LATENT_DIM)))).to("cpu")
        #lin1 = nn.Linear(100, 7*7*64)  # [n,100]->[n,3136] 
        #x.to(device)
        #x = lin1(z)
        #x = F.relu(x)
        #x = x.view(-1, 64, 7, 7) # [n,3136] -> [n, 64, 7, 7]
        #c = emb(what)
        #c = label_lin(c)
        #c = c.view(-1,1,7,7)
        #res = torch.cat((c,x),1)
        #res = conv_concat(res)
        
        x = Variable(imgs.type(Tensor)).to("cpu")
        
        
        nconv1 = nn.Conv2d(2, 64, kernel_size=5)
        nconv2 = nn.Conv2d(64, 128, kernel_size=5)
        pool = nn.MaxPool2d(kernel_size=3)
        pool2 = nn.MaxPool2d(kernel_size=2)
        nfc1 = nn.Linear(1152, 164)
        nfc2 = nn.Linear(164, 1)
    
                
        emb = nn.Embedding(10,50) 
        emb_fc = nn.Linear(50, 784)
        
     
        c = emb(what)
        c = emb_fc(c)
        c = c.view(-1,1,28,28)
        
        x = torch.cat((c,x),1) #concat image[1,28,28] with text [1,28,28]
        
        
        x = F.relu(nconv1(x))
        x = F.relu(nconv2(x))
        x = pool(x)
        x = pool2(x)
        x = x.view(-1, 1152)
        x = F.relu(nfc1(x))
        x = F.dropout(x, training=True)
        x = F.relu(nfc2(x))
        
        break 
    
#%%


for i, (imgs, labels) in enumerate(train_loader):
    
    fake_labels = build_fake_labels(labels)
    
    
    break
