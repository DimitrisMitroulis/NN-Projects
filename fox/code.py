import pandas as pd
import numpy as np
import torch

# %% Import and read data
df = pd.read_excel('wines.xlsx')
data = df.to_numpy()
wines = torch.from_numpy(data)

# %% Check it exists
wines.shape

# %% Split dataset
x = wines[:10,:11]
y = wines[:10,11]

# %% Calculate betas
betas = []
y_mean =  y.mean()

for i in range(len(x[0,:])): 
    num = 0
    denom = 0
    for j in range(len(x[:])):
        x_diff = (x[j][i] - torch.mean(x[:,i]))
        y_diff = (y[j] - y_mean)
        num +=  x_diff * y_diff 
        denom += x_diff**2
    betas.append(num/denom)
# fox is wrong?    

# %% Calculate b_0
sum = 0
for i,(b) in enumerate(betas):
    sum += b * torch.mean(x[:,i])
    
beta_0 = y_mean - sum    
print(beta_0)
    
# %% Calcuate
preds= []
pred = 0
 
for i in range(len(x[:])):
    for j in range(len(x[0,:])):
        pred += betas[j]*x[i][j]
        #print("b : %.4f, x: %.4f" %(b.item(),x[i][j]))
    
    pred = pred+beta_0
    print(pred)
    preds.append(pred)
    pred = 0
  
# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(11,5))
plt.plot(preds)
plt.plot(y)

plt.legend()
plt.show()
    
# %%
print(x_diff)
print(y_diff)
print(num)
print(denom)    
    


