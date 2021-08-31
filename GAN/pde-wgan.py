# This is the GenerativeAdesarialNetwork GAN for solving 
# the non-linear stochastic PDEs.
# actually this example is not full stochastic because it uses the solution
# of a non-linear PDE so we can check output.

import numpy as np
import scipy
from scipy import spatial
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import os
import time
import random
from torch import nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import grad
# from utils.tensorboard_logger import Logger


torch.manual_seed(1)

# defining generator class
class generator(nn.Module):
    
    def __init__(self):
        
        super(generator, self).__init__()
        self.l1 = nn.Linear(1,300)
        self.l2 = nn.Linear(300,1000)
        self.l3 = nn.Linear(1000,800)
        self.l4 = nn.Linear(800,2)
        self.rl = nn.Tanh()
       
        
    def forward(self, x):
        z = self.rl(self.l1(x))
        u = self.rl(self.l2(z))
        u = self.rl(self.l3(u))
        z = self.l4(u)
        return z


class discriminator(nn.Module):
    
    def __init__(self):
        
        super(discriminator, self).__init__()
        self.l1 = nn.Linear(3,300)
        self.relu = nn.LeakyReLU()
        self.l2 = nn.Linear(300,300)
        self.l3 = nn.Linear(300,200)
        self.l4  = nn.Linear(200,1)
        
    def forward(self, z):
        #z = torch.cat((x, y),1)
        u = self.relu(self.l1(z))
        u = self.relu(self.l2(u))
        u = self.relu(self.l3(u))
        out = self.l4(u)
 
        return out


def wasserstein_loss(real, fake):
    return -(torch.mean(real) - torch.mean(fake))

def u_true(x):
    u = np.cos(x)
    return u

def f_true(x):
    f = -np.sin(x)
    return f

def flat(x):
    m = x.shape[0]
    return [x[i] for i in range(m)]

def Du(x,u):
    ''' Represents the differential operator:
        d    //     du  \\
       ---- || (x) ---- ||
        dx   \\     dx  //
    '''
    u_x = grad(flat(u), x, create_graph=True, allow_unused=True)[0] #nth_derivative(flat(u), wrt=x, n=1)
    xv =x[:,0].reshape(batch_size,1)
    z = u_x*xv
    u_xx = grad(flat(z), x, create_graph=True, allow_unused=True)[0] #nth_derivative(flat(u), wrt=x, n=1)
    f = u_xx

    return f

batch_size = 20
num_batches = 20

vxn  = 1000 # number of points
vx =np.linspace(0, 2*np.pi, vxn)  # 2000 evenly spaced points between 0 and pi
ix = torch.FloatTensor(vx).reshape(vxn,1) # [2000 x 1], this looks like dead code
btches = [] # will consist of random points from vx
ubtches = [] # will consist of random points x and u(x)
real_data_batches = [] # will consist of random points x, u(x), and f(x)
for i in range(num_batches): # for i = 0 to 20
    b = random.choices(vx,k=batch_size) # 20 random points from vx with replacement
    bar = np.array(b) # turns b into array of 20 points
    ub = u_true(bar) # true value for u(x) passes 20 points to sin(x)**2
    fb = f_true(bar) # true value for f(x) passes 20 points to sin(2.0 * x) + 2.0 * x * cos(2 * x)
    ub0 = torch.FloatTensor(ub).reshape(batch_size,1) # turns ub (true values for u(x)) into a [20 x 1]
    ub0.requires_grad=True # 
    ib = torch.FloatTensor(b).reshape(batch_size,1) # turns b (20 random points from vx with replacement) into a [20 x 1]
    ifb = torch.FloatTensor(fb).reshape(batch_size,1) # turns fb (true value for f(x)) into a [20 x 1]
    ib.requires_grad=True # 
    ifb.requires_grad = True #

    # Concatenates the given sequence of seq tensors in the given dimension. 
    # All tensors must either have the same shape (except in the concatenating dimension) or be empty.
    real_data_batches.append(torch.cat((ib, ub0, ifb),1)) 
    btches.append(ib) # adds ib into a list
    ubtches.append(torch.cat((ib, ub0),1)) # 


for i in range(num_batches):
    x = real_data_batches[i][:, 0].detach().numpy()
    y = real_data_batches[i][:, 1].detach().numpy()
    plt.scatter(x,y, color = 'b', s=5)


dis = discriminator() # discriminator model
gen = generator() # generator model

# one = torch.FloatTensor([1])
# mone = one * -1
one = torch.ones(batch_size,1)
mone = -1 * torch.ones(batch_size,1)

num_epochs = 100
critic_iter = 10
lr = 5e-5
clip_value = 0.1

optimizerD = optim.RMSprop(dis.parameters(), lr=lr)
optimizerG = optim.RMSprop(gen.parameters(), lr=lr)

for epoch in range(num_epochs):

    for _ in range(critic_iter):

        # Get real data
        i = random.randint(0,num_batches-1) # pick a random integer
        real_data = real_data_batches[i] # pick a real data point consisting of x, u(x), and f(x)

        # Make fake data
        noise = torch.randn(batch_size*1,1) # create some random noise
        noisev = torch.FloatTensor(noise).reshape(batch_size,1)
        noisev.requires_grad=True

        gen_data  = gen(noisev) # pass random noise with shape to generator
        fout = Du(noisev, gen_data[:,1])
        fake_data = torch.cat((gen_data, fout),1)

        fake_output = dis(fake_data)
        fake_output.backward(one, retain_graph = True)

        real_out = dis(real_data)
        real_out.backward(mone, retain_graph = True)
        
        # Update D        
        dis_loss = wasserstein_loss(real_out, fake_output)

        optimizerD.step()

        # Weight clipping
        for p in dis.parameters():
            p.data.clamp_(-clip_value, clip_value)

        # Clear optimizers
        gen.zero_grad()
        dis.zero_grad()

    noise = torch.randn(batch_size*1,1) # create some random noise
    noisev = torch.FloatTensor(noise).reshape(batch_size,1) 
    noisev.requires_grad=True

    gen_data  = gen(noisev) 
    fout = Du(noisev, gen_data[:,1])
    fake_data = torch.cat((gen_data, fout),1)
    fake_output = dis(fake_data)
    fake_output.backward(mone)

    # Calculate G's loss based on this output
    gen_loss = -torch.mean(fake_output)

    # Update G
    optimizerG.step()
    gen.zero_grad()

    if epoch % 10 == 0:        
        print('Epoch: {}/{}; Critic_loss: {}; Generator_loss: {}' .format(epoch, num_epochs, dis_loss.data.numpy(), gen_loss.data.numpy()))
        # print('Epoch: {}/{}; Critic_loss: {}; Generator_loss: {}' .format(epoch, num_epochs, dis_loss.data.numpy(), gen_loss.data.numpy()), file=open('./wgan-out.txt','a'))

    # if epoch % 1000 == 0:
    #     torch.save(gen.state_dict(), 'wgan'+str(epoch))


for i in range(0,num_batches):
    noise = torch.randn(batch_size, 1) 
    noisev = autograd.Variable(noise)  # totally freeze netG
    noisev.requires_grad=True
    fout = gen(noisev)
    z = fout.detach().numpy()

    for j in range(int(batch_size)):
        zx = z[j,0]
        zy = z[j,1]
        if -0.1 < zy < 1.2 and 0 <= zx <= 2*np.pi:
            plt.scatter(zx, zy, c='orange', s=30)

plt.title("WGAN-" + str(num_epochs))
# plt.savefig('./plots/this-is-a-saved-plot.png')
plt.show()