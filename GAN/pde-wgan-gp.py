# WGAN-GP to solve ODEs

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
        u = self.relu(self.l1(z))
        u = self.relu(self.l2(u))
        u = self.relu(self.l3(u))
        out = self.l4(u)
 
        return out

def u_true(x):
    u = x**2
    # u = np.sin(x)**2
    return u

def f_true(x):
    f = 2*x
    # f = np.sin(2.0*x)+2.0*x*np.cos(2*x)
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

def get_noise_tensor(size:int)->np.ndarray:
    noise = torch.randn(batch_size*1,1) # create some random noise
    noisev = torch.FloatTensor(noise).reshape(batch_size,1) # make noise into a tensor
    noisev.requires_grad=True

    return noisev


batch_size = 20
num_batches = 20
LAMBDA = 0.1

vxn  = 1000 # number of points
vx =np.linspace(0, np.pi, vxn)  # 2000 evenly spaced points between 0 and pi

btches = [] # will consist of random points from vx
real_data_batches = [] # will consist of random points x, u(x), and f(x)
for i in range(num_batches): # for i = 0 to 20
    b = random.choices(vx,k=batch_size) # 20 random points from vx with replacement
    bar = np.array(b) # turns b into array of 20 points
    ub = u_true(bar) 
    fb = f_true(bar) 
    ub0 = torch.FloatTensor(ub).reshape(batch_size,1)
    ub0.requires_grad=True 
    ib = torch.FloatTensor(b).reshape(batch_size,1)
    ifb = torch.FloatTensor(fb).reshape(batch_size,1) 
    ib.requires_grad=True 
    ifb.requires_grad = True 

    # Concatenates the given sequence of seq tensors in the given dimension. 
    # All tensors must either have the same shape (except in the concatenating dimension) or be empty.
    real_data_batches.append(torch.cat((ib, ub0, ifb),1)) 
    btches.append(ib) # adds ib (20 random points from vx with replacement) into a list


for i in range(num_batches):
    x = real_data_batches[i][:, 0].detach().numpy()
    y = real_data_batches[i][:, 1].detach().numpy()
    plt.scatter(x,y, color = 'b', s=5)

# the followning adapted from https://github.com/caogang/wgan-gp/blob/master/gan_toy.py 
# by Marvin Cao
def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(real_data.size())

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


dis = discriminator() # discriminator model
gen = generator() # generator model

# one = torch.FloatTensor([1])
# mone = one * -1
one = torch.ones(batch_size,1)
mone = -1 * torch.ones(batch_size,1)

num_epochs = 5000
critic_iter = 10
lr = 5e-5
betas = (0.0, 0.9)

optimizerD = optim.Adam(dis.parameters(), lr=lr, betas=betas)
optimizerG = optim.Adam(gen.parameters(), lr=lr, betas=betas)

for epoch in range(num_epochs): # for epoch in range 0 to 20000

    for _ in range(critic_iter):

        # Get real data
        i = random.randint(0,num_batches-1) # pick a random integer
        real_data = real_data_batches[i] # pick a real data point consisting of x, u(x), and f(x)

        # Train with real
        real_output = dis(real_data)
        # real_out = torch.mean(real_output,0, False)
        real_output.backward(mone)

        # Train with random noise
        noisev = get_noise_tensor(batch_size)
        gen_data  = gen(noisev) 
        fout = Du(noisev, gen_data[:,1])
        fake_data = torch.cat((gen_data, fout),1)

        fake_output = dis(fake_data)
        # fake_output = torch.mean(fake_output,0, False)
        # fake_output = fake_output.mean()
        fake_output.backward(one)

        # train with gradient penalty
        gradient_penalty = calc_gradient_penalty(dis, real_data, fake_data)
        gradient_penalty.backward()

        dis_loss = torch.mean(fake_output - real_output + gradient_penalty)
        # dis_loss = -torch.mean(real_output) + torch.mean(fake_output) + gradient_penalty
        # dis_loss.backward(retain_graph = True) 

        optimizerD.step()

        # Clear optimizers
        optimizerG.zero_grad()
        optimizerD.zero_grad()
        # gen.zero_grad()
        # dis.zero_grad()

    noisev = get_noise_tensor(batch_size)
    gen_data  = gen(noisev) # pass random noise with to generator
    fout = Du(noisev, gen_data[:,1])
    fake_data = torch.cat((gen_data, fout),1)
    fake_output = dis(fake_data)
    # fake_output = fake_output.mean()
    # fake_output = torch.mean(fake_output,0, False)
    fake_output.backward(mone)

    # Calculate G's loss based on this output
    gen_loss = -torch.mean(fake_output)
    # gen_loss.backward(retain_graph=True)

    # Update G
    optimizerG.step()
    gen.zero_grad()

    if epoch % 10 == 0:        
        print('Epoch: {}/{}; Critic_loss: {}; G_loss: {}' .format(epoch, num_epochs, dis_loss.data.numpy(), gen_loss.data.numpy()))
        # print('Iter-{}; D_loss: {}; G_loss: {}'.format(epoch, dis_loss.data.numpy(), gen_loss.data.numpy()), file=open('./wgan-out.txt','a'))

    # if epoch % 1000 == 0:
    #     torch.save(gen.state_dict(), 'wgan'+str(epoch))


for i in range(0,100):
    # noise = torch.randn(batch_size, 1) 
    noisev = autograd.Variable(get_noise_tensor(batch_size))  # totally freeze netG
    noisev.requires_grad=True
    fout = gen(noisev)
    z = fout.detach().numpy()

    for j in range(int(batch_size)):
        zx = z[j,0]
        zy = z[j,1]
        if -0.1 < zy < np.pi**2 and 0 <= zx <= np.pi:
            plt.scatter(zx, zy, c='orange', s=30)

plt.title("WGAN-GP-" + str(num_epochs))
# plt.savefig('./plots/wgan-gp/x-squared/wgan-gp-'+str(num_epochs)+'.png')
plt.show()