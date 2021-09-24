# WGAN-GP to solve ODEs

import numpy as np
import scipy
from scipy import spatial
import matplotlib
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
torch.autograd.set_detect_anomaly(True)

matplotlib.use("Agg")

# Check if GPU is available 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Directory to save plots
save_directory = './running-plots/'

# defining generator class
class generator(nn.Module):
    
    def __init__(self):
        
        super(generator, self).__init__()

        self.main = main = nn.Sequential(
            nn.Linear(1, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 2)
        )
        
    def forward(self, x):
        x = x.to(device=device)
        output = self.main(x)

        return output


class discriminator(nn.Module):
    
    def __init__(self):
        
        super(discriminator, self).__init__()

        self.main = main = nn.Sequential(
            nn.Linear(3, 1000),
            nn.LeakyReLU(),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(),
            nn.Linear(1000, 1),
            # nn.Sigmoid()
        )
        
    def forward(self, z):
        z = z.to(device=device)
        output = self.main(z)

        return output



def u_true(x):
    u = x**2
    # u = np.sin(x)**2
    return u

def f_true(x):
    f = 4*x
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


def get_noise_tensor(size:int)-> np.ndarray:
    ''' Creates a tensor of random points from an normal distribution'''

    # Gets random numbers from a normal distribution N~(0,1)
    # noise = torch.randn(batch_size*1,1) # create some random noise

    # Freezes generator
    # noisev = autograd.Variable(noise)

    # Gets random variables from a uniform distribution with boundaries and freezes generator
    noisev = autograd.Variable(torch.FloatTensor(batch_size*1,1).uniform_(left_bndry, right_bndry))
    noisev.requires_grad=True

    return noisev


def check_directory_else_make(dir: str)-> None:
    '''
        Checks if the directory exists.
        Creates the directory if it doesn't.
    '''
    if not os.path.exists(dir):
        os.makedirs(dir)


def make_plot(real_data, fdata, num_batches, epoch, num_epochs, noise)-> None:
    ''' Plots the real and generated data points'''

    # plot the real data
    for i in range(num_batches):
        x = real_data_batches[i][:, 0].detach().numpy()
        y = real_data_batches[i][:, 1].detach().numpy()
        plt.scatter(x,y, color = 'b', s=5)

    # plot the generated data
    for index, value in enumerate(fdata[:,1]):
        if -0.1 < value < 1.1:

            # plots noise as x-value vs. generated u-value
            plt.scatter(noise[index], value, c='orange', s=5)

            # plots generated x-value vs. generated u-value
            plt.scatter(fdata[index,0], value, c='green', s=5)



    # Make the directory to save plots if it doesn't exits
    check_directory_else_make(save_directory)
    
    # save plot to directory
    plt.title("WGAN-GP-" + str(num_epochs))
    plt.savefig(save_directory+str(num_epochs)+'-iter-'+str(epoch))
    plt.figure().clear()


batch_size = 20
num_batches = 20

LAMBDA = 10 # The gradient penalty coefficient 

vxn  = 10000 # number of points
left_bndry = -1
right_bndry = 1
vx =np.linspace(left_bndry, right_bndry, vxn)  # creates evenly spaces points

real_data_batches = [] # will consist of random points x, u(x), and f(x)

# Creates batches of real data points with the form x, u(x), and f(x)
for i in range(num_batches): # for i = 0 to 20
    b = np.array(random.choices(vx,k=batch_size)) # 20 random points from vx with replacement
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


# the followning adapted from https://github.com/caogang/wgan-gp/blob/master/gan_toy.py 
# by Marvin Cao
def calc_gradient_penalty(netD, real_data, fake_data):
    ''' Calculates the gradient penalty.

        netD: the discriminator/critic model
        real_data: real data points
        fake_data: fake data points
    
    '''
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(real_data.size()).to(device=device)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(
                                  disc_interpolates.size()).to(device=device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    # tensor.norm is depricated and might be removed from future releases.
    # Need to update this next line to use torch.linalg
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


num_epochs = 20000

# Number of times to train the discriminator/ critic before training the generator
critic_iter = 10 

# learning rate
lr = 5e-5

# decay rates for adaptive gradient algorithm and RMS prop
betas = (0.0, 0.2)

# discriminator model
dis = discriminator().to(device=device) 

 # generator model
gen = generator().to(device=device)

# optimizer for the discriminator/critic
optimizerD = optim.Adam(dis.parameters(), lr=lr, betas=betas)

# optimizer for the generator
optimizerG = optim.Adam(gen.parameters(), lr=lr, betas=betas)

# negative_critic_losses = []

# Trains the model 
for epoch in range(num_epochs):

    for _ in range(critic_iter):
        ################################
        ## Train Discriminator/Critic ##
        ################################
        
        # Get real data
        i = random.randint(0,num_batches-1) # pick a random integer
        real_data = real_data_batches[i].to(device=device) # pick a real data point consisting of x, u(x), and f(x)
        
        # Train with random noise
        noisev = get_noise_tensor(batch_size)
        
        # Clear gradient for generator
        gen.zero_grad()
        gen_data  = gen(noisev)

        fout = Du(noisev, gen_data[:,1]).to(device=device)
        fake_data = torch.cat((gen_data, fout),1).to(device=device)
        fake_output = dis(fake_data)

        # Clear gradient for discriminator/critic
        dis.zero_grad()

        # Train with real
        real_output = dis(real_data)
        
        # Calculate gradient penalty
        gradient_penalty = calc_gradient_penalty(dis, real_data, fake_data)

        # gradient penalty is already averaged in the function calc_gradient_penalty()
        dis_loss = fake_output - real_output + gradient_penalty
        dis_loss = dis_loss.mean()
        dis_loss.backward() 


        optimizerD.step()


    #####################
    ## Train Generator ##
    #####################

    # Clear gradient
    gen.zero_grad() 
    noisev = get_noise_tensor(batch_size)
    gen_data  = gen(noisev) # pass random noise with to generator
    fout = Du(noisev, gen_data[:,1]).to(device=device)
    fake_data = torch.cat((gen_data, fout),1)
    fake_output = dis(fake_data)

    # Calculate G's loss based on this output
    gen_loss = -fake_output.mean()
    gen_loss.backward()

    # Update G
    optimizerG.step()

    # negative_critic_losses.append(-dis_loss.data.numpy())

    if (epoch+1) % 10 == 0 or epoch == num_epochs-1:        
        print('Epoch: {}/{}; Critic_loss: {}; G_loss: {}' 
            .format(epoch+1, num_epochs, dis_loss.to(device='cpu').data.numpy(), gen_loss.to(device='cpu').data.numpy()))
        # print('Iter-{}; D_loss: {}; G_loss: {}'.format(epoch, dis_loss.data.numpy(), gen_loss.data.numpy()), file=open('./wgan-out.txt','a'))

    # Make plots every few iterations.  Originally had it at every 10 iterations.
    if (epoch+1) % 10 == 0 or epoch == num_epochs-1:   
        # Create noise and generated data to plot
        stacked_noise = np.empty([0,0])
        stacked_data = np.empty([0,0])
        for i in range(batch_size):
            noisev = get_noise_tensor(batch_size)
            noisev.requires_grad=True
            fout = gen(noisev)
            z = fout.to(device='cpu').detach().numpy()

            stacked_noise = noisev.detach().numpy() if i == 0 else np.vstack((stacked_noise, noisev.detach().numpy()))
            stacked_data = z if i == 0 else np.vstack((stacked_data, z))
  
        make_plot(real_data_batches, stacked_data, num_batches, epoch+1, num_epochs, stacked_noise)