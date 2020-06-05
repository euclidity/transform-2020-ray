# Transform 2020 

The full repo of code and notebooks will be published ahead on the event here. 
You'll just need to check back and pull the latest before the event. 

In the meantime there are a few things to read and a draft conda environment 
that you can get setup and maybe iron out any issues ahead of time.

### What to expect during the tutorial

Ray and it's libraries are meant for implementing scalable ML, that means big problems, big dataset an/or big models.
Ray is beautiful in that is speeds up small things too, and let's you start small and scale later with minimal code 
change. 

My overall aim for the tutorial is toget people excited about Ray and what it can do, and hopefully give you the 
feeling that using ray and following its patterns is a good way to start any ML project, and will give you superpowers
as your project matures.

We have got 3 hoours for the tutorial, so I have chosen small and simple problems. Logistics and limited experience
in presenting this as a tutorial means that while everyone shold be able to trun parts of this locally, some parts of 
this, like multi-gpu training or launching remote clusters may be a matter of watching/following along.

In order to be prepared I suggest:

 - Everyone have a go at environment setup
 - If you have a nvidia gpu on your system make sure it shows up in [nvidia-smi](https://developer.nvidia.com/nvidia-system-management-interface)
 - If you have an AWS account up and running: 
    - make sure your local machine is setup with your access credentials [link](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html)
    - identify an [AWS region](https://docs.aws.amazon.com/general/latest/gr/rande.html) e.g. Ohio  where you have the ability to 
    run instances and make a note of it's code e.g. `us-east-2`
 - If you don't have an AWS account, consider setting one up. During the tutorial I aim to run a small cluster of C5.xlarge instances which are $0.19/hr so AWS cos won;t be large


## Environment Setup

There is an extra step in configuring your local environment in order to get the Ray Dashboard setup.
If you cannot get this working for whatever reason it is not the end of the world, but it is nice to have and use
so let's try.

Do this, this **before** setting up the conda environment

 1. First install Node.js either:
    1. using nvm [link](https://gist.github.com/d2s/372b5943bce17b964a79) or 
    1. by downloading the latest version and installing manually [link](https://nodejs.org/en/download/)
    
 1. Install and build dashboard from source
 
        git clone https://github.com/ray-project/ray.git

        pushd ray/python/ray/dashboard/client
        npm ci
        npm run build
        popd

 1. Build the conda environment:

        pushd transform-2020-ray
        conda env create -f environment.yml


If this doesn't work for you and you want to forgo the Dashboard install locally then just install the conda 
environment as per normal

    cd transform-2020-ray
    conda env create -f environment.yml
    
#### Gotchas

 - if you are on a machine without a GPU, don't worry you'll still be able to run some of the notebooks. However,
  you might need to comment out the ` - cudatoolkit >= 10.0` line from `environment.yml`
  

## Smoke Test

 1. After you have your environment, activate it
 
        conda activate t20-fri-ray
        
 1. Run the jupyter notebook (or lab if you prefer)
   
        jupyter notebook
        
 1. In jupyter, open `pinch_me.ipynb` and run all cells. if all is ok everything should load.
 

#### Gotchas

 - XGBoost Errors on run
 
       XGBoostError: XGBoost Library (libxgboost.dylib) could not be loaded.
        Likely causes:
          * OpenMP runtime is not installed (vcomp140.dll or libgomp-1.dll for Windows, libomp.dylib for Mac OSX, libgomp.so for Linux and other UNIX-like OSes). Mac OSX users: Run `brew install libomp` to install OpenMP runtime.
          * You are running 32-bit Python on a 64-bit OS
          
          
 - install on macos `brew install libomp`
 - install on Ubuntu https://askubuntu.com/questions/144352/how-can-i-install-openmp-in-ubuntu
 
       