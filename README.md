PyFit-FF: 

## Summary

+ __Authors__: James Hickman (NIST) and Adam Robinson (GMU) 
  * Please send any relevant questions, bug reports, or comments to james.hickman@nist.gov

+ __Description__: 
  - PyFit-FF  is a tool for training feed-forward artificial neural network (ANN) interatomic potentials to interpolate between density functional theory (DFT) energy predictions. The training is done by utilizing the automatic differentiation and optimization library [PyTorch](https://pytorch.org/) for the optimization process. The main benefits of PyFit-FF  are its portablility, simplicity, speed, flexibilty, and open source design. 

## Documentation

This README file only contains information on how to install the required dependencies, download the code, and run the included example. For more detailed information on what the code is doing and how it works please refer to the manual in (PYFIT-FF/docs/manual.pdf). 

+ __Manual contents__: 
  - Background information relevant to PyFit-FF 
    * Regression 
    * Neural network overview
    * Overview of machine learning potentials 
    * Atomic fingerprinting 
  - PyFit-FF  Overview
    * Outline of current functionality and future plans
    * PyFit-FF 
    * Description of PyFit-FF's input and outfiles
    * Description of PyFit-FF 's input parameters, their default values, and what they control
  - Getting started: 
    * Dependency installation instructions
    *  PyFit-FF download instructions 

## Installation

The following instructions can also be found in the file PYFIT-FF/docs/manual.pdf, however, for quick reference we include them here as well. 

### Operating systems

Currently PyFit-FF  has been tested and found to work efficently on macOS and Ubuntu-Linux. The code has not yet been tested on Windows, however, code should work provided the required dependancies are met. 

### Dependencies 

In order for the code to run the following  dependencies must be met:

-  [PyTorch](https://pytorch.org/)
- Python 3.x 
- Numpy

The following instructions are for Linux and Mac systems. Instructions for Windows machines will be added in the future. 

+ __Installation option-1 (recommended)__: 

  +  [Conda](https://docs.conda.io/en/latest/) is a popular open source software package management system. This method will create a "self-contained" Conda environment to install python3.7 and the various dependencies for PyFit. Because the environment is self containe  it will not effect your system's default python distribution and the currently installed python packages. 

  + **Step-1: Install Conda** 

      + The Conda installation process is well documented at the following link 
          + https://docs.conda.io/projects/conda/en/latest/user-guide/install/

  + **Step-2: Create a dedicated Conda environment** 

    + Executing the following commands from the command line will create a conda environment named TORCH3.7 and will install the required dependencies. Be sure to answer "yes" to the various prompts. The entire process should only take a few minutes and take up roughly 2.5 GB of disk space.

    + ```shell
      conda deactivate			      # exit current conda enviroment if one is activated
      conda create -n TORCH3.7 python=3.7     # create new conda enviroment named TORCH3.7
      conda activate TORCH3.7		      # activate the TORCH3.7 enviroment 
      conda install -c pytorch pytorch	      # install the pytorch in the TORCH3.7 enviroment 
      conda update --all && conda clean -all  # update and clean
      ```

    + **Note**: Alternatively you can use the following command to  install PyTorch without GPU functionality. This will take up slightly less space on the disk:  "conda install pytorch-cpu -c pytorch"

    + **Note**: NumPy will automatically be installed with the PyTorch installation 

  + **Useful Conda Commands** 

    ```shell
    #EXIT AN ACTIVATED CONDA ENVIROMENT: 
    # To exit the TORCH3.7 enviroment and return to your
    # "default" shell execute the following 
    conda deactivate
    
    #ACTIVATED AN ENVIROMENT: 
    #Once it is installed you can use the TORCH3.7 enviroment
    #any time in the future by simply activating it again 
    conda activate TORCH3.7	
    
    #USEFUL CONDA COMMANGE 
    conda update --all 				# UPDATE ENVIROMENT PACKAGES
    conda clean --all				# CLEAN UNNEEDED PACKAGES
    conda info --envs				# LIST ALL AVAILABLE ENVIROMENTS 
    conda remove --name TORCH3.7 --all		# PERMANTLY DELETE THE ENVIROMENT 
    
    ```

+ **Installation option-2 (manual):** 

  + **Note**: If you already have have python3.X installed you can install the PyTorch package using the following pip command. If Numpy is already install, this method may re-install Numpy to a different version which potentially could 'break' things.

    ```shell
    sudo pip3 install torch  numpy
    ```

### Downloading PyFit-FF   

Once the dependencies are met then PYFIT can be installed on a linux or Mac system using the following instructions 

**Step-1: Install Conda** 

Use the following command to get the PYFIT-FF source code from Github. This will make a directory called "PYFIT-FF" on your machine. 

```shell
git clone https://github.com/jfh3/PYFIT-FF
```

Alternatively you can manually download the folder using the green "clone or download" button on https://github.com/jfh3/PYFIT-FF

**Step-2: Make PyFit "executable" (optional)** 

Use the following commands to make PYFIT executable from any directory on your machine  

```shell
mv PYFIT-FF ~/bin
cd ~/bin/
ln -s PYFIT-FF/src/pyfit.py  pyfit 
```

Once these commands are run your system should automatically find the link to the "pyfit" file in ~/bin/ and add it to the PATH which can then be called from the command line. Because pyfit.py has the line #!/usr/bin/env python3 it can be run like executable 

## Running the example

3) Run PYFIT using the provided example to get started 

 - cd PYFIT-FF/example/
 - PYFIT input.json 
 - Note: The command "PYFIT input.json" points the link in ~/bin/ and executes the program. An alternative is to run pyfit directly from the /src/ directory 

  * cd PYFIT-FF/src/
     * python3.7 pyfit.py input.json 

+ Additional comments 
  - This code has been tested on Linux and macOS systems, however, assuming the following dependencies are met for your python implementation then the code should run on windows OS as well.
  - Note technically PYFIT can be run directly from


