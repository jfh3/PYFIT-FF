# PyFit-FF: 

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
  + First install "conda" on your machine (conda is a popular open source python package management system)
      - The conda installation process is well documented at the following link 
      - https://docs.conda.io/projects/conda/en/latest/user-guide/install/
        - for more information on conda see the following: https://www.youtube.com/watch?v=23aQdrS58e0&t=327s
          2) Use the following commands to create a conda environment with the Python 3.x and PyTorch (numpy will be installed automatically as part of this process) 
+ **Installation option-2 (manual):** 


## Dependencies installation option-1:    

On a linux machine the dependencies can be installed using 

sudo pip3 install torch torchvision numpy



### Downloading PyFit-FF  

Once the dependencies are met then PYFIT can be installed on a linux or Mac system using the following instructions 

1) Use the following command to get the PYFIT-FF source code from Github

  - git clone https://github.com/jfh3/PYFIT-FF
    - Alternatively you can  manually use the green "clone or download" button on https://github.com/jfh3/PYFIT-FF
  - This will make a directory called "PYFIT-FF" on your machine 

2) Use the following commands to make PYFIT executable from any directory on your machine  

  - mv PYFIT-FF ~/bin
    - cd ~/bin/
    - ln -s PYFIT-FF/src/pyfit.py  PYFIT 

Once these commands are run your system should automatically find the link to PYFIT in ~/bin/ and add it to the PATH which can then be called from the command line. Because pyfit.py has the line #!/usr/bin/env python3 it can be run like executable 

3) Run PYFIT using the provided example to get started 
	- cd PYFIT-FF/example/
	- PYFIT input.json 
	- Note: The command "PYFIT input.json" points the link in ~/bin/ and executes the program. An alternative is to run pyfit directly from the /src/ directory 
		* cd PYFIT-FF/src/
		* python3.7 pyfit.py input.json 

+ Additional comments 
  - This code has been tested on Linux and macOS systems, however, assuming the following dependencies are met for your python implementation then the code should run on windows OS as well.
  - Note technically PYFIT can be run directly from


## Dependencies installation option-2: conda (recommended)  



3) Use the following command to activate this conda environment on your local machine (

 - sfdsfdklglds
 - NOTE: I typically put this command in my ~/.bashrc so that it activates automatically and is essentally my default python implementation



## Running the example

