# PyFit-FF

## Summary

- **Authors**: James Hickman (NIST) and Adam Robinson (GMU)
  - Please send any relevant questions, bug reports, or comments to [james.hickman@nist.gov](mailto:james.hickman@nist.gov)
- **Description**:
  - PyFit-FF is a tool for training feed-forward artificial neural network (ANN) interatomic potentials to interpolate between density functional theory (DFT) energy predictions. The training is done using the automatic differentiation and optimization library [PyTorch](https://pytorch.org/) for the optimization process. The main benefits of PyFit-FF are its portablility, simplicity, speed, flexibilty, and open source design.

## Documentation

This README file only contains information on how to install the required dependencies, download the code, and run the included example. For more detailed information on what the code is doing and how it works please refer to the manual in (PYFIT-FF/docs/manual.pdf).

- Manual contents
  - Background information relevant to PyFit-FF
    - Regression
    - Neural network overview
    - Overview of machine learning potentials
    - Atomic fingerprinting
  - PyFit-FF Overview
    - Outline of current functionality and future plans
    - PyFit-FF outline 
    - Description of PyFit-FF's input and output files
    - Description of PyFit-FF 's input parameters, their default values, and what they control
  - Getting started:
    - Dependency installation instructions
    - PyFit-FF download instructions
    - How to run the code

## Installation

The following instructions can also be found in the file PYFIT-FF/docs/manual.pdf, however, for quick reference we include them here as well.

### Operating systems

Currently PyFit-FF has been tested and found to work efficiently on *macOS and Ubuntu-Linux*. The code has not yet been tested on Windows, however, it should work provided the required dependancies are met.

### Dependencies

The following dependencies are required:

- [PyTorch](https://pytorch.org/)
- Python 3.x
- NumPy

The  instructions below are for Linux and Mac systems. Instructions for Windows machines will be added in the future.

- **Installation option-1 (recommended)**:

  - [Conda](https://docs.conda.io/en/latest/) is a popular open source software package management system. The following method will create a "self-contained" Conda environment to install python3.7 and the various dependencies for PyFit. Because the environment is self contained it will not effect your system's default python distribution.

  - **Step-1: Install Conda**

    - The Conda installation process is well documented at the following link
      - https://docs.conda.io/projects/conda/en/latest/user-guide/install/

  - **Step-2: Create a dedicated Conda environment for PyFit**

    - Executing the following commands from the command line will create a conda environment named TORCH3.7 and will install the required dependencies. Be sure to answer "yes" to the various prompts. The entire process should only take a few minutes and take up roughly 2.5 GB of disk space.

    - ```shell
      # exit current conda enviroment if one is activated
      conda deactivate
      # create new conda enviroment named TORCH3.7
      conda create -n TORCH3.7 python=3.7
      # activate the TORCH3.7 enviroment
      conda activate TORCH3.7
      # install the pytorch in the TORCH3.7 enviroment 
      conda install -c pytorch pytorch
      # update and clean
      conda update --all && conda clean -all
      ```

    - **Note**: Alternatively you can use the following command to install PyTorch without GPU functionality. This will take up slightly less space on the disk: "conda install pytorch-cpu -c pytorch"

    - **Note**: NumPy will automatically be installed during the PyTorch installation

  - **Useful Conda Commands**

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

- **Installation option-2 (manual):**

  - **Note**: If you already have have python3.X installed on your system then you can install the PyTorch package using the following pip command. Use caution, if NumPy is already installed, this method will likely re-install NumPy to a different version which could potentially  'break' things.

    ```
    sudo pip3 install torch numpy
    ```

### Downloading PyFit-FF

Once the dependencies are met then PYFIT can be downloaded from Github using the following command. This will make a directory called "PYFIT-FF" on your machine.

```
git clone https://github.com/jfh3/PYFIT-FF
```

Alternatively you can manually download the folder using the green "code" button on https://github.com/jfh3/PYFIT-FF and clicking "Download Zip"

## Running the code

PyFit is simply a collection of python scripts and therefore it can be run just like any other python code. However, the primary  script is pyfit.py. This script runs the program and calls the other subroutines. The first line of pyfit.py is ""#!/usr/bin/env python3". This line allows the file to run like an executable from anywhere on your computer as long as you point to it. The code will run and output files from inside the directory from which it was called but will also "see" and utilize the various subroutines in PYFIT-FF/src.  The code does require a single input json file. This input file is discussed in detail in the "PYFIT-FF/doc/PYFIT-Manual.pdf " file.  When you run PyFit you need to include the path to the input file as a command line argument (see below). 

**Option-1 (recommended):**

To make PyFit accessible from anywhere on your computer simply find a permanent location for the PYFIT-FF directory and put a link to the pyfit.py file in ~/bin/. This will make the file part of your system's PATH and allow you to easily access it. I typically keep the PYFIT-FF directory in ~/bin/ and create the link using the following commands:

```
mv PYFIT-FF ~/bin
cd ~/bin/
ln -s PYFIT-FF/src/pyfit.py  pyfit 
```

To run the example, simply navigate to "PYFIT-FF/examples/MNN" and run the following commands

```
cd PYFIT-FF/examples/MNN
pyfit input.json
```

Notice that the command "pyfit input.json" points to the link in ~/bin/pyfit and executes the program. If everything is working the code should start and begin to write output to the screen and to various log files. 

**Option-2:**

Alternatively if you're just testing PyFit-FF then you can simply navigate to the example directory and point to the pyfit.py script using the relative path "../../src/pyfit.py".  

```
cd PYFIT-FF/examples/MNN
../../src/pyfit.py input.json
```

These commands are equivalent to the more familiar 

```
cd PYFIT-FF/examples/MNN
python3.7 ../../src/pyfit.py input.json 
```

## Removing PyFit

To permanently remove PyFit from your system simply delete the Conda environment and the code directory.

```
rm -rf PYFIT-FF
conda remove --name TORCH3.7 --all		# PERMANTLY DELETE THE ENVIROMENT 
```
