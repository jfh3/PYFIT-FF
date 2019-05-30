# Name of this file, used for error messages.
CONFIG_FNAME                       = __file__

DEBUG = True

# Whether or not to warn when there are lines
# containing only whitespace found in the file.
WARN_ON_WHITESPACE_IN_TRAINING_SET = True

# Whether or not to automatically normalize line
# endings.
NORMALIZE_LINE_ENDINGS             = False

# File to use for writing log information.
LOG_PATH                           = 'output/log.txt'

# The structure file that contains POSCAR structures
# and DFT energies.
TRAINING_SET_FILE                  = 'input/LSParam-mod.dat'

# The neural network file to load.
NEURAL_NETWORK_FILE                = 'input/nn1.dat'


# The path to the file to output the results of this program
# into.
OUTPUT_NBL_PATH                    = 'Nbdlist.dat'

# Energy shift per atom for DFT (for pre-processing use 0)
E_SHIFT = 0.0 

# 1 = Straight Neural Network Function, 2 = BOP Potential
POTENTIAL_TYPE = 1

# The ratio of training data to overall amount of data.
TRAIN_TO_TOTAL_RATIO = 1.0 

# Contains values that indicate how heavily weighted each subgroup
# should be when computing the error.
WEIGHTS = {}
# Example: WEIGHTS['Si_B1']=1.0

# Standard NN learning rate.
LEARNING_RATE = 0.1

# Which torch.optim algorithm to use.
OPTIMIZATION_ALGORITHM = 'LBFGS'

# Maximum number of epochs to run through for training.
MAXIMUM_TRAINING_ITERATIONS = 10**5

