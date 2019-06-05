# Name of this file, used for error messages.
CONFIG_FNAME                       = __file__

# File to use for writing log information.
LOG_PATH                           = 'output/log.txt'

# The neural network file to load.
NEURAL_NETWORK_FILE                = 'input/new/nn1.dat'

# --------------------------------------------------
# Structural Parameter Calculation Configuration
# --------------------------------------------------

POSCAR_DATA_FILE = 'input/new/train.dat'

# The parameter file to output. This is what gets used for neural network
# training during the next step (usually). If you want the program to train
# on this file immediately after generating it, specify the same file for
# the TRAINING_SET_FILE parameter and pass the --run-training flag to the
# program.
LSPARAM_FILE     = 'output/generated.dat'

# The file to store training data and neighbors lists in. If you don't specify this
# it won't get written.
NEIGHBOR_FILE    = 'output/nbl.dat'

# Whether or not to divide gis by r0 squared before taking the natural logarithm.
DIV_BY_R0_SQUARED = True

# Energies put in training file = DFT + n_atoms*e_shift.
# This shifts DFT energies to experimental values.
E_SHIFT = 0.0


# --------------------------------------------------
# Neural Network Training Configuration
# --------------------------------------------------

# If the difference in the loss between each subsequent training iteration is
# less than FLAT_ERROR_STOP for FLAT_ERROR_ITERATIONS, the training will stop
# before reaching MAXIMUM_TRAINING_ITERATIONS.
FLAT_ERROR_STOP       = 1e-6
FLAT_ERROR_ITERATIONS = 4

# After 25 training iterations, if the difference between the training error and
# the validation error is greater than this, the training with stop before reaching
# MAXIMUM_TRAINING_ITERATIONS.
OVERFIT_ERROR_STOP = 2e-2

# The maximum number of times in a row that the difference between the validation
# error and the training error can increase.
OVERFIT_INCREASE_MAX_ITERATIONS = 5

# The directory to backup neural network files in at the
# interval specified below.
NETWORK_BACKUP_DIR                 = 'output/nn_backup/'

# Interval to backup the neural network file on.
NETWORK_BACKUP_INTERVAL            = 25

# If True, all backups are kept. If False, only the last backup 
# is kept.
KEEP_BACKUP_HISTORY                = True

# Network Loss Log File Path
LOSS_LOG_PATH                      = 'output/loss_log.txt'

# The file to log the validation loss in.
VALIDATION_LOG_PATH                = 'output/validation_loss_log.txt'

# Interval on which validation error should be calculated and
# logged in the corresponding file.
VALIDATION_INTERVAL = 5

# Update the progress bar every PROGRESS_INTERVAL epochs.
PROGRESS_INTERVAL = 2

# The structure file that contains POSCAR structures
# and DFT energies.
TRAINING_SET_FILE                  = 'output/generated.dat'

# Where to save the neural network when done training it.
NEURAL_NETWORK_SAVE_FILE           = 'output/nn1.dat'

# The file to store the E_VS_V data in.
# Each line will be all volumes in order followed
# immediately by all energies in order.
E_VS_V_FILE                        = 'output/E_VS_V.txt'

# Interval at which energy vs. volume data is exported.
E_VS_V_INTERVAL = 100

# Energy shift per atom for DFT (for pre-processing use 0)
E_SHIFT = 0.0 

# The ratio of training data to overall amount of data.
TRAIN_TO_TOTAL_RATIO = 0.85

# Contains values that indicate how heavily weighted each subgroup
# should be when computing the error.
WEIGHTS = {}
# Example: WEIGHTS['Si_B1']=1.0

# Standard NN learning rate.
LEARNING_RATE = 0.09

# Which torch.optim algorithm to use.
OPTIMIZATION_ALGORITHM = 'LBFGS'

# Maximum number of iterations for an LBFGS optimization step.
MAX_LBFGS_ITERATIONS = 10

# Maximum number of epochs to run through for training.
MAXIMUM_TRAINING_ITERATIONS = 50