# Name of this file, used for error messages.
CONFIG_FNAME                       = __file__

# Use this to change the file write buffer size.
# This is useful if you are piping stdout into a file
# and you want to check progress during the runtime of
# the program.
#
#  0 = No buffer, probably won't work (requires binary mode)
#  1 = line buffer, should probably use this
# >1 = buffer size, default is usually 4096 - 8192
# See: https://docs.python.org/3/library/functions.html#open
FILE_BUFFERING                     = 8192*32

# File to use for writing log information.
LOG_PATH                           = 'log.txt'

# The neural network file to load.
NEURAL_NETWORK_FILE                = 'input/AB/nn1-60-gi-shifted.dat'

# --------------------------------------------------
# Structural Parameter Calculation Configuration
# --------------------------------------------------

POSCAR_DATA_FILE = 'input/AB/AB-POSCAR-E-full-NO-B00-B01-B02.dat'

# The parameter file to output. This is what gets used for neural network
# training during the next step (usually). If you want the program to train
# on this file immediately after generating it, specify the same file for
# the TRAINING_SET_FILE parameter and pass the --run-training flag to the
# program.
LSPARAM_FILE     = ''

# The file to store training data and neighbors lists in. If you don't specify this
# it won't get written.
NEIGHBOR_FILE    = ''

# Whether or not to divide gis by r0 squared before taking the natural logarithm.
DIV_BY_R0_SQUARED = True

# Energies put in training file = DFT + n_atoms*e_shift.
# This shifts DFT energies to experimental values.
E_SHIFT = 0.795023


# --------------------------------------------------
# Neural Network Training Configuration
# --------------------------------------------------

# If the difference in the loss between each subsequent training iteration is
# less than FLAT_ERROR_STOP for FLAT_ERROR_ITERATIONS, the training will stop
# before reaching MAXIMUM_TRAINING_ITERATIONS.
FLAT_ERROR_STOP       = 1e-7
FLAT_ERROR_ITERATIONS = 10

# When a plateau in the error is reached, based on the FLAT_ERROR_STOP and
# FLAT_ERROR_ITERATIONS conditions, this is the maximum number of times to
# perform an annealing step. In this context, that means adding a small random
# number to all weights in order to temporarily increase error, but potentially
# break out of a local minima. 
PLATEAU_ANNEALING_MAX_ITERATIONS = 0

# This is the amount to reduce the learning rate by each time a 
# plateau in the error is reached.
PLATEAU_ANNEALING_LR_DECREMENT = 0.01

# This is the standard deviation of the normally distributed random values to add.
# The program will generate numbers normally distributed around zero and add
# them onto the weights. This number is multiplied by of each
# network parameter to determine the actually STD to use. 
# Example: if PLATEAU_ANNEALING_RAND_STD = 0.25,
#          numbers will be np.random.normal(0.0, 0.25 * np.abs(nn_parameters[i]))
PLATEAU_ANNEALING_RAND_STD = 0.09

# The number of iterations after which the system should start checking
# for a significant difference between training error and validation error.
CHECK_OVERFIT_AFTER = 25

# After CHECK_OVERFIT_AFTER training iterations, if the difference between the training error and
# the validation error is greater than this, the training with stop before reaching
# MAXIMUM_TRAINING_ITERATIONS.
OVERFIT_ERROR_STOP = 2e5

# The maximum number of times in a row that the difference between the validation
# error and the training error can increase.
OVERFIT_INCREASE_MAX_ITERATIONS = 10000

# The directory to backup neural network files in at the
# interval specified below.
NETWORK_BACKUP_DIR                 = 'nn_backup/'

# Interval to backup the neural network file on.
NETWORK_BACKUP_INTERVAL            = 10000

# If True, all backups are kept. If False, only the last backup 
# is kept.
KEEP_BACKUP_HISTORY                = False

# Network Loss Log File Path
LOSS_LOG_PATH                      = 'output_enki_GPU_8_core_2500_iter_run_2/out/loss_log.txt'

# The file to log the validation loss in.
VALIDATION_LOG_PATH                = 'output_enki_GPU_8_core_2500_iter_run_2/out/validation_loss_log.txt'

# Interval on which validation error should be calculated and
# logged in the corresponding file.
VALIDATION_INTERVAL = 5

# Whether or not to ensure that the validation set is sampled
# equally for every structural group. This prevents the random
# selection of validation data from missing too much of one
# group. 
GROUP_WISE_VALIDATION_SPLIT = True

# Update the progress bar every PROGRESS_INTERVAL epochs.
PROGRESS_INTERVAL = 1

# The structure file that contains POSCAR structures
# and DFT energies.
TRAINING_SET_FILE                  = 'input/AB/AB-LSPARAM-E-full-NO-B00-B01-B02.dat'

# Where to save the neural network when done training it.
NEURAL_NETWORK_SAVE_FILE           = 'nn1.dat'

# The file to store the E_VS_V data in.
# Each line will be all volumes in order followed
# immediately by all energies in order.
E_VS_V_FILE                        = 'E_vs_V.txt'

# Interval at which energy vs. volume data is exported.
E_VS_V_INTERVAL = 10000000

# The ratio of training data to overall amount of data.
TRAIN_TO_TOTAL_RATIO = 1.0


# The weight to assign to any group not explicitely enumerated
# in WEIGHTS
DEFAULT_WEIGHT = 1.0

# Contains values that indicate how heavily weighted each subgroup
# should be when computing the error.
WEIGHTS = {}
# Example: WEIGHTS['Si_B1']=1.0

# This is the means by which error is calculated. 'rmse' is the standard,
# but 'group-targets' allows you to specify an error target for each 
# structural group if desired. This will allow you to train your network
# to very high accuracy in a particular subgroup, while not caring muchx
# about others.
OBJECTIVE_FUNCTION = 'rmse'

# If set to true, the system will score groups below their target error as
# having no error at all. This will allow groups to migrate down below 
# their target error if training of another group drags them down.
UNWEIGHTED_NEGATIVE_ERROR = False

# The file to store the subgroup error in. If you specify --group-error,
# this will be graphed after the main error graph so that you can see how the 
# error of each group varied throughout the training process.
GROUP_ERROR_FILE = 'group_error.txt'

# This is how often the error per-group should be recorded.
GROUP_ERROR_RECORD_INTERVAL = 1

# This is the default rmse value to target for subgroups if
# OBJECTIVE_FUNCTION = 'group-targets'. This is overriden by 
# any values explicitely specified in SUBGROUP_TARGETS.
DEFAULT_TARGET = 0.014

# The rmse target for each subgroup. This is only used if OBJECTIVE_FUNCTION = 'group-targets'
SUBGROUP_TARGETS = {}
# Example: SUBGROUP_TARGETS['Si_B1']= 0.004

# This is multiplied by the subgroup error at the end. This effectively
# makes the parabola that defines the error for each group steaper
# or more shallow.
SUBGROUP_ERROR_COEFFICIENT = 1.0

# Standard NN learning rate.
LEARNING_RATE = 0.05

# Which torch.optim algorithm to use. Currently this is just an
# if statement that only does SGD and LBFGS.
OPTIMIZATION_ALGORITHM = 'LBFGS'

# Maximum number of iterations for an LBFGS optimization step.
MAX_LBFGS_ITERATIONS = 10

# Maximum number of epochs to run through for training.
MAXIMUM_TRAINING_ITERATIONS = 500