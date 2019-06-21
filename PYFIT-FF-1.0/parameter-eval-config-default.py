CONFIG_FNAME                       = __file__
FILE_BUFFERING                     = 8192
OBJECTIVE_FUNCTION                 = 'rmse'
LOG_PATH                           = 'log.txt'
LOSS_LOG_PATH                      = 'loss_log.txt'
VALIDATION_LOG_PATH                = 'validation_loss_log.txt'
E_VS_V_FILE                        = 'E_vs_V.txt'
OPTIMIZATION_ALGORITHM             = 'LBFGS'
MAX_LBFGS_ITERATIONS               = 10
E_VS_V_INTERVAL                    = 1000000


NEURAL_NETWORK_FILE      = ''
NEURAL_NETWORK_SAVE_FILE = 'out_nn.dat'

POSCAR_DATA_FILE  = ''
LSPARAM_FILE      = ''
TRAINING_SET_FILE = 'lsparam-generated.dat'
NEIGHBOR_FILE     = ''

DIV_BY_R0_SQUARED = True
E_SHIFT           = 0.795023

NETWORK_BACKUP_INTERVAL = 1000000
NETWORK_BACKUP_DIR      = 'nn_backup/'
KEEP_BACKUP_HISTORY     = True

VALIDATION_INTERVAL         = 10000
GROUP_WISE_VALIDATION_SPLIT = True
PROGRESS_INTERVAL           = 1
TRAIN_TO_TOTAL_RATIO        = 1.0
LEARNING_RATE               = 0.05
MAXIMUM_TRAINING_ITERATIONS = 500

FLAT_ERROR_STOP       = 1e-15
FLAT_ERROR_ITERATIONS = 1000
PLATEAU_ANNEALING_MAX_ITERATIONS = 0
PLATEAU_ANNEALING_LR_DECREMENT   = 0.01
PLATEAU_ANNEALING_RAND_STD       = 0.09


CHECK_OVERFIT_AFTER = 25 
OVERFIT_ERROR_STOP  = 2e5
OVERFIT_INCREASE_MAX_ITERATIONS = 10000

DEFAULT_WEIGHT = 1.0
WEIGHTS        = {}

UNWEIGHTED_NEGATIVE_ERROR   = False
GROUP_ERROR_FILE            = 'group_error.txt'
GROUP_ERROR_RECORD_INTERVAL = 1
DEFAULT_TARGET              = 0.014
SUBGROUP_TARGETS            = {}
SUBGROUP_ERROR_COEFFICIENT  = 1.0
