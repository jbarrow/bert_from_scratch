
MAX_SEQ_LENGTH = 128 #@param {type:"integer"}
MASKED_LM_PROB = 0.15 #@param
MAX_PREDICTIONS = 20 #@param {type:"integer"}
DO_LOWER_CASE = True #@param {type:"boolean"}

PRETRAINING_DIR = "pretraining_data" #@param {type:"string"}
# controls how many parallel processes xargs can create
PROCESSES = 2 #@param {type:"integer"}



BUCKET_NAME = "contractbert-resources" #@param {type:"string"}
MODEL_DIR = "contractbert-128" #@param {type:"string"}
tf.gfile.MkDir(MODEL_DIR)

if not BUCKET_NAME:
  log.warning("WARNING: BUCKET_NAME is not set. "
              "You will not be able to train the model.")


BUCKET_NAME = "contractbert-resources" #@param {type:"string"}
MODEL_DIR = "contractbert-128" #@param {type:"string"}
PRETRAINING_DIR = "pretraining_data" #@param {type:"string"}
VOC_FNAME = "vocab.txt" #@param {type:"string"}

# Input data pipeline config
TRAIN_BATCH_SIZE = 128 #@param {type:"integer"}
MAX_PREDICTIONS = 20 #@param {type:"integer"}
MAX_SEQ_LENGTH = 128 #@param {type:"integer"}
MASKED_LM_PROB = 0.15 #@param

# Training procedure config
EVAL_BATCH_SIZE = 64
LEARNING_RATE = 2e-5
TRAIN_STEPS = 1000000 #@param {type:"integer"}
SAVE_CHECKPOINTS_STEPS = 2500 #@param {type:"integer"}
NUM_TPU_CORES = 8

if BUCKET_NAME:
  BUCKET_PATH = "gs://{}".format(BUCKET_NAME)
else:
  BUCKET_PATH = "."
