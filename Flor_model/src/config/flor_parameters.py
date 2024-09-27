# ./config/flor_parameters.py

import string


class HTRFlorConfig:
    BATCH_SIZE = 16  # Adjust based on your GPU memory
    LEARNING_RATE = 0.001
    EPOCHS = 1000
    INPUT_SIZE = (1024, 128, 1)  # Input dimensions for the images
    MAX_TEXT_LENGTH = 128  # Maximum length of the text labels
    VOCAB_SIZE = 256  # Adjust according to your dataset's vocabulary size
    BEAM_WIDTH = 10  # Beam search width for decoding
    STOP_TOLERANCE = 20  # Early stopping tolerance
    REDUCE_TOLERANCE = 15  # Reduce learning rate on plateau tolerance
    REDUCE_FACTOR = 0.1  # Factor by which learning rate will be reduced
    REDUCE_COOLDOWN = 0  # Learning rate cooldown after reduction
    CHECKPOINT_DIR = "checkpoints"  # Directory to store model checkpoints
    LOG_DIR = "logs"  # Directory to store training logs
    CHARSET_BASE = string.printable[:95]
