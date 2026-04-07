import random

SPLIT_SEED = 0
VAL_FRACTION = 0.1
MAX_VAL_SAMPLES = 5000

def random_tqdm_color():
    return f'#{random.randint(0, 0xFFFFFF):06x}'