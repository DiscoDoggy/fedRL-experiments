

import torch, numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()


logging.info(f"CUDA avail: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logging.info(f"Devices: {torch.cuda.device_count()} current: {torch.cuda.current_device()}")
    free, total = torch.cuda.mem_get_info()
    logging.info(f"GPU mem free/total (MB): {free/1024**2:.1f}/{total/1024**2:.1f}")