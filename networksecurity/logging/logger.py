import logging
import os

from datetime import datetime

LOG_FILE=f"{datetime.now().strftime('%m-%d-%Y_%H-%M-%S')}.log"

logs_path=os.path.join(os.getcwd(),"logs")

os.makedirs(logs_path,exist_ok=True)
LOG_FILE_PATH=os.path.join(logs_path,LOG_FILE)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler()
    ]
)


