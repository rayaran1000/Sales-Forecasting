import logging
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

logs_path=os.path.join(os.getcwd(),"logs",LOG_FILE) #Every log will start with the word "log" then the LOG_FILE details(Logs will create wrt the current directory)
os.makedirs(logs_path,exist_ok=True) #Even if we have a file, keep appending the logs in the same file

LOG_FILE_PATH = os.path.join(logs_path,LOG_FILE)

#Here we are over riding the functionality of the logging module in this Basic Config(features of the logging file)
logging.basicConfig(
    filename = LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,

)