import os 
import sys 
import logging
# from transformers.utils import logging as hf_logging

logging_str = "%(asctime)s: %(levelname)s: %(module)s: %(message)s"
log_dir = "./logs"
log_filepath = os.path.join(log_dir,"running_logs.log")
print(os.getcwd())
# os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
   level=logging.INFO, 
   format=logging_str, 
   
   handlers=[
    #    logging.FileHandler(log_filepath),
       logging.StreamHandler(sys.stdout)
   ]
)
logger = logging.getLogger("English_Grading")

# hf_logging.add_handler(logging.FileHandler(log_filepath))
# hf_logging.add_handler(logging.StreamHandler(sys.stdout))
# logger = hf_logging.get_logger("Legal-chatdemo")