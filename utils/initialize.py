import sys, random, torch, logging
import numpy as np
import os.path as osp
from utils import common_tools as ct
from datetime import datetime  


def init(args):
    '''
    Step 1.1 : Initialize configuration parameters
    '''
    def _update(src, tmp):
        for key in tmp:
            if key != "gpuid":
                src[key] = tmp[key]
    
    conf_path = osp.join(args.conf)  
    info = ct.load_json_file(conf_path)  
    _update(vars(args), info)

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    vars(args)["path"] = osp.join(args.model_path, f"{args.logname}-{current_time}-seed{args.seed}")

    ct.mkdirs(args.path) 
    del info 


def seed_anything(seed=42):
    '''
    Step 1.2: Initialize random seed
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def init_log(args):
    '''
    Step 1.3: Initialize the logging object
    '''
    log_dir, log_filename = args.path, args.logname
    logger = logging.getLogger(__name__)
    ct.mkdirs(log_dir)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(osp.join(log_dir, log_filename+".log"))
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)  
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info("logger name:%s", osp.join(log_dir, log_filename+".log"))
    vars(args)["logger"] = logger

