import os, re, json, torch
import os.path as osp
import numpy as np


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_json_file(file_path):
    with open(file_path, "r") as f:
        s = f.read()
        s = re.sub('\s',"", s)
    return json.loads(s)


def z_score(data):
    """
    Calculate the standardized value of the data, that is, subtract the mean from the data 
    and divide it by the standard deviation to ensure that the data follows a standard normal distribution in a statistical sense
    """
    return (data - np.mean(data)) / np.std(data)


def load_best_model(args):
    if (args.load_first_year and args.year <= args.begin_year +  1) or args.train == 0:
        if osp.join(args.save_data_path).split('/')[1] =='PEMS':
            load_path = './log/PEMS.pkl'
        elif osp.join(args.save_data_path).split('/')[1] =='CA':
            load_path = './log/CA.pkl'
        elif osp.join(args.save_data_path).split('/')[1] =='ENERGY-Wind':
            load_path = './log/ENERGY-Wind.pkl'
        elif osp.join(args.save_data_path).split('/')[1] =='AIR':
            load_path = './log/AIR.pkl'
        loss = [9999]
    else:
        loss = []
        for filename in os.listdir(osp.join(args.path, str(args.year-1))): 
            loss.append(filename[0:-4])
        loss = sorted(loss)
        load_path = osp.join(args.path, str(args.year-1), loss[0]+".pkl")
        
    args.logger.info("[*] load from {}".format(load_path)) 
    state_dict = torch.load(load_path, map_location=args.device)["model_state_dict"]  

    model = args.methods[args.method](args)
    
    if args.method == 'STBP':
        if args.year == args.begin_year:
            model.expand_adaptive_params(args.base_node_size)
        else:
            for idx, _ in enumerate(range(args.year - args.begin_year)):
                model.expand_adaptive_params(args.graph_size_list[idx])

    model.load_state_dict(state_dict, strict=False) 
    model = model.to(args.device)  
    return model, loss[0] 
