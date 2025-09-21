import sys, argparse, torch
sys.path.append("src/")
import os
import numpy as np
import os.path as osp
import networkx as nx
from torch_geometric.loader import DataLoader
from utils.data_convert import generate_samples
from src.model.model import STBP_Model
from src.dataer.SpatioTemporalDataset import SpatioTemporalDataset
from utils.initialize import init, seed_anything, init_log
from utils.common_tools import mkdirs, load_best_model
from src.trainer.default_trainer import train, test_model

def main(args):
    args.logger.info("params : %s", vars(args))
    args.result = {"3":{" MAE":{}, "MAPE":{}, "RMSE":{}}, "6":{" MAE":{}, "MAPE":{}, "RMSE":{}}, "12":{" MAE":{}, "MAPE":{}, "RMSE":{}}, "Avg":{" MAE":{}, "MAPE":{}, "RMSE":{}}}
    mkdirs(args.save_data_path)
    vars(args)["graph_size_list"] = []

    for year in range(args.begin_year, args.end_year+1): 
            
        graph = nx.from_numpy_array(np.load(osp.join(args.graph_path, str(year)+"_adj.npz"))["x"])
        
        if year == args.begin_year:
            vars(args)["init_graph_size"] = graph.number_of_nodes()
            vars(args)["subgraph"] = graph
            vars(args)["base_node_size"] = graph.number_of_nodes()
        else:
            vars(args)["init_graph_size"] = args.graph_size

        vars(args)["graph_size"] = graph.number_of_nodes()
        vars(args)["year"] = year
        args.graph_size_list.append(graph.number_of_nodes())
        
        inputs = generate_samples(31, osp.join(args.save_data_path, str(year)), np.load(osp.join(args.raw_data_path, str(year)+".npz"))["x"], graph, val_test_mix=False) \
            if args.data_process else np.load(osp.join(args.save_data_path, str(year)+".npz"), allow_pickle=True)
        
        args.logger.info("[*] Year {} load from {}.npz".format(args.year, osp.join(args.save_data_path, str(year))))
        
        adj = np.load(osp.join(args.graph_path, str(args.year)+"_adj.npz"))["x"]
        adj = adj / (np.sum(adj, 1, keepdims=True) + 1e-6)
        vars(args)["adj"] = torch.from_numpy(adj).to(torch.float).to(args.device) 
        
        if year == args.begin_year and args.load_first_year:
            model, _ = load_best_model(args)
            test_loader = DataLoader(SpatioTemporalDataset(inputs, "test"), batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=16)
            test_model(model, args, test_loader, True)
            continue
        
        if args.train:
            train(inputs, args)
        else:
            model, _= load_best_model(args)
            test_loader = DataLoader(SpatioTemporalDataset(inputs, "test"), batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=16)
            test_model(model, args, test_loader, True)
        
    args.logger.info("\n\n")
    for i in ["3", "6", "12", "Avg"]:
        for j in [" MAE", "RMSE", "MAPE"]:
            info = ""
            info_list = []
            for year in range(args.begin_year, args.end_year+1):
                if i in args.result:
                    if j in args.result[i]:
                        if year in args.result[i][j]:
                            info += "{:>10.2f}\t".format(args.result[i][j][year])
                            info_list.append(args.result[i][j][year])
            args.logger.info("{:<4}\t{}\t".format(i, j) + info + "\t{:>8.2f}".format(np.mean(info_list)))

    total_time = 0
    for year in range(args.begin_year, args.end_year+1):
        if year in args.result:
            info = "year\t{:<4}\ttotal_time\t{:>10.4f}\taverage_time\t{:>10.4f}\tepoch\t{}".format(year, args.result[year]["total_time"], args.result[year]["average_time"], args.result[year]['epoch_num'])
            total_time += args.result[year]["total_time"]
            args.logger.info(info)
    args.logger.info("total time: {:.4f}".format(total_time))


if __name__ == "__main__":
    current_pid = os.getpid()
    print(f"当前进程号(PID): {current_pid}")
    parser = argparse.ArgumentParser(formatter_class = argparse.RawTextHelpFormatter)
    parser.add_argument("--conf", type = str, default = "conf/test.json")
    parser.add_argument("--seed", type = int, default = 42)
    parser.add_argument("--paral", type = int, default = 0)
    parser.add_argument("--gpuid", type = int, default = 2)
    parser.add_argument("--logname", type = str, default = "info")
    parser.add_argument("--method", type = str, default = "STBP")
    parser.add_argument("--load_first_year", type = int, default = 0, help="0: training first year, 1: load from model path of first year")
    parser.add_argument("--first_year_model_path", type = str, default = "", help='specify a pretrained model root')
    args = parser.parse_args()
    vars(args)["device"] = torch.device("cuda:{}".format(args.gpuid)) if torch.cuda.is_available() and args.gpuid != -1 else "cpu"
    vars(args)["methods"] = {'STBP': STBP_Model}
    init(args)
    seed_anything(args.seed)
    init_log(args)
    main(args)
