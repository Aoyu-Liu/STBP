import torch
import numpy as np
import os.path as osp
from torch import optim
from datetime import datetime
from torch_geometric.utils import to_dense_batch
from torch_geometric.loader import DataLoader
from dataer.SpatioTemporalDataset import SpatioTemporalDataset
from utils.metric import cal_metric, masked_mae_np, MAE_torch
from utils.common_tools import mkdirs, load_best_model


def train(inputs, args):
    path = osp.join(args.path, str(args.year))
    mkdirs(path)

    train_loader = DataLoader(SpatioTemporalDataset(inputs, "train"), batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=16)
    val_loader = DataLoader(SpatioTemporalDataset(inputs, "val"), batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=16)
    test_loader = DataLoader(SpatioTemporalDataset(inputs, "test"), batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=16)
    vars(args)["sub_adj"] = vars(args)["adj"] 
    
    args.logger.info("[*] Year " + str(args.year) + " Dataset load!")

    if args.init == True and args.year > args.begin_year:
        CSTL_model, _  = load_best_model(args) 
        model = CSTL_model
        
        if args.method == 'STBP':
            for name, param in model.named_parameters():
                if "pattern_bank" in name: 
                    param.requires_grad = True
                else:
                    param.requires_grad = False 
            model.expand_adaptive_params(args.graph_size)
        
    else:
        CSTL_model = args.methods[args.method](args).to(args.device)  
        model = CSTL_model
        if args.method == 'STBP':
            model.expand_adaptive_params(args.graph_size)
    
    model.count_parameters()
    
    for name, param in model.named_parameters():
        print(f"Parameter: {name} | Shape: {param.shape} | Requires Grad: {param.requires_grad}")
    
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    args.logger.info("[*] Year " + str(args.year) + " Training start")
    lowest_validation_loss = 1e7
    counter = 0
    patience = 30
    model.train()
    use_time = []
    
    for epoch in range(args.epoch):
        
        start_time = datetime.now()
        
        cn = 0
        training_loss = 0.0
        for batch_idx, data in enumerate(train_loader):
            if epoch == 0 and batch_idx == 0:
                args.logger.info("node number {}".format(data.x.shape))
            data = data.to(args.device, non_blocking=True)
            optimizer.zero_grad()
            pred = model(data, args.sub_adj)
            
            loss = MAE_torch(pred, data.y, 0.0)
            
            training_loss += float(loss)
            cn += 1
            
            loss.backward()
            optimizer.step()
        
        if epoch == 0:
            total_time = (datetime.now() - start_time).total_seconds()
        else:
            total_time += (datetime.now() - start_time).total_seconds()
        use_time.append((datetime.now() - start_time).total_seconds())
        training_loss = training_loss / cn 
        
        validation_loss = 0.0
        cn = 0
        with torch.no_grad():
            for batch_idx, data in enumerate(val_loader):
                data = data.to(args.device, non_blocking=True)
                pred = model(data, args.sub_adj)
                loss = masked_mae_np(data.y.cpu().data.numpy(), pred.cpu().data.numpy(), 0)
                validation_loss += float(loss)
                cn += 1
        validation_loss = float(validation_loss/cn)

        args.logger.info(f"epoch:{epoch}, training loss:{training_loss:.4f} validation loss:{validation_loss:.4f}")
        
        if validation_loss <= lowest_validation_loss:
            counter = 0
            lowest_validation_loss = round(validation_loss, 4)
            torch.save({'model_state_dict': model.state_dict()}, osp.join(path, str(round(validation_loss,4))+".pkl"))
        else:
            counter += 1
            if counter > patience:
                break
        
    best_model_path = osp.join(path, str(lowest_validation_loss)+".pkl")
    best_model = model
    
    best_model.load_state_dict(torch.load(best_model_path, args.device)["model_state_dict"])
    best_model = best_model.to(args.device)
    
    test_model(best_model, args, test_loader, True)
    args.result[args.year] = {"total_time": total_time, "average_time": sum(use_time)/len(use_time), "epoch_num": epoch+1}
    args.logger.info("Finished optimization, total time:{:.2f} s, best model:{}".format(total_time, best_model_path))


def test_model(model, args, testset, pin_memory):
    model.eval()
    pred_ = []
    truth_ = []
    loss = 0.0
    with torch.no_grad():
        cn = 0
        for data in testset:
            data = data.to(args.device, non_blocking=pin_memory)
            pred = model(data, args.adj)
            loss += MAE_torch(pred, data.y, 0.0)
            pred, _ = to_dense_batch(pred, batch=data.batch)
            data.y, _ = to_dense_batch(data.y, batch=data.batch)
            pred_.append(pred.cpu().data.numpy())
            truth_.append(data.y.cpu().data.numpy())
            cn += 1
        loss = loss / cn
        args.logger.info("[*] loss:{:.4f}".format(loss))
        pred_ = np.concatenate(pred_, 0)
        truth_ = np.concatenate(truth_, 0)
        cal_metric(truth_, pred_, args)
