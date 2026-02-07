import numpy as np
import torch


def MAE_torch(prediction: torch.Tensor, target: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    """Masked mean absolute error.

    Args:
        prediction (torch.Tensor): predicted values
        target (torch.Tensor): labels
        null_val (float, optional): null value. Defaults to np.nan.

    Returns:
        torch.Tensor: masked mean absolute error
    """

    if np.isnan(null_val):
        mask = ~torch.isnan(target)
    else:
        eps = 5e-5
        mask = ~torch.isclose(target, torch.tensor(null_val).expand_as(target).to(target.device), atol=eps, rtol=0.)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(prediction-target)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def mask_np(array, null_val):
    if np.isnan(null_val):
        return (~np.isnan(null_val)).astype('float32')
    else:
        return np.not_equal(array, null_val).astype('float32')


def masked_mape_np(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = mask_np(y_true, null_val)
        mask /= mask.mean()
        mape = np.abs((y_pred - y_true) / y_true)
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100


def masked_mse_np(y_true, y_pred, null_val=np.nan):
    mask = mask_np(y_true, null_val)
    mask /= mask.mean()
    mse = (y_true - y_pred) ** 2
    return np.mean(np.nan_to_num(mask * mse))


def masked_mae_np(y_true, y_pred, null_val=np.nan):
    mask = mask_np(y_true, null_val)
    mask /= mask.mean()
    mae = np.abs(y_true - y_pred)
    return np.mean(np.nan_to_num(mask * mae))

# 前人工作计算方式有些问题，是前3、前6、前12时间步的平均，而非第3、第6、第12，最终Avg指标也是有问题的
# def cal_metric(ground_truth, prediction, args):
#     args.logger.info("[*] year {}, testing".format(args.year))
#     mae_list, rmse_list, mape_list = [], [], []
#     for i in range(1, 13):
#         mae = masked_mae_np(ground_truth[:, :, :i], prediction[:, :, :i], 0)
#         rmse = masked_mse_np(ground_truth[:, :, :i], prediction[:, :, :i], 0) ** 0.5
#         mape = masked_mape_np(ground_truth[:, :, :i], prediction[:, :, :i], 0)
#         mae_list.append(mae)
#         rmse_list.append(rmse)
#         mape_list.append(mape)
#         if i==3 or i==6 or i==12:
#             args.logger.info("T:{:d}\tMAE\t{:.4f}\tRMSE\t{:.4f}\tMAPE\t{:.4f}".format(i,mae,rmse,mape))
#             args.result[str(i)][" MAE"][args.year] = mae
#             args.result[str(i)]["MAPE"][args.year] = mape
#             args.result[str(i)]["RMSE"][args.year] = rmse
#     args.result["Avg"][" MAE"][args.year] = np.mean(mae_list)
#     args.result["Avg"]["RMSE"][args.year] = np.mean(rmse_list)
#     args.result["Avg"]["MAPE"][args.year] = np.mean(mape_list)
#     args.logger.info("T:Avg\tMAE\t{:.4f}\tRMSE\t{:.4f}\tMAPE\t{:.4f}".format(np.mean(mae_list), np.mean(rmse_list), np.mean(mape_list)))


def cal_metric(ground_truth, prediction, args):
    """Calculate metrics for each time step."""
    args.logger.info(f"[*] year {args.year}, testing")
    
    mae_list, rmse_list, mape_list = [], [], []
    
    # 假设ground_truth和prediction的形状为 [batch_size, num_nodes, 12]
    # 我们计算每个时间步的指标
    num_time_steps = ground_truth.shape[2]  # 应该是12
    
    for t in range(1, num_time_steps + 1):
        # 取当前时间步的数据
        # 假设我们要计算第t个时间步的预测（从1开始计数）
        gt_t = ground_truth[:, :, t-1:t]  # 第t个时间步的真实值
        pred_t = prediction[:, :, t-1:t]  # 第t个时间步的预测值
        
        mae = masked_mae_np(gt_t, pred_t, 0)
        rmse = masked_rmse_np(gt_t, pred_t, 0)
        mape = masked_mape_np(gt_t, pred_t, 0)
        
        mae_list.append(mae)
        rmse_list.append(rmse)
        mape_list.append(mape)
        
        # 输出第3、6、12个时间步的结果
        if t in [3, 6, 12]:
            args.logger.info(f"T:{t}\tMAE\t{mae:.4f}\tRMSE\t{rmse:.4f}\tMAPE\t{mape:.4f}")
            args.result[str(t)][" MAE"][args.year] = mae
            args.result[str(t)]["MAPE"][args.year] = mape
            args.result[str(t)]["RMSE"][args.year] = rmse
    
    # 计算所有时间步的平均指标
    avg_mae = np.mean(mae_list)
    avg_rmse = np.mean(rmse_list)
    avg_mape = np.mean(mape_list)
    
    args.result["Avg"][" MAE"][args.year] = avg_mae
    args.result["Avg"]["RMSE"][args.year] = avg_rmse
    args.result["Avg"]["MAPE"][args.year] = avg_mape
    
    args.logger.info(f"T:Avg\tMAE\t{avg_mae:.4f}\tRMSE\t{avg_rmse:.4f}\tMAPE\t{avg_mape:.4f}")
    
    return mae_list, rmse_list, mape_list
