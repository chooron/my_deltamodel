"""
测试 nan 问题 - 对比 Triton 和 autograd 后端
"""
import os
import sys
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.getenv("PROJ_PATH"))

import torch
from dmg import ModelHandler
from dmg.core.utils import import_data_loader, set_randomseed
from project.triton_accelerate import load_config

CONFIG_PATH = r"conf/config_dhbv_ann.yaml"

def test_backend(backend_name):
    """测试指定后端"""
    print(f"\n{'='*60}")
    print(f"Testing backend: {backend_name}")
    print('='*60)
    
    config = load_config(CONFIG_PATH)
    config["mode"] = "train"
    config["train"]["epochs"] = 1
    set_randomseed(config["random_seed"])
    
    model_handler = ModelHandler(config, verbose=False)
    
    # 修改后端 - model_dict 包含 DplModel，需要访问其 phy_model
    for name, dpl_model in model_handler.model_dict.items():
        if hasattr(dpl_model, 'phy_model'):
            phy_m = dpl_model.phy_model
            if hasattr(phy_m, 'backend'):
                phy_m.backend = backend_name
                print(f"  Set {phy_m.name} backend to: {phy_m.backend}")
    
    data_loader_cls = import_data_loader(config["data_loader"])
    data_loader = data_loader_cls(config, test_split=True, overwrite=False)
    
    # 手动跑几个batch
    from torch.utils.data import DataLoader
    train_loader = DataLoader(
        data_loader.train_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
    )
    
    # 设置训练模式
    for dpl_model in model_handler.model_dict.values():
        dpl_model.train()
    
    optimizer = torch.optim.Adadelta(model_handler.get_parameters(), lr=1.0)
    
    nan_batch = None
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= 150:
            break
            
        optimizer.zero_grad()
        
        # 准备数据
        x_phy = batch['x_phy'].to(config['device'])
        x_nn = batch['x_nn'].to(config['device'])
        c_phy = batch['c_phy'].to(config['device'])
        c_nn = batch['c_nn'].to(config['device'])
        target = batch['target'].to(config['device'])
        
        try:
            # 前向传播
            output_dict = model_handler({
                'x_phy': x_phy,
                'x_nn': x_nn,
                'c_phy': c_phy,
                'c_nn': c_nn,
            })
            
            # 获取第一个模型的输出
            model_name = list(output_dict.keys())[0]
            output = output_dict[model_name]
            
            # 简单的loss
            pred = output['streamflow']
            loss = ((pred - target[:,:,0])**2).mean()
            
            if torch.isnan(loss):
                print(f"  Batch {batch_idx}: loss = nan ❌")
                nan_batch = batch_idx
                
                # 检查输出中的nan
                for k, v in output.items():
                    if v is not None and torch.is_tensor(v):
                        nan_count = torch.isnan(v).sum().item()
                        if nan_count > 0:
                            print(f"    {k}: {nan_count} NaN values")
                break
            else:
                if batch_idx % 20 == 0:
                    print(f"  Batch {batch_idx}: loss = {loss.item():.4f} ✓")
                    
            # 反向传播
            loss.backward()
            
            # 检查梯度中的nan
            has_nan_grad = False
            for name, param in model_handler.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    has_nan_grad = True
                    print(f"  Batch {batch_idx}: NaN gradient in {name}")
                    break
            
            if has_nan_grad:
                nan_batch = batch_idx
                break
                
            optimizer.step()
            
        except Exception as e:
            print(f"  Batch {batch_idx}: Error - {e}")
            import traceback
            traceback.print_exc()
            nan_batch = batch_idx
            break
    
    if nan_batch is None:
        print(f"\n  ✓ {backend_name} backend: No NaN in 150 batches")
    else:
        print(f"\n  ✗ {backend_name} backend: NaN at batch {nan_batch}")
    
    return nan_batch

if __name__ == "__main__":
    # 先测试 autograd 后端
    autograd_nan = test_backend("autograd")
    
    # 再测试 triton 后端
    triton_nan = test_backend("triton")
    
    print("\n" + "="*60)
    print("Summary:")
    print("="*60)
    print(f"  autograd backend: {'NaN at batch ' + str(autograd_nan) if autograd_nan else 'OK'}")
    print(f"  triton backend:   {'NaN at batch ' + str(triton_nan) if triton_nan else 'OK'}")
