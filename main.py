import torch
import torch.nn.functional as F # 追加
import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import random
import numpy as np
from src.models.evflownet import EVFlowNet
from src.datasets import DatasetProvider
from enum import Enum, auto
from src.datasets import train_collate
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any
import os
import time


class RepresentationType(Enum):
    VOXEL = auto()
    STEPAN = auto()

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def compute_epe_error(pred_flows: Dict[str, torch.Tensor], gt_flow: torch.Tensor, epoch, total_epochs):
    '''
    異なるスケールのflowを用いてend-point-errorを計算
    pred_flows: Dict[str, torch.Tensor] => 予測したオプティカルフローデータの辞書. keyはflow0~flow3
    gt_flow: torch.Tensor, Shape: torch.Size([B, 2, 480, 640]) => 正解のオプティカルフローデータ
    '''

    epe_loss = 0

    scale = epoch / total_epochs
    weights = [0.10 * (1 - scale), 
               0.15 * (1 - scale), 
               0.20 * (1 - scale), 
               1.0]

    _, _, h, w = gt_flow.shape

    for i in range(4):
        flow_key = f'flow{i}'
        scale_factor = 2**(3-i) 

        gt_flow_resized = F.interpolate(gt_flow, size=(h // scale_factor, w // scale_factor), mode='bicubic', align_corners=True)
        epe_loss += weights[i] * torch.mean(torch.mean(torch.norm(pred_flows[flow_key] - gt_flow_resized, p=2, dim=1), dim=(1, 2)), dim=0)

    flow = pred_flows["flow3"]
    flow_ucrop = flow[..., 1:]
    flow_dcrop = flow[..., :-1]
    flow_lcrop = flow[..., 1:, :]
    flow_rcrop = flow[..., :-1, :]

    flow_ulcrop = flow[..., 1:, 1:]
    flow_drcrop = flow[..., :-1, :-1]
    flow_dlcrop = flow[..., :-1, 1:]
    flow_urcrop = flow[..., 1:, :-1]
    
    smoothness_loss = charbonnier_loss(flow_lcrop - flow_rcrop) +\
                    charbonnier_loss(flow_ucrop - flow_dcrop) +\
                    charbonnier_loss(flow_ulcrop - flow_drcrop) +\
                    charbonnier_loss(flow_dlcrop - flow_urcrop)
        
    
    loss = epe_loss + 0.1 * smoothness_loss

    return loss

def charbonnier_loss(x, alpha=0.45, epsilon=1e-3):
    return torch.mean(torch.pow(torch.pow(x, 2) + epsilon**2, alpha))

def compute_last_epe_error(pred_flows: Dict[str, torch.Tensor], gt_flow: torch.Tensor):
    loss = torch.mean(torch.mean(torch.norm(pred_flows["flow3"] - gt_flow, p=2, dim=1), dim=(1, 2)), dim=0)        
    return loss

def save_optical_flow_to_npy(flow: torch.Tensor, file_name: str):
    '''
    optical flowをnpyファイルに保存
    flow: torch.Tensor, Shape: torch.Size([2, 480, 640]) => オプティカルフローデータ
    file_name: str => ファイル名
    '''
    np.save(f"{file_name}.npy", flow.cpu().numpy())

@hydra.main(version_base=None, config_path="configs", config_name="base")
def main(args: DictConfig):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    '''
        ディレクトリ構造:

        data
        ├─test
        |  ├─test_city
        |  |    ├─events_left
        |  |    |   ├─events.h5
        |  |    |   └─rectify_map.h5
        |  |    └─forward_timestamps.txt
        └─train
            ├─zurich_city_11_a
            |    ├─events_left
            |    |       ├─ events.h5
            |    |       └─ rectify_map.h5
            |    ├─ flow_forward
            |    |       ├─ 000134.png
            |    |       |.....
            |    └─ forward_timestamps.txt
            ├─zurich_city_11_b
            └─zurich_city_11_c
        '''
    
    # ------------------
    #    Dataloader
    # ------------------
    loader = DatasetProvider(
        dataset_path=Path(args.dataset_path),
        representation_type=RepresentationType.VOXEL,
        delta_t_ms=100,
        num_bins=4
    )
    train_set = loader.get_train_dataset()
    test_set = loader.get_test_dataset()
    
    # データセットのサイズを取得
    dataset_size = len(train_set)

    # 8:2で分割
    train_size = int(dataset_size * 0.8)
    val_size = dataset_size - train_size

    # データセットを分割
    train_dataset, val_dataset = torch.utils.data.random_split(train_set, [train_size, val_size])
    
    collate_fn = train_collate
    train_data = DataLoader(train_dataset,
                                 batch_size=args.data_loader.train.batch_size,
                                 shuffle=args.data_loader.train.shuffle,
                                 collate_fn=collate_fn,
                                 drop_last=False)
    val_data = DataLoader(val_dataset,
                                 batch_size=args.data_loader.test.batch_size,
                                 shuffle=args.data_loader.test.shuffle,
                                 collate_fn=collate_fn,
                                 drop_last=False)
    test_data = DataLoader(test_set,
                                 batch_size=args.data_loader.test.batch_size,
                                 shuffle=args.data_loader.test.shuffle,
                                 collate_fn=collate_fn,
                                 drop_last=False)

    '''
    train data:
        Type of batch: Dict
        Key: seq_name, Type: list
        Key: event_volume, Type: torch.Tensor, Shape: torch.Size([Batch, 4, 480, 640]) => イベントデータのバッチ
        Key: flow_gt, Type: torch.Tensor, Shape: torch.Size([Batch, 2, 480, 640]) => オプティカルフローデータのバッチ
        Key: flow_gt_valid_mask, Type: torch.Tensor, Shape: torch.Size([Batch, 1, 480, 640]) => オプティカルフローデータのvalid. ベースラインでは使わない
    
    test data:
        Type of batch: Dict
        Key: seq_name, Type: list
        Key: event_volume, Type: torch.Tensor, Shape: torch.Size([Batch, 4, 480, 640]) => イベントデータのバッチ
    '''
    # ------------------
    #       Model
    # ------------------
    model = EVFlowNet(args.train).to(device)

    # ------------------
    #   optimizer
    # ------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=args.train.initial_learning_rate, weight_decay=args.train.weight_decay)
    # ------------------
    #   Start training
    # ------------------
    best_val_loss = float('inf')  # 初期のバリデーションロスを無限大に設定
    
    # Create the directory if it doesn't exist
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    
    for epoch in range(args.train.epochs):
        model.train()
        total_loss = 0
        print("on epoch: {}".format(epoch+1))
        for i, batch in enumerate(tqdm(train_data)):
            batch: Dict[str, Any]
            event_image = batch["event_volume"].to(device) # [B, 4, 480, 640]
            ground_truth_flow = batch["flow_gt"].to(device) # [B, 2, 480, 640]
            flow_dict = model(event_image) # Dict[str, torch.Tensor]
            loss: torch.Tensor = compute_epe_error(flow_dict, ground_truth_flow, epoch, args.train.epochs)
            # print(f"batch {i} loss: {loss.item()}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_data)}')
        
        # --- バリデーションフェーズ ---
        model.eval()  # モデルを評価モードに切り替え
        val_loss = 0
        with torch.no_grad():  # 勾配計算を行わない
            for j, val_batch in enumerate(tqdm(val_data)):
                val_event_image = val_batch["event_volume"].to(device)
                val_ground_truth_flow = val_batch["flow_gt"].to(device)
                val_flow_dict = model(val_event_image)
                val_batch_loss = compute_last_epe_error(val_flow_dict, val_ground_truth_flow)
                val_loss += val_batch_loss.item()

        avg_val_loss = val_loss / len(val_data)
        print(f"Epoch {epoch+1}, Validation Loss: {avg_val_loss:.4f}")
        
        # --- ベストモデルの保存 ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # ベストモデルを保存
            current_time = time.strftime("%Y%m%d%H%M%S")
            best_model_path = f"checkpoints/best_model_{current_time}.pth"
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved to {best_model_path}")

    # ------------------
    #   Start predicting
    # ------------------
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    flow: torch.Tensor = torch.tensor([]).to(device)
    with torch.no_grad():
        print("start test")
        for batch in tqdm(test_data):
            batch: Dict[str, Any]
            event_image = batch["event_volume"].to(device)
            flow_dict = model(event_image) # 返り値はflow_dict
            batch_flow = flow_dict["flow3"] # [1, 2, 480, 640]
            flow = torch.cat((flow, batch_flow), dim=0)  # [N, 2, 480, 640]
        print("test done")
    # ------------------
    #  save submission
    # ------------------
    file_name = "submission"
    save_optical_flow_to_npy(flow, file_name)

if __name__ == "__main__":
    main()
