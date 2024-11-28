"""
Usage:
    python3 -m homework.train_planner --your_args here
"""

import torch
import argparse
import numpy as np
import torch.utils.tensorboard as tb

from pathlib import Path
from datetime import datetime
from .metrics import PlannerMetric
from .models import TransformerPlanner, CNNPlanner, MLPPlanner, save_model, load_model
from .datasets import road_dataset, road_transforms


@torch.inference_mode()
def compute_metrics(model, data, device):
    model.eval()
    metrics = PlannerMetric()
    metrics.reset()
    for batch in data:
        img = batch['image'].to(device)
        target = batch['target'].to(device)
        pred = model(img)
        metrics.add(pred, target)
    return metrics.compute()


def train_MLP(
    model_name: str = "mlp_planner",
    exp_dir: str = "logs",
    num_epoch: int = 50,
    lr: float = 1e-4,
    batch_size: int = 32,
    seed: int = 2024,
    **kwargs,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = tb.SummaryWriter(log_dir)

    model = MLPPlanner()
    model = model.to(device)

    train_data = road_dataset.load_data(
        "drive_data/train",
        transform_pipeline="aug",
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    val_data = road_dataset.load_data(
        "drive_data/val",
        transform_pipeline="default",
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    optimizer = torch.optim.adamw(model.parameters(), lr=lr)

    global_step = 0
    for epoch in range(num_epoch):
        model.train()
        for batch in train_data:
            track_left = batch['track_left'].to(device)
            track_right = batch['track_right'].to(device)
            target_waypoints = batch['waypoints'].to(device)

            optimizer.zero()
            pred = model(track_left, track_right)
            loss = torch.nn.MSELoss(pred, target_waypoints)
            loss.backward()
            optimizer.step()

            global_step += 1
            if global_step % 10 == 0:
                logger.add_scalar("train/loss", loss.item(), global_step)

        model.eval()
        with torch.no_grad():
            for batch in val_data:
                track_left = batch['track_left'].to(device)
                track_right = batch['track_right'].to(device)
                target_waypoints = batch['waypoints'].to(device)

                pred = model(track_left, track_right)
                loss = torch.nn.MSELoss(pred, target_waypoints)
                logger.add_scalar("val/loss", loss.item(), global_step)
        

    save_model(model)
    torch.save(model.state_dict(), log_dir / "mlp.th")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, default="mlp_planner") 
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2024)

    args = parser.parse_args()
    train_MLP(**vars(args))
    