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
    num_epoch: int = 100, 
    lr: float = 1e-3,
    batch_size: int = 64,
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

    loss_fn = torch.nn.L1Loss()

    train_data = road_dataset.load_data(
        "drive_data/train",
        transform_pipeline="state_only",
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    val_data = road_dataset.load_data(
        "drive_data/val",
        transform_pipeline="state_only",
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.2,
        patience=7,
        verbose=True,
        min_lr=1e-6 
    )

    global_step = 0
    for epoch in range(num_epoch):
        model.train()
        train_metrics = PlannerMetric()
        train_metrics.reset()

        for batch in train_data:
            track_left = batch['track_left'].to(device)
            track_right = batch['track_right'].to(device)
            target_waypoints = batch['waypoints'].to(device)
            waypoints_mask = batch['waypoints_mask'].to(device)

            optimizer.zero_grad()
            pred = model(track_left, track_right)
            loss = loss_fn(pred, target_waypoints)
            loss.backward()
            optimizer.step()

            train_metrics.add(pred, target_waypoints, waypoints_mask)

            global_step += 1
            if global_step % 10 == 0:
                logger.add_scalar("train/loss", loss.item(), global_step)
        
        train_results = train_metrics.compute()

        model.eval()
        val_metrics = PlannerMetric()
        val_metrics.reset()

        with torch.no_grad():
            val_loss = 0
            val_batches = 0
            for batch in val_data:
                track_left = batch['track_left'].to(device)
                track_right = batch['track_right'].to(device)
                target_waypoints = batch['waypoints'].to(device)
                waypoints_mask = batch['waypoints_mask'].to(device)

                pred = model(track_left, track_right)
                loss = loss_fn(pred, target_waypoints)
                val_loss += loss
                val_batches += 1

                val_metrics.add(pred, target_waypoints, waypoints_mask)
                logger.add_scalar("val/loss", loss.item(), global_step)

            avg_val_loss = val_loss / val_batches
            logger.add_scalar("val/loss", avg_val_loss, global_step)
            scheduler.step(avg_val_loss)
        
        val_results = val_metrics.compute()

        print(f"Epoch [{epoch+1}/{num_epoch}]")
        print(f"Train - Lateral: {train_results['lateral_error']:.4f}, Long: {train_results['longitudinal_error']:.4f}")
        print(f"Val   - Lateral: {val_results['lateral_error']:.4f}, Long: {val_results['longitudinal_error']:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print("-" * 50)     
        

    save_model(model)
    torch.save(model.state_dict(), log_dir / "mlp.th")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, default="mlp_planner") 
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=2024)

    args = parser.parse_args()
    train_MLP(**vars(args))
    