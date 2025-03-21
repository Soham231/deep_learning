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

# class WeightedL1Loss(torch.nn.Module):
#     def __init__(self, lateral_weight=3.0):
#         super().__init__()
#         self.lateral_weight = lateral_weight
#     def forward(self, pred, target):
#         # Calculate L1 loss for each component
#         loss = torch.abs(pred - target)  # Shape: (batch, waypoints, 2)
#         # Apply higher weight to lateral error (index 1)
#         loss[..., 1] *= self.lateral_weight
#         # Return mean of all errors
#         return loss.mean()

# @torch.inference_mode()
# def compute_metrics(model, data, device):
#     model.eval()
#     metrics = PlannerMetric()
#     metrics.reset()
#     for batch in data:
#         img = batch['image'].to(device)
#         target = batch['target'].to(device)
#         pred = model(img)
#         metrics.add(pred, target)
#     return metrics.compute()


# def train_MLP(
#     model_name: str = "mlp_planner",
#     exp_dir: str = "logs",
#     num_epoch: int = 100, 
#     lr: float = 1e-3,
#     batch_size: int = 64,
#     seed: int = 2024,
#     **kwargs,
# ):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     best_lateral = float('inf')
#     best_state = None
#     patience = 0
#     max_patience = 10

#     log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
#     log_dir.mkdir(parents=True, exist_ok=True)
#     logger = tb.SummaryWriter(log_dir)

#     model = MLPPlanner()
#     model = model.to(device)

#     loss_fn = WeightedL1Loss()

#     train_data = road_dataset.load_data(
#         "drive_data/train",
#         transform_pipeline="state_only",
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=2
#     )

#     val_data = road_dataset.load_data(
#         "drive_data/val",
#         transform_pipeline="state_only",
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=2
#     )

#     optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)

#     scheduler = torch.optim.lr_scheduler.CyclicLR(
#         optimizer,
#         base_lr=1e-4,
#         max_lr=5e-4,
#         step_size_up=len(train_data)*5,  # 5 epochs up
#         mode='triangular2',  # Learning rate will decrease over time
#         cycle_momentum=False
#     )

#     global_step = 0
#     for epoch in range(num_epoch):
#         model.train()
#         train_metrics = PlannerMetric()
#         train_metrics.reset()

#         for batch in train_data:
#             track_left = batch['track_left'].to(device)
#             track_right = batch['track_right'].to(device)
#             target_waypoints = batch['waypoints'].to(device)
#             waypoints_mask = batch['waypoints_mask'].to(device)

#             optimizer.zero_grad()
#             pred = model(track_left, track_right)
#             loss = loss_fn(pred, target_waypoints)
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             optimizer.step()
#             scheduler.step()

#             train_metrics.add(pred, target_waypoints, waypoints_mask)

#             global_step += 1
#             if global_step % 10 == 0:
#                 logger.add_scalar("train/loss", loss.item(), global_step)
        
#         train_results = train_metrics.compute()

#         model.eval()
#         val_metrics = PlannerMetric()
#         val_metrics.reset()

#         with torch.no_grad():
#             val_loss = 0
#             val_batches = 0
#             for batch in val_data:
#                 track_left = batch['track_left'].to(device)
#                 track_right = batch['track_right'].to(device)
#                 target_waypoints = batch['waypoints'].to(device)
#                 waypoints_mask = batch['waypoints_mask'].to(device)

#                 pred = model(track_left, track_right)
#                 loss = loss_fn(pred, target_waypoints)
#                 val_loss += loss
#                 val_batches += 1

#                 val_metrics.add(pred, target_waypoints, waypoints_mask)
#                 logger.add_scalar("val/loss", loss.item(), global_step)

#             avg_val_loss = val_loss / val_batches
#             logger.add_scalar("val/loss", avg_val_loss, global_step)
        
#         val_results = val_metrics.compute()
#         current_lateral = val_results['lateral_error']
        
#         if current_lateral < best_lateral:
#             best_lateral = current_lateral
#             best_state = {k: v.cpu() for k, v in model.state_dict().items()}
#             patience = 0
#         else:
#             patience += 1
            
#         if patience >= max_patience:
#             print(f"Early stopping at epoch {epoch+1}. Best lateral error: {best_lateral:.4f}")
#             model.load_state_dict(best_state)
#             break

#         print(f"Epoch [{epoch+1}/{num_epoch}]")
#         print(f"Train - Lateral: {train_results['lateral_error']:.4f}, Long: {train_results['longitudinal_error']:.4f}")
#         print(f"Val   - Lateral: {val_results['lateral_error']:.4f}, Long: {val_results['longitudinal_error']:.4f}")
#         print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
#         print("-" * 50)     
        

#     save_model(model)
#     torch.save(model.state_dict(), log_dir / "mlp.th")


# def train_transformer(
#     model_name: str = "transformer_planner",
#     exp_dir: str = "logs",
#     num_epoch: int = 50,
#     lr: float = 1e-4,  # Lower initial learning rate for transformer
#     batch_size: int = 32,
#     seed: int = 2024,
#     **kwargs,
# ):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)

#     log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
#     log_dir.mkdir(parents=True, exist_ok=True)
#     logger = tb.SummaryWriter(log_dir)

#     model = TransformerPlanner()
#     model = model.to(device)

#     loss_fn = torch.nn.L1Loss()

#     train_data = road_dataset.load_data(
#         "drive_data/train",
#         transform_pipeline="state_only",
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=2
#     )

#     val_data = road_dataset.load_data(
#         "drive_data/val",
#         transform_pipeline="state_only",
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=2
#     )

#     optimizer = torch.optim.AdamW(
#         model.parameters(), 
#         lr=lr,
#         weight_decay=1e-5,  # Lighter weight decay for transformer
#         betas=(0.9, 0.999)
#     )

#     # Cyclic learning rate with warmup
#     scheduler = torch.optim.lr_scheduler.CyclicLR(
#         optimizer,
#         base_lr=5e-5,   # Lower base learning rate
#         max_lr=2e-4,    # Lower max learning rate
#         step_size_up=len(train_data)*3,  # 3 epochs up
#         mode='triangular2',
#         cycle_momentum=False
#     )

#     best_lateral = float('inf')
#     best_state = None
#     patience = 0
#     max_patience = 10
    
#     global_step = 0
#     for epoch in range(num_epoch):
#         model.train()
#         train_metrics = PlannerMetric()
#         train_metrics.reset()
        
#         for batch in train_data:
#             track_left = batch['track_left'].to(device)
#             track_right = batch['track_right'].to(device)
#             target_waypoints = batch['waypoints'].to(device)
#             waypoints_mask = batch['waypoints_mask'].to(device)

#             optimizer.zero_grad()
#             pred = model(track_left, track_right)
#             loss = loss_fn(pred, target_waypoints)
#             loss.backward()
            
#             # Clip gradients - important for transformer stability
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
#             optimizer.step()
#             scheduler.step()

#             train_metrics.add(pred, target_waypoints, waypoints_mask)

#             global_step += 1
#             if global_step % 10 == 0:
#                 logger.add_scalar("train/loss", loss.item(), global_step)

#         train_results = train_metrics.compute()

#         model.eval()
#         val_metrics = PlannerMetric()
#         val_metrics.reset()
        
#         with torch.no_grad():
#             val_loss = 0
#             val_batches = 0
#             for batch in val_data:
#                 track_left = batch['track_left'].to(device)
#                 track_right = batch['track_right'].to(device)
#                 target_waypoints = batch['waypoints'].to(device)
#                 waypoints_mask = batch['waypoints_mask'].to(device)

#                 pred = model(track_left, track_right)
#                 loss = loss_fn(pred, target_waypoints)
#                 val_loss += loss.item()
#                 val_batches += 1

#                 val_metrics.add(pred, target_waypoints, waypoints_mask)
#                 logger.add_scalar("val/loss", loss.item(), global_step)
            
#             avg_val_loss = val_loss / val_batches
#             logger.add_scalar("val/loss", avg_val_loss, global_step)
        
#         val_results = val_metrics.compute()
#         current_lateral = val_results['lateral_error']
        
#         # Early stopping check
#         if current_lateral < best_lateral:
#             best_lateral = current_lateral
#             best_state = {k: v.cpu() for k, v in model.state_dict().items()}
#             patience = 0
#         else:
#             patience += 1
            
#         if patience >= max_patience:
#             print(f"Early stopping at epoch {epoch+1}. Best lateral error: {best_lateral:.4f}")
#             model.load_state_dict(best_state)
#             break

#         print(f"Epoch [{epoch+1}/{num_epoch}]")
#         print(f"Train - Lateral: {train_results['lateral_error']:.4f}, Long: {train_results['longitudinal_error']:.4f}")
#         print(f"Val   - Lateral: {val_results['lateral_error']:.4f}, Long: {val_results['longitudinal_error']:.4f}")
#         print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
#         print("-" * 50)

#     # Save the best model
#     model.load_state_dict(best_state)
#     save_model(model)
#     torch.save(model.state_dict(), log_dir / "transformer.th")

def train_cnn(
    model_name: str = "cnn_planner",
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

    model = CNNPlanner()
    model = model.to(device)

    loss_fn = torch.nn.L1Loss()

    train_data = road_dataset.load_data(
        "drive_data/train",
        transform_pipeline="default",
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=5e-5,
        max_lr=2e-4,
        step_size_up=len(train_data)*3,
        mode='triangular2',
        cycle_momentum=False
    )

    best_lateral = float('inf')
    best_state = None
    patience = 0
    max_patience = 10

    global_step = 0
    for epoch in range(num_epoch):
        model.train()
        train_metrics = PlannerMetric()
        train_metrics.reset()

        for batch in train_data:
            img = batch['image'].to(device)
            target_waypoints = batch['waypoints'].to(device)
            waypoints_mask = batch['waypoints_mask'].to(device)

            optimizer.zero_grad()
            pred = model(img)
            loss = loss_fn(pred, target_waypoints)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()
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
                img = batch['image'].to(device)
                target_waypoints = batch['waypoints'].to(device)
                waypoints_mask = batch['waypoints_mask'].to(device)

                pred = model(img)
                loss = loss_fn(pred, target_waypoints)
                val_loss += loss.item()
                val_batches += 1

                val_metrics.add(pred, target_waypoints, waypoints_mask)
                logger.add_scalar("val/loss", loss.item(), global_step)

            avg_val_loss = val_loss / val_batches
            logger.add_scalar("val/loss", avg_val_loss, global_step)
        
        val_results = val_metrics.compute()
        current_lateral = val_results['lateral_error']

        if current_lateral < best_lateral:
            best_lateral = current_lateral
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            
        if patience >= max_patience:
            print(f"Early stopping at epoch {epoch+1}. Best lateral error: {best_lateral:.4f}")
            model.load_state_dict(best_state)
            break

        print(f"Epoch [{epoch+1}/{num_epoch}]")
        print(f"Train - Lateral: {train_results['lateral_error']:.4f}, Long: {train_results['longitudinal_error']:.4f}")
        print(f"Val   - Lateral: {val_results['lateral_error']:.4f}, Long: {val_results['longitudinal_error']:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print("-" * 50)

    model.load_state_dict(best_state)
    save_model(model)
    print(f"Model saved to {log_dir / 'cnn_planner.th'}")
    torch.save(model.state_dict(), log_dir / "cnn_planner.th")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, default="cnn_planner") 
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2024)

    args = parser.parse_args()
    train_cnn(**vars(args))
    