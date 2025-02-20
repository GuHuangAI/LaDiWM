import os
import sys
sys.path.append('/home/huang/code/ATM')
import hydra
import torch, torchvision
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm import tqdm
import lightning
from lightning.fabric import Fabric

from ladiwm.model import *
from ladiwm.dataloader import ATMPretrainDataset5, get_dataloader
from ladiwm.utils.log_utils import BestAvgLoss, MetricLogger
from ladiwm.utils.train_utils import init_wandb, setup_lr_scheduler, setup_optimizer

@hydra.main(config_path="../conf/train_track_transformer", version_base="1.3")
def main(cfg: DictConfig):
    work_dir = HydraConfig.get().runtime.output_dir
    setup(cfg)
    OmegaConf.save(config=cfg, f=os.path.join(work_dir, "config.yaml"))

    fabric = Fabric(accelerator="cuda", devices=list(cfg.train_gpus), precision="bf16-mixed" if cfg.mix_precision else None, strategy="deepspeed")
    fabric.launch()

    None if (cfg.dry or not fabric.is_global_zero) else init_wandb(cfg)

    train_dataset = ATMPretrainDataset5(dataset_dir=cfg.train_dataset, **cfg.dataset_cfg, aug_prob=cfg.aug_prob)
    train_loader = get_dataloader(train_dataset, mode="train", num_workers=cfg.num_workers, batch_size=cfg.batch_size)

    train_vis_dataset = ATMPretrainDataset5(dataset_dir=cfg.train_dataset, vis=True, **cfg.dataset_cfg, aug_prob=cfg.aug_prob)
    train_vis_dataloader = get_dataloader(train_vis_dataset, mode="train", num_workers=1, batch_size=1)

    val_dataset = ATMPretrainDataset5(dataset_dir=cfg.val_dataset, **cfg.dataset_cfg, aug_prob=0.)
    val_loader = get_dataloader(val_dataset, mode="val", num_workers=cfg.num_workers, batch_size=cfg.batch_size * 2)

    val_vis_dataset = ATMPretrainDataset5(dataset_dir=cfg.val_dataset, vis=True, **cfg.dataset_cfg, aug_prob=0.)
    val_vis_dataloader = get_dataloader(val_vis_dataset, mode="val", num_workers=1, batch_size=1)

    # model_cls = eval(cfg.model_name)
    # model = model_cls(**cfg.model_cfg)
    from ladiwm.utils.build_utils import construct_class_by_name
    vae = construct_class_by_name(**cfg.vae_cfg)
    transformer = construct_class_by_name(**cfg.trans_cfg)
    model_kwargs = OmegaConf.to_container(cfg.model_cfg, resolve=True)
    model_kwargs.update({'vae': vae})
    model_kwargs.update({'transformer': transformer})
    model = construct_class_by_name(**model_kwargs)
    del model_kwargs
    optimizer = setup_optimizer(cfg.optimizer_cfg, model)
    scheduler = setup_lr_scheduler(optimizer, cfg.scheduler_cfg)

    model, optimizer = fabric.setup(model, optimizer)
    train_loader = fabric.setup_dataloaders(train_loader)

    lbd_track = cfg.lbd_track
    lbd_img = cfg.lbd_img
    p_img = cfg.p_img

    # Pick ckpt based on  the average of the last 5 epochs
    metric_logger = MetricLogger(delimiter=" ")
    best_loss_logger = BestAvgLoss(window_size=5)

    for epoch in metric_logger.log_every(range(cfg.epochs), 1, ""):
        # val_metrics, out_img = evaluate(
        #     model,
        #     val_loader,
        #     lbd_track=lbd_track,
        #     lbd_img=lbd_img,
        #     p_img=p_img,
        #     mix_precision=cfg.mix_precision,
        #     tag="val",
        # )
        # print(val_metrics)
        train_metrics = run_one_epoch(
            fabric,
            model,
            train_loader,
            optimizer,
            lbd_track=lbd_track,
            lbd_img=lbd_img,
            p_img=p_img,
            scheduler=scheduler,
            mix_precision=cfg.mix_precision,
            clip_grad=cfg.clip_grad,
            epoch=epoch,
        )

        train_metrics["train/lr"] = optimizer.param_groups[0]["lr"]
        metric_logger.update(**train_metrics)

        if fabric.is_global_zero:
            None if cfg.dry else wandb.log(train_metrics, step=epoch)

            if epoch % cfg.val_freq == 0:
                val_metrics, out_img = evaluate(
                    model,
                    val_loader,
                    lbd_track=lbd_track,
                    lbd_img=lbd_img,
                    p_img=p_img,
                    mix_precision=cfg.mix_precision,
                    tag="val",
                )
                # Save img
                save_dir = os.path.join(work_dir, f'vis_{epoch}')
                os.makedirs(save_dir, exist_ok=True)
                for i in range(out_img.shape[0]):
                    torchvision.utils.save_image(out_img[i], os.path.join(save_dir, f'batch_{i}.png'))

                # Save best checkpoint
                metric_logger.update(**val_metrics)

                val_metrics = {**val_metrics}
                loss_metric = val_metrics["val/loss"]
                is_best = best_loss_logger.update_best(loss_metric, epoch)

                if is_best:
                    model.save(f"{work_dir}/model_best.ckpt")
                    with open(f"{work_dir}/best_epoch.txt", "w") as f:
                        f.write(
                            "Best epoch: %d, Best %s: %.4f"
                            % (epoch, "loss", best_loss_logger.best_loss)
                        )
                None if cfg.dry else wandb.log(val_metrics, step=epoch)

            if epoch % cfg.save_freq == 0:
                model.save(f"{work_dir}/model_{epoch}.ckpt")

                def vis_and_log(model, vis_dataloader, mode="train"):
                    vis_dict = visualize(model, vis_dataloader, mix_precision=cfg.mix_precision)

                    caption = f"reconstruction (right) @ epoch {epoch}; \n Track MSE: {vis_dict['track_loss']:.4f}"
                    wandb_vis_track = wandb.Video(vis_dict["combined_track_vid"], fps=10, format="mp4", caption=caption)
                    None if cfg.dry else wandb.log({f"{mode}/reconstruct_track": wandb_vis_track}, step=epoch)

                # vis_and_log(model, train_vis_dataloader, mode="train")
                # vis_and_log(model, val_vis_dataloader, mode="val")

    if fabric.is_global_zero:
        model.save(f"{work_dir}/model_final.ckpt")
        None if cfg.dry else print(f"finished training in {wandb.run.dir}")
        None if cfg.dry else wandb.finish()


def run_one_epoch(fabric,
                  model,
                  dataloader,
                  optimizer,
                  lbd_track,
                  lbd_img,
                  p_img,
                  mix_precision=False,
                  scheduler=None,
                  clip_grad=1.0,
                  epoch=1000):
    """
    Optimize the policy. Return a dictionary of the loss and any other metrics.
    """
    track_loss, vid_loss, tot_loss, tot_items = 0, 0, 0, 0

    model.train()
    i = 0
    for d_id, (his, preds, track, vis, task_emb, actions) in tqdm(enumerate(dataloader)):
        if mix_precision:
            his, preds, track, vis, task_emb, actions = \
                his.bfloat16(), preds.bfloat16(), track.bfloat16(), \
                    vis.bfloat16(), task_emb.bfloat16(), actions.bfloat16()
        # b, t, c, h, w = vid.shape
        b, tl, n, _ = track.shape
        b, tl, n = vis.shape
        loss, ret_dict = model.forward_loss(
            his,
            preds,
            track,
            task_emb,
            actions,
            lbd_track=lbd_track,
            lbd_img=lbd_img,
            p_img=p_img,
            epoch=epoch)  # do not use vis
        if d_id % 5 == 0:
            print(ret_dict)
        optimizer.zero_grad()
        fabric.backward(loss)

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        optimizer.step()

        track_loss += ret_dict["track_loss"]
        # vid_loss += ret_dict["img_loss"]
        tot_loss += ret_dict["loss"]
        tot_items += b

        i += 1

    out_dict = {
        "train/track_loss": track_loss / tot_items,
        # "train/vid_loss": vid_loss / tot_items,
        "train/loss": tot_loss / tot_items,
    }

    if scheduler is not None:
        scheduler.step()

    return out_dict

@torch.no_grad()
def evaluate(model, dataloader, lbd_track, lbd_img, p_img, mix_precision=False, tag="val"):
    track_loss, vid_loss, tot_loss, tot_items = 0, 0, 0, 0
    model.eval()

    i = 0
    for his, preds, track, vis, task_emb, actions in tqdm(dataloader):
        his, preds, track, vis, task_emb, actions = his.cuda(), preds.cuda(), track.cuda(), \
            vis.cuda(), task_emb.cuda(), actions.cuda()
        if mix_precision:
            his, preds, track, vis, task_emb, actions = \
                his.bfloat16(), preds.bfloat16(), track.bfloat16(), \
                    vis.bfloat16(), task_emb.bfloat16(), actions.bfloat16()
            # vid, track, vis, task_emb, masks = vid.bfloat16(), track.bfloat16(), vis.bfloat16(), task_emb.bfloat16(), masks.bfloat16()
        # b, t, c, h, w = vid.shape
        b, tl, n, _ = track.shape

        _, ret_dict, out_img = model.evaluate_loss(
            his,
            preds,
            track,
            task_emb,
            actions,
            lbd_track=lbd_track,
            lbd_img=lbd_img,
            p_img=p_img,
            vis=vis)
        # loss, ret_dict = model.forward_loss(
        #     his,
        #     preds,
        #     track,
        #     task_emb,
        #     actions,
        #     lbd_track=lbd_track,
        #     lbd_img=lbd_img,
        #     p_img=p_img,
        #     )  # do not use vis

        track_loss += ret_dict["track_loss"]
        # vid_loss += ret_dict["img_loss"]
        tot_loss += ret_dict["loss"]
        tot_items += b

        i += 1
        if i >= 50:
            break

    out_dict = {
        f"{tag}/track_loss": track_loss / tot_items,
        # f"{tag}/vid_loss": vid_loss / tot_items,
        f"{tag}/loss": tot_loss / tot_items,
    }
    out_img = torch.rand(8, 3, 128, 128)
    return out_dict, out_img

@torch.no_grad()
def visualize(model, dataloader, mix_precision=False):
    model.eval()
    keep_eval_dict = None

    for i, (vid, track, vis, task_emb) in enumerate(dataloader):
        vid, track, task_emb = vid.cuda(), track.cuda(), task_emb.cuda()
        if mix_precision:
            vid, track, task_emb = vid.bfloat16(), track.bfloat16(), task_emb.bfloat16()
        _, eval_dict = model.forward_vis(vid, track, task_emb, p_img=0)
        if keep_eval_dict is None or torch.rand(1) < 0.1:
            keep_eval_dict = eval_dict

        if i == 10:
            break
    return keep_eval_dict

def setup(cfg):
    import warnings

    warnings.simplefilter("ignore")

    lightning.seed_everything(cfg.seed)


if __name__ == "__main__":
    main()
