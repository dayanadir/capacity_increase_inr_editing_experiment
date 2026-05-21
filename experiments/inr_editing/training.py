import copy
from typing import Dict, Callable, Tuple, List, Union
import logging

import torch
import numpy as np
import tqdm
import wandb
import hydra
from einops import rearrange
from omegaconf import DictConfig
from torch import nn

from torch.optim.lr_scheduler import LRScheduler

from torch.utils.data import DataLoader, Dataset
from experiments.data.gradient_data import GradientDatasetForEditing
from experiments.data.image_processing import style_edit
from experiments.data.scalegmn_data_utils import mask_hidden, mask_input, GraphBatcher, get_batch_from_wb
from nn import ScaleGMN_equiv
from nn.dws.models import DWSModel
from experiments.utils.wandb_tag import build_tag
from experiments.utils import set_seed, set_logger, count_parameters, get_device
from nn.inr import BatchSiren

set_logger()


class ScaledDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        grad_transform: Callable,
        weight_transform: Callable,
        bias_transform: Callable,
    ):
        super().__init__()
        self.dataset = dataset
        self.grad_transform = grad_transform
        self.weight_transform = weight_transform
        self.bias_transform = bias_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        x, w, b, t, orig_t = self.dataset[idx]
        x = self.grad_transform(x)
        original_w, original_b = w, b
        w = self.weight_transform(w)
        b = self.bias_transform(b)
        return x, w, b, t, original_w, original_b, orig_t


def get_dataloaders(
    dataset_dir: str,
    image_data_path: str,
    val_size: Union[float, int],
    batch_size: int,
    normalize: bool = True,
    style_function: str = None,
):
    """
    Get dataloaders for training, validation, and testing.
    """
    train_dataset = GradientDatasetForEditing(
        dataset_dir, image_data_path=image_data_path, target_transform=style_edit(style_function), train=True
    )
    test_dataset = GradientDatasetForEditing(
        dataset_dir, image_data_path=image_data_path, target_transform=style_edit(style_function), train=False
    )

    # Filter by split marker using full path (works for nested checkpoint layout).
    test_dataset.grad_paths = list(filter(lambda p: "testing" in p.as_posix(), test_dataset.grad_paths))
    train_dataset.grad_paths = list(filter(lambda p: "training" in p.as_posix(), train_dataset.grad_paths))

    # Calculate split sizes
    total_size = len(train_dataset)
    if isinstance(val_size, float):
        val_size = int(val_size * total_size)

    train_size = total_size - val_size

    # Split dataset
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )

    if normalize:
        logging.info("Normalizing data")
        train_dataset, val_dataset, test_dataset = normalize_data(
            train_dataset, val_dataset, test_dataset
        )

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # Keep validation/test order fixed so rendered comparison images are stable across epochs.
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    logging.info(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")

    return train_loader, val_loader, test_loader, train_dataset[0]


def normalize_data(
    train_dataset: Dataset, val_dataset: Dataset, test_dataset: Dataset, max_samples: int = 1000  # todo: make configurable
) -> Tuple[Dataset, Dataset, Dataset]:

    # Collect all from training set
    all_grads = []
    all_weights = []
    all_biases = []
    for i, (grads, weights, biases, target, _) in enumerate(train_dataset):
        if i >= max_samples:
            break
        all_grads.append([b.clone() for b in grads])
        all_weights.append([w.clone() for w in weights])
        all_biases.append([b.clone() for b in biases])

    # Stack from each layer
    stacked_grads = []
    for layer_idx in range(len(all_grads[0])):
        layer_inputs = torch.cat(
            [batch[layer_idx] for batch in all_grads], dim=0
        )
        stacked_grads.append(layer_inputs)

    stacked_weights = []
    for layer_idx in range(len(all_weights[0])):
        layer_weights = torch.cat(
            [weights[layer_idx] for weights in all_weights], dim=0
        )
        stacked_weights.append(layer_weights)

    stacked_biases = []
    for layer_idx in range(len(all_biases[0])):
        layer_biases = torch.cat(
            [biases[layer_idx] for biases in all_biases], dim=0
        )
        stacked_biases.append(layer_biases)

    # Compute mean and std for each layer
    grad_means = []
    grad_stds = []
    for layer_inputs in stacked_grads:
        # todo: maybe we need to normalize on dim=(0, 1) instead of dim=0 here
        grad_means.append(layer_inputs.mean(dim=0))
        grad_stds.append(layer_inputs.std(dim=0))

    weight_means = []
    weight_stds = []
    for layer_weights in stacked_weights:
        weight_means.append(layer_weights.mean(dim=0))
        weight_stds.append(layer_weights.std(dim=0))

    bias_means = []
    bias_stds = []
    for layer_biases in stacked_biases:
        bias_means.append(layer_biases.mean(dim=0))
        bias_stds.append(layer_biases.std(dim=0))

    # Create normalization functions
    def normalize_grads(inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        normalized = []
        for input, mean, std in zip(inputs, grad_means, grad_stds):
            normalized.append((input - mean) / std)
        return normalized

    def normalize_weights(weights: List[torch.Tensor]) -> List[torch.Tensor]:
        normalized = []
        for weight, mean, std in zip(weights, weight_means, weight_stds):
            normalized.append((weight - mean) / std)
        return normalized

    def normalize_biases(biases: List[torch.Tensor]) -> List[torch.Tensor]:
        normalized = []
        for bias, mean, std in zip(biases, bias_means, bias_stds):
            normalized.append((bias - mean) / std)
        return normalized

    # Apply normalization to dataset
    train_dataset = ScaledDataset(
        train_dataset,
        grad_transform=normalize_grads,
        weight_transform=normalize_weights,
        bias_transform=normalize_biases
    )
    val_dataset = ScaledDataset(
        val_dataset, grad_transform=normalize_grads, weight_transform=normalize_weights, bias_transform=normalize_biases
    )
    test_dataset = ScaledDataset(
        test_dataset, grad_transform=normalize_grads, weight_transform=normalize_weights, bias_transform=normalize_biases
    )

    return train_dataset, val_dataset, test_dataset

#
# def loss_fn(
#     predictions: torch.Tensor, targets: torch.Tensor
# ) -> torch.Tensor:
#     return torch.nn.functional.cross_entropy(predictions, targets)


def residual_param_update(weights, biases, delta_weights, delta_biases):
    new_weights = [weights[j] + delta_weights[j] for j in range(len(weights))]
    new_biases = [biases[j] + delta_biases[j] for j in range(len(weights))]
    return new_weights, new_biases


def render_inr_image(
    inr_model: BatchSiren,
    weights: List[torch.Tensor],
    biases: List[torch.Tensor],
    image_height: int,
) -> torch.Tensor:
    """Render grayscale image(s) from INR parameters.

    Backward-compatible behavior:
    - If feature dim is 1, render exactly as before.
    - If feature dim > 1, render one image per feature and average per-pixel.
    """
    feature_dim = weights[0].shape[-1]
    if feature_dim == 1:
        pred_image = inr_model(weights, biases)
        return rearrange(pred_image, "b (h w) c -> b c h w", h=image_height)

    bs = weights[0].shape[0]
    expanded_weights = [
        w.permute(0, 3, 1, 2).reshape(bs * feature_dim, w.shape[1], w.shape[2], 1)
        for w in weights
    ]
    expanded_biases = [
        b.permute(0, 2, 1).reshape(bs * feature_dim, b.shape[1], 1) for b in biases
    ]

    pred_image = inr_model(expanded_weights, expanded_biases)
    pred_image = rearrange(pred_image, "(b f) (h w) c -> b f c h w", b=bs, f=feature_dim, h=image_height)
    return pred_image.mean(dim=1)


@torch.no_grad()
def evaluate(
        gradnet: torch.nn.Module,
        ws_model: torch.nn.Module,
        weight_scale: nn.ParameterList,
        bias_scale: nn.ParameterList,
        inr_model: BatchSiren,
        graph_batcher: GraphBatcher,
        data_loader: DataLoader,
        device: str,
        model_type: str,
        predict_delta: bool
) -> Dict:
    gradnet.eval()
    ws_model.eval()

    loss_list = []
    sample_images = None
    for neuron_features, weights, biases, target, original_w, original_b, original_target in data_loader:
        # Load batch to device
        batch = {
            "neuron_features": [t.to(device) for t in neuron_features],
            "weights": [t.to(device) for t in weights],
            "biases": [t.to(device) for t in biases],
            "targets": target.to(device),
            "original_weights": [t.to(device) for t in original_w],
            "original_biases": [t.to(device) for t in original_b],
            "original_targets": original_target.to(device),
        }

        if model_type == "graddws":
            weights, biases = gradnet(batch["neuron_features"])
            new_weights = [torch.cat([w, p], dim=-1) for w, p in
                           zip(batch["weights"], weights)]
            new_biases = [torch.cat([b, p], dim=-1) for b, p in zip(batch["biases"], biases)]

            new_weights, new_biases = ws_model((new_weights, new_biases))

        elif model_type == "gradnet":
            new_weights, new_biases = gradnet(batch["neuron_features"])

        elif model_type == "dws":
            new_weights, new_biases = ws_model((batch["weights"], batch["biases"]))

        elif model_type == "gradscalegmn":
            # gradnet + scalegmn
            weights, biases = gradnet(batch["neuron_features"])
            new_weights = [
                torch.cat([w, p], dim=-1) for w, p in zip(batch["original_weights"], weights)
            ]
            new_biases = [torch.cat([b, p], dim=-1) for b, p in zip(batch["original_biases"], biases)]

            # we need to convert the weights and biases to the format that scalegmn expects
            scalegmn_batch = get_batch_from_wb(new_weights, new_biases, graph_batcher, label=None)
            scalegmn_batch = scalegmn_batch.to(device)
            new_weights, new_biases = ws_model(scalegmn_batch, w=batch["original_weights"], b=batch["original_biases"])

        else:
            assert model_type == "scalegmn"
            scalegmn_batch = get_batch_from_wb(batch["original_weights"], batch["original_biases"], graph_batcher,
                                               label=None)
            scalegmn_batch = scalegmn_batch.to(device)
            new_weights, new_biases = ws_model(scalegmn_batch, w=batch["original_weights"], b=batch["original_biases"])

        if predict_delta:
            new_weights = [w * s for w, s in zip(new_weights, weight_scale)]
            new_biases = [b * s for b, s in zip(new_biases, bias_scale)]
            new_weights, new_biases = residual_param_update(
                batch["original_weights"], batch["original_biases"], new_weights, new_biases
            )

        pred_image = render_inr_image(
            inr_model=inr_model,
            weights=new_weights,
            biases=new_biases,
            image_height=target.shape[-2],
        )
        original_inr_image = render_inr_image(
            inr_model=inr_model,
            weights=batch["original_weights"],
            biases=batch["original_biases"],
            image_height=target.shape[-2],
        )
        # loss = torch.nn.functional.mse_loss(pred_image, batch['targets'], reduction='mean')
        loss = ((pred_image - batch['targets']) ** 2).mean(dim=(1, 2, 3))
        loss_list.append(loss.detach().cpu())

        if sample_images is None:
            sample_images = dict(
                recon_images=pred_image[:4].detach().cpu().numpy(),
                target_images=target[:4].detach().cpu().numpy(),
                original_target_images=batch["original_targets"][:4].detach().cpu().numpy(),
                original_inr_images=original_inr_image[:4].detach().cpu().numpy(),
            )

    # Handle edge-case of empty loader gracefully.
    if sample_images is None:
        sample_images = dict(
            recon_images=np.empty((0,)),
            target_images=np.empty((0,)),
            original_target_images=np.empty((0,)),
            original_inr_images=np.empty((0,)),
        )

    gradnet.train()
    ws_model.train()
    return dict(
        loss=torch.cat(loss_list).mean(),
        recon_images=sample_images["recon_images"],
        target_images=sample_images["target_images"],
        original_target_images=sample_images["original_target_images"],
        original_inr_images=sample_images["original_inr_images"],
    )


class WarmupLRScheduler(LRScheduler):
    def __init__(self, optimizer, warmup_steps=10000, last_epoch=-1, verbose=False):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch=last_epoch, verbose=verbose)

    def get_lr(self):
        if self._step_count < self.warmup_steps:
            return [
                base_lr * self._step_count / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        else:
            return self.base_lrs


@hydra.main(config_path=".", config_name="inr_editing", version_base=None)
def main(cfg: DictConfig):
    logging.info(cfg)
    set_seed(cfg.train.seed)
    device = get_device()

    if cfg.wandb.log:
        wandb.init(
            settings=wandb.Settings(start_method="thread"),
            project=cfg.wandb.project_name,
            name=build_tag(prefix=cfg.wandb.prefix, attributes={}),
            config=dict(cfg),
        )

    # Get dataloaders
    train_loader, val_loader, test_loader, sample = get_dataloaders(
        cfg.data.dir,
        image_data_path=cfg.data.image_data_path,
        val_size=cfg.data.val_size,
        batch_size=cfg.data.batch_size,
        normalize=cfg.data.normalize,
        style_function=cfg.data.style_function,
    )

    nf, weights, biases, target, _, _, _ = sample
    gradient_models = {"graddws", "gradnet", "gradscalegmn"}

    gradnet = nn.Identity().to(device)

    graph_batcher = None
    if cfg.train.model in ["graddws", "dws"]:
        weight_shapes = tuple(w.shape[:2] for w in weights)
        bias_shapes = tuple(b.shape[:1] for b in biases)
        logging.info(f"weight shapes: {weight_shapes}, bias shapes: {bias_shapes}")

        dws_cfg = cfg.dws_args
        if cfg.train.model in ["dws"]:
            dws_cfg['input_features'] = 1

        ws_model = DWSModel(
            weight_shapes=weight_shapes,
            bias_shapes=bias_shapes,
            **cfg.dws_args
        )
        ws_model = ws_model.to(device)

    else:
        # scalegmn model
        if cfg.train.model in ["gradscalegmn"]:
            cfg.scalegmn_args.d_in_v = 1 + cfg.gradnet.output_bias_feature_dim
            cfg.scalegmn_args.graph_init.d_in_v = 1 + cfg.gradnet.output_bias_feature_dim
            cfg.scalegmn_args.d_in_e = 1 + cfg.gradnet.output_weight_feature_dim
            cfg.scalegmn_args.graph_init.d_in_e = 1 + cfg.gradnet.output_weight_feature_dim

        equiv_on_hidden = mask_hidden(cfg)
        get_first_layer_mask = mask_input(cfg)

        graph_batcher = GraphBatcher(
            layer_layout=cfg.data.layer_layout,
            direction=cfg.scalegmn_args.direction,
            node_pos_embed=cfg.data.node_pos_embed,
            edge_pos_embed=cfg.data.edge_pos_embed,
            equiv_on_hidden=equiv_on_hidden,
            get_first_layer_mask=get_first_layer_mask
        )
        ws_model = ScaleGMN_equiv(cfg.scalegmn_args)
        ws_model = ws_model.to(device)

    # scaling modules
    weight_scale = nn.ParameterList(
        [
            nn.Parameter(torch.tensor(cfg.scalegmn_args.out_scale))  # same as ScaleGMN paper
            for _ in range(4)  # todo: remove hardcoding
        ]
    )
    bias_scale = nn.ParameterList(
        [
            nn.Parameter(torch.tensor(cfg.scalegmn_args.out_scale))  # same as ScaleGMN paper
            for _ in range(4)  # todo: remove hardcoding
        ]
    )

    grad_params = count_parameters(gradnet)
    ws_params = count_parameters(ws_model)
    total_model_params = grad_params + ws_params
    logging.info(f"Using model: {cfg.train.model}.")
    logging.info(f"grad model parameters: {grad_params}, dws model parameters: {ws_params}")
    if cfg.wandb.log:
        wandb.log(
            {
                "model/gradnet_params": grad_params,
                "model/ws_model_params": ws_params,
                "model/total_params": total_model_params,
            }
        )

    # Create optimizer
    optimizer = (
        hydra.utils.instantiate(cfg.optimizer, params=gradnet.parameters())
        if cfg.train.model in gradient_models
        else None
    )
    ws_optimizer = hydra.utils.instantiate(
        cfg.optimizer,
        params=list(ws_model.parameters()) + list(weight_scale.parameters()) + list(bias_scale.parameters())
    )
    lr_scheduler = (
        hydra.utils.instantiate(cfg.lr_scheduler, optimizer=optimizer)
        if optimizer is not None
        else None
    )
    ws_lr_scheduler = hydra.utils.instantiate(cfg.lr_scheduler, optimizer=ws_optimizer)

    # lr_scheduler = WarmupLRScheduler(optimizer)
    # ws_lr_scheduler = WarmupLRScheduler(ws_optimizer)

    # init batch INR
    inr_model = BatchSiren(img_shape=target.shape[-2:]).to(device)  # todo: infer other args from batch

    # Initialize best validation loss and epoch
    best_validation_loss = float("inf")
    best_validation_epoch = 0
    test_loss_on_best_validation = float("inf")

    # Train loop
    for epoch in range(1, cfg.train.epochs + 1):
        loss_list = []

        for neuron_features, weights, biases, target, original_w, original_b, original_target in (
            pbar := tqdm.tqdm(train_loader, total=len(train_loader))
        ):
            # Load batch to device
            batch = {
                "neuron_features": [t.to(device) for t in neuron_features],
                "weights": [t.to(device) for t in weights],
                "biases": [t.to(device) for t in biases],
                "targets": target.to(device),
                "original_weights": [t.to(device) for t in original_w],
                "original_biases": [t.to(device) for t in original_b],
                "original_targets": original_target.to(device),
            }

            # Zero parameter gradients
            if optimizer is not None:
                optimizer.zero_grad()
            ws_optimizer.zero_grad()

            # Forward pass
            if cfg.train.model == "graddws":
                weights, biases = gradnet(batch["neuron_features"])
                new_weights = [torch.cat([w, p], dim=-1) for w, p in zip(batch["weights"], weights)]
                new_biases = [torch.cat([b, p], dim=-1) for b, p in zip(batch["biases"], biases)]

                new_weights, new_biases = ws_model((new_weights, new_biases))

            elif cfg.train.model == "gradnet":
                new_weights, new_biases = gradnet(batch["neuron_features"])

            elif cfg.train.model == "dws":
                # dws only
                new_weights, new_biases = ws_model((batch["weights"], batch["biases"]))

            elif cfg.train.model == "gradscalegmn":
                # gradnet + scalegmn
                weights, biases = gradnet(batch["neuron_features"])
                new_weights = [
                    torch.cat([w, p], dim=-1) for w, p in zip(batch["original_weights"], weights)
                ]
                new_biases = [torch.cat([b, p], dim=-1) for b, p in zip(batch["original_biases"], biases)]

                # we need to convert the weights and biases to the format that scalegmn expects
                scalegmn_batch = get_batch_from_wb(new_weights, new_biases, graph_batcher, label=None)
                scalegmn_batch = scalegmn_batch.to(device)
                new_weights, new_biases = ws_model(
                    scalegmn_batch, w=batch["original_weights"], b=batch["original_biases"]
                )

            else:
                assert cfg.train.model == "scalegmn"
                scalegmn_batch = get_batch_from_wb(batch["original_weights"], batch["original_biases"], graph_batcher,
                                                   label=None)
                scalegmn_batch = scalegmn_batch.to(device)
                new_weights, new_biases = ws_model(
                    scalegmn_batch, w=batch["original_weights"], b=batch["original_biases"]
                )

            if cfg.train.predict_delta:
                new_weights = [w * s for w, s in zip(new_weights, weight_scale)]
                new_biases = [b * s for b, s in zip(new_biases, bias_scale)]
                new_weights, new_biases = residual_param_update(
                    batch["original_weights"], batch["original_biases"], new_weights, new_biases
                )

            # Compute loss
            pred_image = render_inr_image(
                inr_model=inr_model,
                weights=new_weights,
                biases=new_biases,
                image_height=target.shape[-2],
            )
            loss = torch.nn.functional.mse_loss(pred_image, batch['targets'], reduction='mean')

            # Backward pass and optimize
            loss.backward()
            if optimizer is not None:
                optimizer.step()
            ws_optimizer.step()

            # Log
            pbar.set_description(f"Epoch: {epoch}, batch loss: {loss.item(): .4f}")
            loss_list.append(loss.item())

            if cfg.wandb.log:
                wandb.log({"train/loss": loss.item()})

        # Compute train evaluation
        train_loss = np.mean(loss_list)
        print(f"---> Train loss: {train_loss: .4f}")

        # Compute validation evaluation
        validation_dict = evaluate(
            gradnet, ws_model, weight_scale=weight_scale, bias_scale=bias_scale,
            graph_batcher=graph_batcher, inr_model=inr_model, data_loader=val_loader,
            device=device, model_type=cfg.train.model, predict_delta=cfg.train.predict_delta
        )
        print(f"---> Validation loss: {validation_dict['loss']: .4f}")

        # Compute test evaluation
        test_dict = evaluate(
            gradnet, ws_model, weight_scale=weight_scale, bias_scale=bias_scale,
            graph_batcher=graph_batcher, inr_model=inr_model, data_loader=test_loader,
            device=device, model_type=cfg.train.model, predict_delta=cfg.train.predict_delta
        )
        print(f"---> Test loss: {test_dict['loss']: .4f}")

        # Update best validation loss and epoch
        if validation_dict['loss'] < best_validation_loss:
            best_validation_loss = validation_dict['loss']
            best_validation_epoch = epoch
            test_loss_on_best_validation = test_dict['loss']
        print(
            f"---> Test loss on best validation: {test_loss_on_best_validation: .4f}"
        )
        print(f"---> Best validation epoch: {best_validation_epoch}")

        # lr scheduler
        if lr_scheduler is not None:
            lr_scheduler.step(train_loss)
        ws_lr_scheduler.step(train_loss)

        # Log
        if cfg.wandb.log:
            wandb.log(
                {
                    "train/loss_epoch": train_loss,
                    "validation/loss_epoch": validation_dict["loss"],
                    "test/loss_epoch": test_dict["loss"],
                    "test/loss_on_best_validation": test_loss_on_best_validation,
                    "validation/best_epoch": best_validation_epoch,
                    # log images to wandb
                    "validation/recon_images": [wandb.Image(img) for img in validation_dict["recon_images"]],
                    "validation/target_images": [wandb.Image(img) for img in validation_dict["target_images"]],
                    "validation/original_target_images": [wandb.Image(img) for img in validation_dict["original_target_images"]],
                    "validation/original_inr_images": [wandb.Image(img) for img in validation_dict["original_inr_images"]],
                    "test/recon_images": [wandb.Image(img) for img in test_dict["recon_images"]],
                    "test/target_images": [wandb.Image(img) for img in test_dict["target_images"]],
                    "test/original_target_images": [wandb.Image(img) for img in test_dict["original_target_images"]],
                    "test/original_inr_images": [wandb.Image(img) for img in test_dict["original_inr_images"]],
                }
            )


if __name__ == "__main__":
    main()
