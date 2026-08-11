import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms

from detr.main import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer
import IPython
e = IPython.embed

from roma.mappings import special_gramschmidt


def _build_image_normalizer(image_channels, normalization_mode='imagenet'):
    normalization_mode = str(normalization_mode)
    if normalization_mode in ('signed_event_u8_centered', 'shifted_3chef_centered'):
        if image_channels <= 0:
            raise ValueError(f"image_channels must be positive, got {image_channels}")
        mean = [128.0 / 255.0] * image_channels
        std = [127.0 / 255.0] * image_channels
        return transforms.Normalize(mean=mean, std=std)
    if normalization_mode != 'imagenet':
        raise ValueError(f"Unsupported image_normalization mode: {normalization_mode!r}")

    base_mean = [0.485, 0.456, 0.406]
    base_std = [0.229, 0.224, 0.225]
    if image_channels == 1:
        # Collapse ImageNet RGB stats to one channel for event-only mode.
        mean = [sum(base_mean) / 3.0]
        std = [sum(base_std) / 3.0]
    elif image_channels == 3:
        mean = base_mean
        std = base_std
    elif image_channels in (6, 9):
        repeats = image_channels // 3
        mean = base_mean * repeats
        std = base_std * repeats
    else:
        raise ValueError(f"Unsupported image_channels={image_channels}")
    return transforms.Normalize(mean=mean, std=std)


def _collect_backbone_conv1_in_channels(model):
    in_channels = []
    model_backbones = getattr(model, 'backbones', None)
    if model_backbones is None:
        return in_channels

    for backbone in model_backbones:
        try:
            conv1 = backbone[0].body.conv1
            in_channels.append(int(conv1.in_channels))
        except Exception:
            continue
    return in_channels


def _assert_backbone_channels(model, expected_channels):
    channels = _collect_backbone_conv1_in_channels(model)
    if not channels:
        return
    if any(channel != expected_channels for channel in channels):
        raise AssertionError(
            f"Backbone conv1 in_channels mismatch. expected={expected_channels}, actual={channels}"
        )
    print(f"[INFO] Backbone conv1 in_channels={channels}, expected={expected_channels}")

class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
        # Most tasks use the last action dim as a binary gripper channel.
        # Toy 2D control uses fully continuous actions, so disable BCE on last dim.
        self.use_bce_last_action_dim = args_override.get(
            'use_bce_last_action_dim',
            False
        )
        self.image_channels = int(args_override.get('image_channels', 3))
        self.input_modality = args_override.get('input_modality', 'rgb')
        self.image_normalization = args_override.get('image_normalization', 'imagenet')
        if self.input_modality == 'sparse_ball':
            mean = torch.as_tensor(args_override['sparse_mean'], dtype=torch.float32)
            std = torch.as_tensor(args_override['sparse_std'], dtype=torch.float32)
            if mean.shape != (4,) or std.shape != (4,):
                raise ValueError("Sparse normalization statistics must have shape (4,)")
            self.register_buffer('sparse_mean', mean)
            self.register_buffer('sparse_std', std.clamp_min(1e-4))
            self.normalize = None
        else:
            self.normalize = _build_image_normalizer(self.image_channels, self.image_normalization)
        _assert_backbone_channels(self.model, self.image_channels)
        print(f'KL Weight {self.kl_weight}')

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None
        if self.input_modality == 'sparse_ball':
            image = (image - self.sparse_mean) / self.sparse_std
        else:
            image = self.normalize(image)
        if actions is not None: # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)
            # kl divergence loss
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar) # train with CVAE encoder
            if self.use_bce_last_action_dim:
                # position l1 loss
                all_l1 = F.l1_loss(actions[:,:,:-1], a_hat[:,:,:-1], reduction='none')
                l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
                # gripper state binary cross entropy loss
                all_bce = F.binary_cross_entropy_with_logits(a_hat[:,:,-1:], actions[:,:,-1:], reduction='none')
                bce = (all_bce * ~is_pad.unsqueeze(-1)).mean()
            else:
                # Fully continuous action tasks (e.g. toy 2D control).
                all_l1 = F.l1_loss(actions, a_hat, reduction='none')
                l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
                bce = torch.zeros((), device=actions.device)
            # total loss
            loss_dict = dict()
            loss_dict['action_l1'] = l1
            loss_dict['hand'] = bce
            loss_dict['kl'] = total_kld[0]  # train with CVAE encoder
            loss_dict['loss'] = loss_dict['action_l1'] + loss_dict['hand'] * 0.1 + loss_dict['kl'] * self.kl_weight  # train with CVAE encoder
            # loss_dict['loss'] = loss_dict['l1'] # train without CVAE encoder
            return loss_dict
        else: # inference time
            a_hat, _, (_, _) = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer

class ACTTaskPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
        self.image_channels = int(args_override.get('image_channels', 3))
        self.image_normalization = args_override.get('image_normalization', 'imagenet')
        self.normalize = _build_image_normalizer(self.image_channels, self.image_normalization)
        _assert_backbone_channels(self.model, self.image_channels)
        print(f'KL Weight {self.kl_weight}')

    def __call__(self, pose, image, actions=None, is_pad=None):
        env_state = None
        image = self.normalize(image)
        if actions is not None: # training time
            # change 9D rotation to 6D GSO representation
            pose_6D_rot = torch.cat([pose[:,:6], pose[:,9:]], dim=-1)
            actions_6D_rot = torch.cat([actions[:,:,:6], actions[:,:,9:]], dim=-1)
            # model output
            a_hat, is_pad_hat, (mu, logvar) = self.model(pose_6D_rot, image, env_state, actions_6D_rot, is_pad)
            # kl divergence loss
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            # position l1 loss
            pos_all_l1 = F.l1_loss(a_hat[:,:,6:9], actions[:,:,9:12], reduction='none')
            pos_l1 = (pos_all_l1 * ~is_pad.unsqueeze(-1)).mean()
            # 6D GSO rotation l2 loss
            batch, chunk, _ = a_hat.shape
            R = special_gramschmidt(a_hat[:,:,:6].reshape(batch, chunk, 2, 3).transpose(-1,-2))
            rot_all_l2 = F.mse_loss(R.transpose(-1,-2).flatten(-2,-1), actions[:,:,:9], reduction='none')
            rot_l2 = (rot_all_l2 * ~is_pad.unsqueeze(-1)).mean()
            # hand state binary cross entropy loss
            hand_all_bce = F.binary_cross_entropy_with_logits(a_hat[:,:,9:], actions[:,:,12:], reduction='none')
            hand_bce = (hand_all_bce * ~is_pad.unsqueeze(-1)).mean()
            # total loss
            loss_dict = dict()
            loss_dict['pos'] = pos_l1
            loss_dict['rot'] = rot_l2
            loss_dict['hand'] = hand_bce
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['pos'] + loss_dict['rot'] * 100 + loss_dict['hand'] * 0.1 + loss_dict['kl'] * self.kl_weight
            return loss_dict
        else: # inference time
            # change 9D rotation to 6D GSO representation
            pose_6D_rot = torch.cat([pose[:,:6], pose[:,9:]], dim=-1)
            a_hat, _, (_, _) = self.model(pose_6D_rot, image, env_state) # no action, sample from prior
            return a_hat
            '''
            # change to 9D rotation and apply sigmoid to hand state output
            pos = a_hat[:,:,6:9]
            batch, chunk, _ = a_hat.shape
            R = special_gramschmidt(a_hat[:,:,:6].reshape(batch, chunk, 2, 3).transpose(-1,-2))
            rot = R.transpose(-1,-2).flatten(-2,-1)
            hand = F.sigmoid(a_hat[:,:,9:])
            return torch.cat([rot, pos, hand], dim=-1)
            '''

    def configure_optimizers(self):
        return self.optimizer


class CNNMLPPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model # decoder
        self.optimizer = optimizer
        self.image_channels = int(args_override.get('image_channels', 3))
        self.image_normalization = args_override.get('image_normalization', 'imagenet')
        self.normalize = _build_image_normalizer(self.image_channels, self.image_normalization)
        _assert_backbone_channels(self.model, self.image_channels)

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None # TODO
        image = self.normalize(image)
        if actions is not None: # training time
            actions = actions[:, 0]
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)
            loss_dict = dict()
            loss_dict['mse'] = mse
            loss_dict['loss'] = loss_dict['mse']
            return loss_dict
        else: # inference time
            a_hat = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld
