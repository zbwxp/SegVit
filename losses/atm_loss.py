import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.models.builder import LOSSES
from .criterion import SetCriterion
# from .criterion_point import SetCriterion_point
@LOSSES.register_module()
class ATMLoss(nn.Module):
    """ATMLoss.

    """

    def __init__(self,
                 num_classes,
                 dec_layers,
                 mask_weight=20.0,
                 dice_weight=1.0,
                 cls_weight=1.0,
                 loss_weight=1.0,
                 use_point=False):
        super(ATMLoss, self).__init__()
        weight_dict = {"loss_ce": cls_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}
        aux_weight_dict = {}
        for i in range(dec_layers - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
        if use_point:
            self.criterion = SetCriterion_point(
                num_classes,
                weight_dict=weight_dict,
                losses=["labels", "masks"],
            )
        else:
            self.criterion = SetCriterion(
                num_classes,
                weight_dict=weight_dict,
                losses=["labels", "masks"],
            )
        self.loss_weight = loss_weight

    def forward(self,
                outputs,
                label,
                ignore_index,
                ):
        """Forward function."""
        self.ignore_index = ignore_index
        targets = self.prepare_targets(label)
        losses = self.criterion(outputs, targets)

        for k in list(losses.keys()):
            if k in self.criterion.weight_dict:
                losses[k] = losses[k] * self.criterion.weight_dict[k] * self.loss_weight
            else:
                # remove this loss if not specified in `weight_dict`
                losses.pop(k)

        return losses

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            # gt_cls
            gt_cls = targets_per_image.unique()
            gt_cls = gt_cls[gt_cls != self.ignore_index]
            masks = []
            for cls in gt_cls:
                masks.append(targets_per_image == cls)
            if len(gt_cls) == 0:
                masks.append(targets_per_image == self.ignore_index)

            masks = torch.stack(masks, dim=0)
            new_targets.append(
                {
                    "labels": gt_cls,
                    "masks": masks,
                }
            )
        return new_targets