import torch.nn.functional as F
import torch.nn as nn
import torch, math
from models.matcher import HungarianMatcher, HungarianMatcher_absa
from pdb import set_trace as stop

class SetCriterion(nn.Module):
    """ This class computes the loss for Set_RE.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class, subject position and object position)
    """
    def __init__(self, num_classes, na_coef, losses, matcher):
        """ Create the criterion.
        Parameters:
            num_classes: number of relation categories
            matcher: module able to compute a matching between targets and proposals
            loss_weight: dict containing as key the names of the losses and as values their relative weight.
            na_coef: list containg the relative classification weight applied to the NA category and positional classification weight applied to the [SEP]
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = HungarianMatcher(matcher)
        self.losses = losses
        rel_weight = torch.ones(self.num_classes)
        rel_weight[0] = na_coef
        self.register_buffer('rel_weight', rel_weight)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs, targets)
        # Compute all the requested losses
        # stop()
        losses = {}
        for loss in self.losses:
            if loss == "entity" and self.empty_targets(targets):
                pass
            else:
                losses.update(self.get_loss(loss, outputs, targets, indices))
        losses = sum(losses[k] for k in losses.keys())
        return losses

    def relation_loss(self, outputs, targets, indices):
        """Classification loss (NLL)
        targets dicts must contain the key "relation" containing a tensor of dim [bsz]
        """
        src_logits = outputs['pred_rel_logits'] # [bsz, num_generated_triples, num_rel]
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["relation"][i] for t, (_, i) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], 0,
                                    dtype=torch.int64, device=src_logits.device) # target_classes: bsz, num_generated_triples
        # stop()
        target_classes[idx] = target_classes_o
        loss = F.cross_entropy(src_logits.flatten(0, 1), target_classes.flatten(0, 1), weight=self.rel_weight)
        losses = {'relation': loss}
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty triples
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_rel_logits = outputs['pred_rel_logits']
        device = pred_rel_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_rel_logits.argmax(-1) != pred_rel_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices,  **kwargs):
        loss_map = {
            'relation': self.relation_loss,
            'cardinality': self.loss_cardinality,
            'entity': self.entity_loss,
            'entiy_absa': self.entity_absa_loss
        }
        return loss_map[loss](outputs, targets, indices, **kwargs)

    def entity_absa_loss(self, outputs, targets, indices):
        """Compute the losses related to the position of head entity or tail entity
           only compute the loss of aspect, opinion, sentiment
        """
        idx = self._get_src_permutation_idx(indices)
        # selected_pred_sub_start = outputs["sub_start_logits"][idx]
        # selected_pred_sub_end = outputs["sub_end_logits"][idx]
        # selected_pred_obj_start = outputs["obj_start_logits"][idx]
        # selected_pred_obj_end = outputs["obj_end_logits"][idx]
        selected_pred_aspect_start = outputs["aspect_start_logits"][idx]
        selected_pred_aspect_end = outputs["aspect_end_logits"][idx]
        selected_pred_opinion_start = outputs["opinion_start_logits"][idx]
        selected_pred_opinion_end = outputs["opinion_end_logits"][idx]

        # target_sub_start = torch.cat([t["sub_start_index"][i] for t, (_, i) in zip(targets, indices)])
        # target_sub_end = torch.cat([t["sub_end_index"][i] for t, (_, i) in zip(targets, indices)])
        # target_obj_start = torch.cat([t["obj_start_index"][i] for t, (_, i) in zip(targets, indices)])
        # target_obj_end = torch.cat([t["obj_end_index"][i] for t, (_, i) in zip(targets, indices)])
        target_aspect_start = torch.cat([t["aspect_start_index"][i] for t, (_, i) in zip(targets, indices)])
        target_aspect_end = torch.cat([t["aspect_end_index"][i] for t, (_, i) in zip(targets, indices)])
        target_opinion_start = torch.cat([t["opinion_start_index"][i] for t, (_, i) in zip(targets, indices)])
        target_opinion_end = torch.cat([t["opinion_end_index"][i] for t, (_, i) in zip(targets, indices)])


        # sub_start_loss = F.cross_entropy(selected_pred_sub_start, target_sub_start)
        # sub_end_loss = F.cross_entropy(selected_pred_sub_end, target_sub_end)
        # obj_start_loss = F.cross_entropy(selected_pred_obj_start, target_obj_start)
        # obj_end_loss = F.cross_entropy(selected_pred_obj_end, target_obj_end)
        aspect_start_loss = F.cross_entropy(selected_pred_aspect_start, target_aspect_start)
        aspect_end_loss = F.cross_entropy(selected_pred_aspect_end, target_aspect_end)
        opinion_start_loss = F.cross_entropy(selected_pred_opinion_start, target_opinion_start)
        opinion_end_loss = F.cross_entropy(selected_pred_opinion_end, target_opinion_end)
        losses = {
            # 'sub': 1/2*(sub_start_loss + sub_end_loss), 
            # 'obj': 1/2*(obj_start_loss + obj_end_loss), 
            'aspect': 1/2*(aspect_start_loss + aspect_end_loss), 
            'opinion': 1/2*(opinion_start_loss + opinion_end_loss), 
        }
        # print(losses)
        return losses
    
    def entity_loss(self, outputs, targets, indices):
        """Compute the losses related to the position of head entity or tail entity
        """
        idx = self._get_src_permutation_idx(indices)
        selected_pred_sub_start = outputs["sub_start_logits"][idx]
        selected_pred_sub_end = outputs["sub_end_logits"][idx]
        selected_pred_obj_start = outputs["obj_start_logits"][idx]
        selected_pred_obj_end = outputs["obj_end_logits"][idx]
        selected_pred_aspect_start = outputs["aspect_start_logits"][idx]
        selected_pred_aspect_end = outputs["aspect_end_logits"][idx]
        selected_pred_opinion_start = outputs["opinion_start_logits"][idx]
        selected_pred_opinion_end = outputs["opinion_end_logits"][idx]

        target_sub_start = torch.cat([t["sub_start_index"][i] for t, (_, i) in zip(targets, indices)])
        target_sub_end = torch.cat([t["sub_end_index"][i] for t, (_, i) in zip(targets, indices)])
        target_obj_start = torch.cat([t["obj_start_index"][i] for t, (_, i) in zip(targets, indices)])
        target_obj_end = torch.cat([t["obj_end_index"][i] for t, (_, i) in zip(targets, indices)])
        target_aspect_start = torch.cat([t["aspect_start_index"][i] for t, (_, i) in zip(targets, indices)])
        target_aspect_end = torch.cat([t["aspect_end_index"][i] for t, (_, i) in zip(targets, indices)])
        target_opinion_start = torch.cat([t["opinion_start_index"][i] for t, (_, i) in zip(targets, indices)])
        target_opinion_end = torch.cat([t["opinion_end_index"][i] for t, (_, i) in zip(targets, indices)])


        sub_start_loss = F.cross_entropy(selected_pred_sub_start, target_sub_start)
        sub_end_loss = F.cross_entropy(selected_pred_sub_end, target_sub_end)
        obj_start_loss = F.cross_entropy(selected_pred_obj_start, target_obj_start)
        obj_end_loss = F.cross_entropy(selected_pred_obj_end, target_obj_end)
        aspect_start_loss = F.cross_entropy(selected_pred_aspect_start, target_aspect_start)
        aspect_end_loss = F.cross_entropy(selected_pred_aspect_end, target_aspect_end)
        opinion_start_loss = F.cross_entropy(selected_pred_opinion_start, target_opinion_start)
        opinion_end_loss = F.cross_entropy(selected_pred_opinion_end, target_opinion_end)
        losses = {
            'sub': 1/2*(sub_start_loss + sub_end_loss), 
            'obj': 1/2*(obj_start_loss + obj_end_loss), 
            'aspect': 1/2*(aspect_start_loss + aspect_end_loss), 
            'opinion': 1/2*(opinion_start_loss + opinion_end_loss), 
        }
        # print(losses)
        return losses

    @staticmethod
    def empty_targets(targets):
        flag = True
        for target in targets:
            if len(target["relation"]) != 0:
                flag = False
                break
        return flag
