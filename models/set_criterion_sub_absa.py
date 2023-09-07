import torch.nn.functional as F
import torch.nn as nn
import torch, math
from models.matcher import HungarianMatcher, HungarianMatcher_absa, HungarianMatcher_sub_absa
from pdb import set_trace as stop


class SetCriterion_sub_absa(nn.Module):
    def __init__(self, num_classes, na_coef, losses, matcher):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = HungarianMatcher_sub_absa(matcher)
        self.losses = losses
        # rel_weight = torch.ones(self.num_classes)
        # rel_weight[0] = na_coef
        # self.register_buffer('rel_weight', rel_weight)
        compare_weight = torch.ones(self.num_classes)
        compare_weight[0] = na_coef
        self.register_buffer('compare_weight', compare_weight)

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
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices)) # 去除了relation不为空的判断条件
        losses = sum(losses[k] for k in losses.keys())

        return losses
    
    def compare_loss(self, outputs, targets, indices):
        """Classification loss (NLL)
        targets dicts must contain the key "relation" containing a tensor of dim [bsz]
        """
        src_logits = outputs['pred_rel_logits'] # [bsz, num_generated_triples, num_rel]
        idx = self._get_src_permutation_idx(indices)

        all_empty = all([not (t[0].numel() or t[1].numel()) for t in indices]) # 判断indices中的元素是否全部空，如果是则loss=0,不全为空的话，即正常计算
        if all_empty == False:
            target_classes_o = torch.cat([torch.full_like(i, 1, dtype=torch.int64, device=src_logits.device) for t, (_, i) in zip(targets, indices) if i.numel() != 0 ]) 
            target_classes = torch.full(src_logits.shape[:2], 0,
                                        dtype=torch.int64, device=src_logits.device) # target_classes.shape=bsz, num_generated_triples
            
            target_classes[idx] = target_classes_o
            loss = F.cross_entropy(src_logits.flatten(0, 1), target_classes.flatten(0, 1), weight=self.compare_weight)
        else:
            loss = torch.tensor(float(0), requires_grad=True)
        losses = {'compare_loss': loss}
        return losses
    
    def relation_loss(self, outputs, targets, indices):
        """Classification loss (NLL)
        targets dicts must contain the key "relation" containing a tensor of dim [bsz]
        """
        src_logits = outputs['pred_rel_logits'] # [bsz, num_generated_triples, num_rel]
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["relation"][i] for t, (_, i) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], 0,
                                    dtype=torch.int64, device=src_logits.device)
        # bsz, num_generated_triples
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
            'entity_absa': self.entity_absa_loss,
            'entity_sub_absa': self.entity_sub_absa_loss,
            'compare_loss': self.compare_loss
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

    def entity_sub_absa_loss(self, outputs, targets, indices):
        """Compute the losses related to the position of head entity or tail entity
        """
        all_empty = all([not (t[0].numel() or t[1].numel()) for t in indices])
        if all_empty == True: # 如果all_empty=True,loss=nan,所以需要处理
            losses = {
                'sub': torch.tensor(float(0), requires_grad=True), 
                'obj': torch.tensor(float(0), requires_grad=True), 
                'aspect': torch.tensor(float(0), requires_grad=True), 
            }
        else:
            idx = self._get_src_permutation_idx(indices)
            selected_pred_sub_start = outputs["sub_start_logits"][idx]
            selected_pred_sub_end = outputs["sub_end_logits"][idx]
            selected_pred_obj_start = outputs["obj_start_logits"][idx]
            selected_pred_obj_end = outputs["obj_end_logits"][idx]
            selected_pred_aspect_start = outputs["aspect_start_logits"][idx]
            selected_pred_aspect_end = outputs["aspect_end_logits"][idx]

            target_sub_start = torch.cat([t["sub_start_index"][i] for t, (_, i) in zip(targets, indices)])
            target_sub_end = torch.cat([t["sub_end_index"][i] for t, (_, i) in zip(targets, indices)])
            target_obj_start = torch.cat([t["obj_start_index"][i] for t, (_, i) in zip(targets, indices)])
            target_obj_end = torch.cat([t["obj_end_index"][i] for t, (_, i) in zip(targets, indices)])
            target_aspect_start = torch.cat([t["aspect_start_index"][i] for t, (_, i) in zip(targets, indices)])
            target_aspect_end = torch.cat([t["aspect_end_index"][i] for t, (_, i) in zip(targets, indices)])

            sub_start_loss = F.cross_entropy(selected_pred_sub_start, target_sub_start)
            sub_end_loss = F.cross_entropy(selected_pred_sub_end, target_sub_end)
            obj_start_loss = F.cross_entropy(selected_pred_obj_start, target_obj_start)
            obj_end_loss = F.cross_entropy(selected_pred_obj_end, target_obj_end)
            aspect_start_loss = F.cross_entropy(selected_pred_aspect_start, target_aspect_start)
            aspect_end_loss = F.cross_entropy(selected_pred_aspect_end, target_aspect_end)

            losses = {
                'sub': 1/2*(sub_start_loss + sub_end_loss), 
                'obj': 1/2*(obj_start_loss + obj_end_loss), 
                'aspect': 1/2*(aspect_start_loss + aspect_end_loss), 
            }
        # stop()
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
