import torch
from sampling_free.structures import boxlist_iou

class OvSampler(object):
    def __init__(self, batch_size_per_image, positive_fraction, num_bins=2):
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        self.num_bins = num_bins

    def __call__(self, matched_idxs, proposals, max_eg_ious):
        pos_idx, neg_idx = [], []
        for matched_idxs_per_image, proposals_per_image, max_eg_ious_per_image in zip(
                matched_idxs, proposals, max_eg_ious):
            positive = torch.nonzero(matched_idxs_per_image >= 1).squeeze(1)
            negative = torch.nonzero(matched_idxs_per_image == 0).squeeze(1)

            num_pos = int(self.batch_size_per_image * self.positive_fraction)
            # protect against not enough positive examples
            num_pos = min(positive.numel(), num_pos)
            num_neg = self.batch_size_per_image - num_pos
            # protect against not enough negative examples
            num_neg = min(negative.numel(), num_neg)
            
            if num_pos > 0 and num_neg > 0:
                iou_mat = boxlist_iou(
                    proposals_per_image[positive],
                    proposals_per_image[negative]
                )
                iou_n2p = iou_mat.max(dim=0)[0]
                # sample negatives
                iou_split = 0.5 * (iou_n2p.min() + iou_n2p.max())
                sampled_probs = torch.zeros_like(iou_n2p, device=iou_n2p.device)
                mask = iou_n2p <= iou_split
                numel_neg, numel_mask = iou_n2p.numel(), mask.sum().item()
                sampled_probs[mask] = 1 / max(numel_mask, 1)
                sampled_probs[~mask] = 1 / max(numel_neg - numel_mask, 1)
                perm2 = sampled_probs.multinomial(num_neg)
            else:
                perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]
            perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]

            pos_idx_per_image = positive[perm1]
            neg_idx_per_image = negative[perm2]

            # create binary mask from indices
            pos_idx_per_image_mask = torch.zeros_like(
                matched_idxs_per_image, dtype=torch.bool
            )
            neg_idx_per_image_mask = torch.zeros_like(
                matched_idxs_per_image, dtype=torch.bool
            )
            pos_idx_per_image_mask[pos_idx_per_image] = 1
            neg_idx_per_image_mask[neg_idx_per_image] = 1

            pos_idx.append(pos_idx_per_image_mask)
            neg_idx.append(neg_idx_per_image_mask)

        return pos_idx, neg_idx
