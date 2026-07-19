from collections import defaultdict

import torch
import torch.nn.functional as F

from model_ave_backbone import AveBackbone


def build_relation_role_prior(
    triples,
    num_ent,
    num_rel,
    similar_roles=4,
    direct_weight=0.5,
):
    """Build the train-only relation domain/range prior used by the final model."""
    if not 0.0 <= direct_weight <= 1.0:
        raise ValueError("direct_weight must be between 0 and 1")
    role_count = 2 * int(num_rel)
    counts = torch.zeros(role_count, int(num_ent), dtype=torch.float32)
    for head, relation, tail in triples:
        counts[relation, tail] += 1.0
        counts[relation + num_rel, head] += 1.0

    incidence = counts.gt(0).float()
    normalized = incidence / incidence.norm(dim=1, keepdim=True).clamp_min(1.0)
    similarity = normalized @ normalized.transpose(0, 1)
    similarity.fill_diagonal_(-torch.inf)
    neighbor_count = min(max(0, int(similar_roles)), max(0, role_count - 1))
    if neighbor_count:
        values, indices = similarity.topk(neighbor_count, dim=1)
        values = values.clamp_min(0.0)
        weights = values / values.sum(dim=1, keepdim=True).clamp_min(1e-12)
        smoothed = (weights.unsqueeze(-1) * incidence[indices]).sum(dim=1)
    else:
        smoothed = torch.zeros_like(counts)

    direct = torch.log1p(counts)
    direct = direct / direct.amax(dim=1, keepdim=True).clamp_min(1.0)
    prior = direct_weight * direct + (1.0 - direct_weight) * smoothed
    return prior.clamp_(0.0, 1.0)


def role_prior_statistics(prior, triples, num_rel):
    if not triples:
        return {
            "coverage": 0.0,
            "forward_coverage": 0.0,
            "inverse_coverage": 0.0,
            "mean_support": float(prior.gt(0).sum(dim=1).float().mean()),
        }
    forward = [
        float(prior[relation, tail] > 0)
        for head, relation, tail in triples
    ]
    inverse = [
        float(prior[relation + num_rel, head] > 0)
        for head, relation, tail in triples
    ]
    forward_coverage = sum(forward) / len(forward)
    inverse_coverage = sum(inverse) / len(inverse)
    return {
        "coverage": 0.5 * (forward_coverage + inverse_coverage),
        "forward_coverage": forward_coverage,
        "inverse_coverage": inverse_coverage,
        "mean_support": float(prior.gt(0).sum(dim=1).float().mean()),
    }


class Ave(AveBackbone):
    """Ave backbone plus detached relation-role calibration."""

    def __init__(
        self,
        num_ent,
        num_rel,
        role_prior,
        max_role_strength=2.0,
        *args,
        **kwargs,
    ):
        super().__init__(
            num_ent=num_ent,
            num_rel=num_rel,
            *args,
            **kwargs,
        )
        expected_shape = (2 * self.num_rel, self.num_ent)
        if tuple(role_prior.shape) != expected_shape:
            raise ValueError(
                f"role_prior shape must be {expected_shape}, got {tuple(role_prior.shape)}"
            )
        self.register_buffer("role_prior", role_prior.float())
        self.raw_role_strength = torch.nn.Parameter(
            torch.zeros(2 * self.num_rel)
        )
        self.max_role_strength = float(max_role_strength)

    def role_strength(self):
        return self.max_role_strength * torch.tanh(self.raw_role_strength)

    def query_role_ids(self, queries):
        relation_ids = queries[:, 1] - self.num_ent
        mask_id = self.num_ent + self.num_rel
        predicts_head = queries[:, 0].eq(mask_id)
        return relation_ids + predicts_head.long() * self.num_rel

    def score_components(self, entities, relations, queries):
        base_scores = self.score(entities, relations, queries)
        role_ids = self.query_role_ids(queries)
        strength = self.role_strength()[role_ids].unsqueeze(1)
        role_bias = strength * self.role_prior[role_ids]
        return base_scores, role_bias

    def role_regularization(self):
        return self.role_strength().square().mean()

    @torch.no_grad()
    def role_strength_summary(self):
        values = self.role_strength()
        forward = values[: self.num_rel]
        inverse = values[self.num_rel :]
        return {
            "forward_mean": float(forward.mean()),
            "inverse_mean": float(inverse.mean()),
            "maximum": float(values.abs().max()),
        }

    def finegrained_contrastive_loss(self, first_view):
        """The fine-grained contrastive objective used by the backbone."""
        entity_token = self.ent_token.expand(self.num_ent, -1, -1)
        structure = (
            self.embdr(self.str_ent_ln(self.ent_embeddings)) + self.pos_str_ent
        )
        visual = self.visdr(
            self.vis_ln(
                self.proj_ent_vis(
                    self.visual_token_embedding(self.visual_token_index)
                )
            )
        ) + self.pos_vis_ent
        text = self.txtdr(
            self.txt_ln(
                self.proj_ent_txt(
                    self.text_token_embedding(self.text_token_index)
                )
            )
        ) + self.pos_txt_ent
        sequence = torch.cat([entity_token, structure, visual, text], dim=1)
        encoded = self.ent_encoder(
            sequence, src_key_padding_mask=self.ent_mask
        )
        views = [
            torch.cat([encoded[:, 0], self.lp_token], dim=0),
            torch.cat([encoded.mean(dim=1), self.lp_token], dim=0),
            torch.cat(
                [
                    encoded[:, 2 : 2 + self.num_vis].mean(dim=1),
                    self.lp_token,
                ],
                dim=0,
            ),
            torch.cat(
                [
                    encoded[:, 2 + self.num_vis : -1].mean(dim=1),
                    self.lp_token,
                ],
                dim=0,
            ),
        ]
        count = min(self.num_con, first_view.shape[0])
        selected = torch.randperm(
            first_view.shape[0], device=first_view.device
        )[:count]
        labels = torch.arange(count, device=first_view.device)
        loss = first_view.new_zeros(())
        for view in views:
            first_selected = F.normalize(first_view[selected], dim=-1)
            second_selected = F.normalize(view[selected], dim=-1)
            logits = (
                first_selected @ second_selected.transpose(0, 1)
            ) / 0.5
            loss = loss + F.cross_entropy(logits, labels)
        return loss / len(views)


def build_train_filter(triples):
    train_filter = defaultdict(set)
    for head, relation, tail in triples:
        train_filter[(-1, relation, tail)].add(head)
        train_filter[(head, relation, -1)].add(tail)
    return train_filter
