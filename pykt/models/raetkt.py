"""Raw KC responsibility and parallel Beta; set-level supervision lives in trainer.

p_k is contextual KC success evidence, not KC-level ground-truth mastery.
"""
import math

import torch
from torch import nn

from .akt import Architecture

RAETKT_VERSION = "responsibility_partial_label_mean_unit_evidence_v5"


class RAETKT(nn.Module):
    """Evidence is local to each input sequence and reset on every forward.

    The first AKT estimates KC-local probabilities without Beta input. Its
    normalized failure probabilities route interactions and update downstream
    Beta memory. Neither path detaches. A masked mean of candidate KCs forms
    the response-free current-question query, while responsibility alone routes
    student history. No sequence-wise Python recurrence is used.
    """

    def __init__(self, n_question, n_pid, d_model, n_blocks, dropout,
                 d_ff=256, kq_same=1, final_fc_dim=512, num_attn_heads=8,
                 separate_qa=False, l2=1e-5, emb_type="qid",
                 evidence_prior_alpha=1.0, evidence_prior_beta=1.0,
                 responsibility_temperature=1.0,
                 eps=1e-8):
        super().__init__()
        if n_question < 1 or n_blocks < 1:
            raise ValueError("Require n_question >= 1 and n_blocks >= 1.")
        if n_pid < 0:
            raise ValueError("n_pid must be nonnegative.")
        if d_model <= 0 or num_attn_heads <= 0 or d_model % num_attn_heads:
            raise ValueError("d_model must be positive and divisible by num_attn_heads.")
        if emb_type != "qid":
            raise ValueError("RAETKT supports emb_type='qid' only.")
        if (not math.isfinite(float(responsibility_temperature))
                or float(responsibility_temperature) <= 0):
            raise ValueError("responsibility_temperature must be finite and positive.")
        for name, value in (("evidence_prior_alpha", evidence_prior_alpha),
                            ("evidence_prior_beta", evidence_prior_beta), ("eps", eps)):
            if not math.isfinite(float(value)) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        self.model_name = "raetkt"
        self.model_type = "akt"
        self.n_question, self.n_pid = n_question, n_pid
        self.d_model, self.separate_qa = d_model, separate_qa
        self.emb_type, self.l2 = emb_type, l2
        self.evidence_prior_alpha = float(evidence_prior_alpha)
        self.evidence_prior_beta = float(evidence_prior_beta)
        self.responsibility_temperature = float(responsibility_temperature)
        self.eps = float(eps)
        self.q_embed = nn.Embedding(n_question + 1, d_model, padding_idx=0)
        self.qa_embed = (nn.Embedding(2 * (n_question + 1), d_model, padding_idx=0)
                         if separate_qa else nn.Embedding(2, d_model))
        if n_pid > 0:
            self.difficult_param = nn.Embedding(n_pid + 1, 1, padding_idx=0)
            self.q_embed_diff = nn.Embedding(n_question + 1, d_model, padding_idx=0)
            self.qa_embed_diff = nn.Embedding(2, d_model)
        self.model = Architecture(
            n_question=n_question, n_blocks=n_blocks, n_heads=num_attn_heads,
            dropout=dropout, d_model=d_model, d_feature=d_model // num_attn_heads,
            d_ff=d_ff, kq_same=kq_same, model_type="akt", emb_type=emb_type)
        self.kc_evidence = nn.Sequential(
            nn.Linear(3 * d_model, d_model), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model, 1))
        # Downstream history encoder only; never enters KC probabilities/query.
        self.beta_state_encoder = nn.Sequential(
            nn.Linear(3, d_model), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model, d_model))
        self.out = nn.Sequential(
            nn.Linear(2 * d_model, final_fc_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(final_fc_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 1))
        if n_pid > 0:
            nn.init.constant_(self.difficult_param.weight, 0.0)

    @staticmethod
    def _masked_mean(x, mask):
        valid = mask.unsqueeze(-1).to(x.dtype)
        return (x * valid).sum(dim=2) / valid.sum(dim=2).clamp_min(1)

    @staticmethod
    def _masked_weighted_sum(x, weight):
        return (x * weight.unsqueeze(-1)).sum(dim=2)

    def _masked_softmax(self, logits, mask, dim=-1):
        masked_logits = logits.masked_fill(~mask, -1e9)
        probs = torch.softmax(masked_logits, dim=dim)
        probs = probs * mask.to(probs.dtype)
        return probs / probs.sum(dim=dim, keepdim=True).clamp_min(self.eps)


    def _uniform_question_embedding(self, c_data, c_mask):
        """Average valid KC embeddings; padding contributes neither mass nor count."""
        c_emb = self.q_embed(c_data)
        return self._masked_mean(c_emb, c_mask), c_emb

    @staticmethod
    def _validate_binary(data, name):
        if torch.any((data != 0) & (data != 1)):
            raise ValueError(f"{name} must be binary.")

    @staticmethod
    def _validate_ids(data, maximum, name):
        if torch.is_floating_point(data) and torch.any(data != data.floor()):
            raise ValueError(f"{name} IDs must be integers.")
        if torch.any((data < 0) | (data > maximum)):
            raise ValueError(f"{name} IDs must be within [0, {maximum}].")

    def _wrong_responsibility(self, kc_probs, mask):
        weakness = (1.0 - kc_probs).clamp_min(self.eps)
        if self.responsibility_temperature == 1.0:
            weakness = weakness * mask.to(kc_probs.dtype)
            return weakness / weakness.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        logits = weakness.log() / self.responsibility_temperature
        return self._masked_softmax(logits, mask, dim=-1)

    def _compute_kc_evidence(self, hidden, c_emb, query, c_mask):
        shape = c_emb.shape
        features = torch.cat((hidden.unsqueeze(2).expand(shape), c_emb,
                              query.unsqueeze(2).expand(shape)), dim=-1)
        logits = self.kc_evidence(features).squeeze(-1)
        return torch.sigmoid(logits) * c_mask.to(logits.dtype)

    @staticmethod
    def _exclusive_prefix(delta):
        # Shift inclusive cumsum rather than subtracting the current increment.
        # Equivalent to V1.5's exclusive prefix without cancellation by y_t.
        return torch.cat((torch.zeros_like(delta[:, :1]),
                          torch.cumsum(delta[:, :-1], dim=1)), dim=1)

    def _responsibility_statistics(self, rho_wrong, c_mask):
        # DIAGNOSTIC ONLY. DO NOT use effective_k for evidence updates,
        # interaction routing, prediction or loss.
        entropy = -(rho_wrong * rho_wrong.clamp_min(self.eps).log()).sum(-1)
        effective_k = entropy.exp() * c_mask.any(-1).to(rho_wrong.dtype)
        return entropy, effective_k

    def _build_responsibility_trajectory(self, c_data, c_mask, target,
                                         evidence_update_mask, kc_probs):
        batch, steps, _ = c_data.shape
        # Cast preserves autograd; accumulate counts in at least float32.
        probs = kc_probs.float() if kc_probs.dtype in (torch.float16, torch.bfloat16) else kc_probs
        rho_wrong = self._wrong_responsibility(probs, c_mask)
        valid = c_mask.to(probs.dtype)
        uniform = valid / valid.sum(-1, keepdim=True).clamp_min(1)
        response = target.to(probs.dtype).unsqueeze(-1)
        route = torch.where(response.bool(), uniform, rho_wrong)
        update = (evidence_update_mask & c_mask.any(-1)).unsqueeze(-1)
        # Every valid item contributes one unit of evidence regardless of K:
        # correct credit is shared uniformly, wrong credit follows rho.
        success_increment = response * uniform * update
        failure_increment = (1 - response) * rho_wrong * update
        source = torch.stack((success_increment, failure_increment), dim=-1)
        indices = c_data.unsqueeze(-1).expand(-1, -1, -1, 2)
        delta = source.new_zeros((batch, steps, self.n_question + 1, 2)).scatter_add(
            2, indices, source)
        candidate_prefix = self._exclusive_prefix(delta).gather(2, indices)
        success = self.evidence_prior_alpha + candidate_prefix[..., 0]
        failure = self.evidence_prior_beta + candidate_prefix[..., 1]
        concentration = success + failure
        mastery = success / concentration
        strength = self.evidence_prior_alpha + self.evidence_prior_beta
        stats = torch.stack((mastery, torch.log(concentration / strength),
                             strength / concentration), dim=-1)
        state_emb = self.beta_state_encoder(stats.to(self.q_embed.weight.dtype))
        state_emb = state_emb * valid.unsqueeze(-1)
        # Affected diagnostics also use a vectorized exclusive prefix.
        wrong_multi = (target == 0) & (c_mask.sum(-1) >= 2)
        seen_delta = torch.zeros((batch, steps, self.n_question + 1),
                                 device=c_data.device, dtype=torch.long).scatter_add(
            2, c_data, (c_mask & wrong_multi.unsqueeze(-1)).long())
        seen = self._exclusive_prefix(seen_delta).gather(2, c_data) > 0
        entropy, effective_k = self._responsibility_statistics(rho_wrong, c_mask)
        return {
            "route_weight": route, "rho_wrong": rho_wrong, "kc_probs": kc_probs,
            "mastery_mean": mastery * valid, "success_evidence": success * valid,
            "failure_evidence": failure * valid, "success_update": success_increment,
            "failure_update": failure_increment, "beta_state_emb": state_emb,
            "affected_mask": (seen & c_mask).any(-1),
            "responsibility_entropy": entropy, "effective_k": effective_k}

    def forward(self, q_data, target, pid_data=None, c_mask=None, qtest=False,
                return_details=False, evidence_update_mask=None):
        if q_data.dim() == 2:
            q_data = q_data.unsqueeze(-1)
            if c_mask is not None and c_mask.dim() == 2:
                c_mask = c_mask.unsqueeze(-1)
        if q_data.dim() != 3 or min(q_data.shape) < 1:
            raise ValueError("Concept IDs require nonempty [B,S,K] or [B,S].")
        self._validate_ids(q_data, self.n_question, "concept")
        c_data = q_data.long()
        if c_mask is None:
            c_mask = c_data != 0
        else:
            self._validate_binary(c_mask, "c_mask")
            if c_mask.shape != c_data.shape:
                raise ValueError("Concept mask shape must match concept IDs.")
            c_mask = c_mask.bool()
            if torch.any(c_mask != (c_data != 0)):
                raise ValueError("Padding ID 0 and concept mask must agree.")
        # The dataset supplies unique candidate sets; reject duplicate updates.
        ordered = c_data.sort(dim=-1).values
        if torch.any((ordered[..., 1:] == ordered[..., :-1]) & (ordered[..., 1:] != 0)):
            raise ValueError("Each item must contain unique candidate KC IDs.")
        if target.shape != c_data.shape[:2]:
            raise ValueError("Response shape must match [B,S].")
        self._validate_binary(target, "response")
        target = target.long()
        if evidence_update_mask is None:
            evidence_update_mask = c_mask.any(-1)
        else:
            if evidence_update_mask.shape != target.shape:
                raise ValueError("Evidence update mask must match [B,S].")
            self._validate_binary(evidence_update_mask, "evidence_update_mask")
            evidence_update_mask = evidence_update_mask.bool()
        if pid_data is not None:
            if pid_data.shape != target.shape:
                raise ValueError("Question IDs must match [B,S].")
            self._validate_ids(pid_data, self.n_pid, "question")
            pid_data = pid_data.long()

        q_embed_data, c_emb = self._uniform_question_embedding(c_data, c_mask)
        if self.separate_qa:
            qa_ids = c_data + (self.n_question + 1) * target.unsqueeze(-1)
            qa_ids = qa_ids.masked_fill(~c_mask, 0)
            qa_candidates = self.qa_embed(qa_ids)
            qa_first = self._masked_mean(qa_candidates, c_mask)
        else:
            qa_candidates = c_emb
            qa_first = q_embed_data + self.qa_embed(target)
        pid_embed_data = None
        c_reg_loss = c_emb.new_zeros(())
        concept_diff = None
        if self.n_pid > 0 and pid_data is not None:
            pid_embed_data = self.difficult_param(pid_data)
            concept_diff = self.q_embed_diff(c_data)
            question_diff = self._masked_mean(concept_diff, c_mask)
            q_embed_data = q_embed_data + pid_embed_data * question_diff
            qa_first = qa_first + pid_embed_data * (
                self.qa_embed_diff(target) + question_diff)
            c_reg_loss = pid_embed_data.square().sum() * self.l2
        item_valid = c_mask.any(-1).unsqueeze(-1)
        hidden = self.model(q_embed_data, qa_first * item_valid, pid_embed_data)
        kc_probs = self._compute_kc_evidence(hidden, c_emb, q_embed_data, c_mask)
        trajectory = self._build_responsibility_trajectory(
            c_data, c_mask, target, evidence_update_mask, kc_probs)
        route = trajectory["route_weight"].to(c_emb.dtype)
        # Both routes are differentiable: KT -> routed history -> rho and
        # KT -> Beta history -> exclusive prefix -> failure credit -> rho.
        qa_embed_data = self._masked_weighted_sum(
            qa_candidates + trajectory["beta_state_emb"], route)
        if not self.separate_qa:
            qa_embed_data = qa_embed_data + self.qa_embed(target)
        if concept_diff is not None:
            qa_embed_data = qa_embed_data + pid_embed_data * (
                self.qa_embed_diff(target) + self._masked_weighted_sum(concept_diff, route))
        qa_embed_data = qa_embed_data * item_valid
        # Reuse the same response-free mean query in both passes.
        d_output = self.model(q_embed_data, qa_embed_data, pid_embed_data)
        concat_q = torch.cat([d_output, q_embed_data], dim=-1)
        preds = torch.sigmoid(self.out(concat_q).squeeze(-1))
        if return_details:
            return preds, c_reg_loss, {
                **trajectory, "rho_route": trajectory["route_weight"],
                "concat_q": concat_q, "q_embed": q_embed_data, "qa_embed": qa_embed_data,
            }
        if qtest:
            return preds, c_reg_loss, concat_q
        return preds, c_reg_loss
