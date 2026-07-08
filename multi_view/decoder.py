"""Phase 4+ - the projective-attention decoder (pure-PyTorch MVGFormer imitation).

Phase 4 provides ProjectiveAttention: given 3D query points, gather image
features from every view at the pixels those queries project to, and aggregate
across views into one feature per query.

    query_3d   (B, Q, 3)              Q query points in the WORLD frame
    feat_maps  (B, N, C, Hf, Wf)      per-view feature maps (MultiViewBackbone)
    proj       (B, N, 3, 4)           per-view P = K[R|t]
    image_hw   (H_img, W_img)         ORIGINAL image size the proj maps into (512)

    -> feat     (B, Q, C)             one aggregated feature per query
       sampled  (B, N, Q, C)          per-view sampled features (pre-aggregation)
       uv       (B, N, Q, 2)          where each query projected in each view (px)

Iter-1 is parameter-free (mean over views). Learned per-view confidence weighting
arrives with the Phase-5 decoder layer.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from multi_view.geometry import triangulate_dlt_batch


class ProjectiveAttention(nn.Module):
    def __init__(self, d_model: int = 256) -> None:
        super().__init__()
        self.d_model = d_model  # iter1: no learnable params (mean aggregation)

    def forward(
        self,
        query_3d: torch.Tensor,      # (B, Q, 3)
        feat_maps: torch.Tensor,     # (B, N, C, Hf, Wf)
        proj: torch.Tensor,          # (B, N, 3, 4)
        image_hw: tuple[int, int],   # (H_img, W_img) the proj matrices map into
    ):
        B, N, C, Hf, Wf = feat_maps.shape
        Q = query_3d.shape[1]
        H_img, W_img = image_hw

        # same query points seen from every view -> (B, N, Q, 3)
        q = query_3d.unsqueeze(1).expand(B, N, Q, 3)

        # BLANK 1: project q into each view -> uv (B, N, Q, 2) in PIXELS.
        # Batched version of geometry.project: homogenize (append a 1 -> (B,N,Q,4)),
        # multiply by proj (per (b,n): x_h @ P.T; einsum 'bnij,bnqj->bnqi'), then
        # perspective-divide the first two coords by the third.
        ones = torch.ones(B, N, Q, 1)
        hom = torch.cat([q, ones], dim=-1)                  # (B,N,Q,4)  append the 1
        uvw = torch.einsum('bnij,bnqj->bnqi', proj, hom)    # (B,N,Q,3)  the [a,b,c]
        uv = uvw[..., :2] / uvw[..., 2:3]                   # (B,N,Q,2)  perspective divide

        # BLANK 2: normalize uv (pixels) to a grid_sample grid in [-1, 1].
        # We use align_corners=False below, whose inverse mapping is
        #     x_norm = 2 * (x_pix + 0.5) / size - 1
        # applied PER AXIS: channel 0 (u, x) uses W_img, channel 1 (v, y) uses H_img.
        # Result shape (B, N, Q, 2).
        gx = 2 * (uv[..., 0] + 0.5) / W_img - 1              # (B,N,Q)
        gy = 2 * (uv[..., 1] + 0.5) / H_img - 1              # (B,N,Q)
        grid = torch.stack([gx, gy], dim=-1)                # (B,N,Q,2)

        # --- plumbing: sample each view's feature map at its Q grid points ---
        fm = feat_maps.reshape(B * N, C, Hf, Wf)
        g = grid.reshape(B * N, 1, Q, 2)                                  # (BN,1,Q,2)
        sampled = F.grid_sample(fm, g, mode="bilinear",
                                padding_mode="zeros", align_corners=False)  # (BN,C,1,Q)
        sampled = sampled.reshape(B, N, C, Q).permute(0, 1, 3, 2)          # (B,N,Q,C)

        # BLANK 3: aggregate the per-view features into one feature per query.
        # Start simple: mean over the N view axis -> (B, Q, C).
        feat = sampled.mean(dim=1)                          # (B,Q,C)

        return feat, sampled, uv


class OffsetConfidenceHead(nn.Module):
    """Phase 5 sub-step 2 - per-view, per-query 2D offset + confidence.

    From each per-view sampled feature it predicts:
      offset (B,N,Q,2)  a pixel nudge added to that query's projected point, so
                        the 2D that feeds triangulation moves toward the image
                        evidence (not just where the current 3D guess projects).
      conf   (B,N,Q)    a non-negative weight (softplus) for that view in the
                        confidence-weighted DLT - learns to trust views where the
                        landmark is actually visible.

    Zero-init on the offset layer => the FIRST forward predicts exactly 0 offset,
    so the layer starts as a no-op refinement (a stable identity) and learns to
    deviate. Confidence starts roughly equal across views.
    """

    def __init__(self, d_model: int = 256) -> None:
        super().__init__()
        self.offset = nn.Linear(d_model, 2)
        self.conf = nn.Linear(d_model, 1)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)

    def forward(self, sampled: torch.Tensor):
        # sampled: (B, N, Q, C)
        offset = self.offset(sampled)                       # (B,N,Q,2)
        conf = F.softplus(self.conf(sampled)).squeeze(-1)   # (B,N,Q), >= 0
        return offset, conf


class QueryUpdate(nn.Module):
    """Phase 5 sub-step 3 - let the Q queries share information.

    A standard pre-norm transformer block over the query tokens (B, Q, d_model):
    multi-head self-attention (each of the 68 landmarks attends to the others -
    e.g. the two eye corners constrain each other) followed by an FFN. Both are
    residual, so at init it stays close to identity.
    """

    def __init__(self, d_model: int = 256, n_heads: int = 8,
                 ffn_dim: int = 1024, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )

    def forward(self, query_feat: torch.Tensor) -> torch.Tensor:
        # query_feat: (B, Q, d_model)
        x = self.norm1(query_feat)
        attn, _ = self.self_attn(x, x, x)                   # queries attend to queries
        query_feat = query_feat + attn                      # residual
        x = self.norm2(query_feat)
        query_feat = query_feat + self.ffn(x)               # residual FFN
        return query_feat


class DecoderLayer(nn.Module):
    """Phase 5 sub-step 4 - one refinement iteration.

        1. ProjectiveAttention: sample image features at where ref_3d projects.
        2. Inject those into the query tokens; self-attention + FFN update them.
        3. Heads predict a per-view 2D offset + confidence.
        4. Triangulate the offset-corrected 2D (confidence-weighted) -> new ref_3d.

    Returns the updated query features, the new 3D estimate, and the refined 2D
    (kept for the Phase-6 2D loss that supervises what triangulation consumes).
    """

    def __init__(self, d_model: int = 256, n_heads: int = 8, ffn_dim: int = 1024) -> None:
        super().__init__()
        self.proj_attn = ProjectiveAttention(d_model)
        self.query_update = QueryUpdate(d_model, n_heads, ffn_dim)
        self.head = OffsetConfidenceHead(d_model)

    def forward(self, query_feat, ref_3d, feat_maps, proj, image_hw):
        # 1. gather image evidence at the current 3D guess
        feat, sampled, uv = self.proj_attn(ref_3d, feat_maps, proj, image_hw)
        # 2. inject into query tokens, then let queries talk to each other
        query_feat = self.query_update(query_feat + feat)
        # 3. per-view 2D nudge + confidence. Head sees the per-view sampled
        #    feature PLUS the updated (self-attended) query feature broadcast over
        #    views -> the learned query tokens actually drive the prediction.
        head_in = sampled + query_feat.unsqueeze(1)         # (B,N,Q,C)
        offset, conf = self.head(head_in)                   # (B,N,Q,2), (B,N,Q)
        refined_uv = uv + offset                            # (B,N,Q,2)
        # 4. confidence-weighted triangulation -> refined 3D
        new_ref_3d = self._triangulate(refined_uv, proj, conf)
        return query_feat, new_ref_3d, refined_uv

    @staticmethod
    def _triangulate(refined_uv, proj, conf):
        B, N, Q, _ = refined_uv.shape
        # fold (B,Q) into the batch of points M = B*Q; each point has N views.
        pts = refined_uv.permute(0, 2, 1, 3).reshape(B * Q, N, 2)      # (M,N,2)
        Ps = proj[:, None].expand(B, Q, N, 3, 4).reshape(B * Q, N, 3, 4)
        w = conf.permute(0, 2, 1).reshape(B * Q, N)                    # (M,N)
        X = triangulate_dlt_batch(pts, Ps, w)                          # (M,3)
        return X.reshape(B, Q, 3)


class MultiViewDecoder(nn.Module):
    """Phase 5 - stack of L DecoderLayers refining 3D landmarks from a template.

    Query tokens are learned embeddings; the 3D reference starts at the mean-face
    template placed at the query-box center (SPACE_CENTER), and each layer refines
    it. Returns per-layer 3D and 2D predictions (deep supervision in Phase 6).
    """

    def __init__(self, mean_face, space_center, num_layers: int = 4,
                 d_model: int = 256, n_heads: int = 8, ffn_dim: int = 1024,
                 num_queries: int = 68) -> None:
        super().__init__()
        self.query_embed = nn.Parameter(torch.empty(num_queries, d_model))
        nn.init.normal_(self.query_embed, std=0.02)

        # ref_3d init: centered template shape + world box center -> (Q,3). Fixed
        # (buffer), same start for every sample; the decoder learns the pose.
        mf = torch.as_tensor(np.asarray(mean_face), dtype=torch.float32)
        sc = torch.as_tensor(np.asarray(space_center), dtype=torch.float32)
        self.register_buffer("ref0", mf + sc)

        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, ffn_dim) for _ in range(num_layers)])

    @classmethod
    def from_assets(cls, assets_dir, **kw):
        assets_dir = Path(assets_dir)
        mean_face = np.load(assets_dir / "mean_face_68.npy")
        space = json.loads((assets_dir / "query_space.json").read_text())
        return cls(mean_face, space["SPACE_CENTER"], **kw)

    def forward(self, feat_maps, proj, image_hw):
        B = feat_maps.shape[0]
        query_feat = self.query_embed.unsqueeze(0).expand(B, -1, -1)   # (B,Q,d)
        ref_3d = self.ref0.unsqueeze(0).expand(B, -1, -1)              # (B,Q,3)

        preds_3d, preds_2d = [], []
        for layer in self.layers:
            query_feat, ref_3d, refined_uv = layer(
                query_feat, ref_3d, feat_maps, proj, image_hw)
            preds_3d.append(ref_3d)                                    # (B,Q,3)
            preds_2d.append(refined_uv)                                # (B,N,Q,2)
        return preds_3d, preds_2d
