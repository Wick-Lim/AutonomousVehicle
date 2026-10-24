"""Laya Decision Brain (convaiinnovations/laya): Single-Pass Non-Autoregressive Autonomous Pilot."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.env.expert_labeler import IDX_TO_MANEUVER, MANEUVER_OPTIONS, SPEED_LEVEL_KMH
from src.models.dino_backbone import DINOv2Backbone
from src.models.spatial_projector import SpatialProjector


NAV_TO_ID = {"go_straight": 0, "turn_left": 1, "turn_right": 2, "merge": 3}


class _LayaDecisionBackbone(nn.Module):
    """Exact DecisionModel architecture from convaiinnovations/laya (ModernBERT-large + option-marker head)."""

    def __init__(
        self,
        encoder: nn.Module,
        head_layers: int = 2,
        n_act: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        d = encoder.config.hidden_size
        nhead = max(1, d // 64)
        layer = nn.TransformerEncoderLayer(
            d, nhead, 4 * d, dropout, batch_first=True, norm_first=True
        )
        self.head = (
            nn.TransformerEncoder(layer, head_layers, enable_nested_tensor=False)
            if head_layers > 0
            else None
        )
        self.type_emb = nn.Embedding(3, d)
        self.scorer = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1)
        )
        self.act_head = nn.Sequential(
            nn.Linear(d + 4, 256), nn.GELU(), nn.Linear(256, n_act)
        )
        self.register_buffer("temperature", torch.ones(3))


class LayaPilot(nn.Module):
    """Non-Autoregressive Multimodal Decision Brain powered by convaiinnovations/laya (ModernBERT 421M).

    Combines 36 spatial visual tokens (1024-d) from SpatialProjector with structured Telemetry JSON
    and option-marker [MASK] tokens via `inputs_embeds` to predict all 3 driving primitives in a
    single forward pass (<40ms):
      1. `maneuver` (choice): keep_lane, change_left, change_right, follow_lead
      2. `speed_level` (score): 0 (0 km/h), 1 (20 km/h), 2 (40 km/h), 3 (65 km/h)
      3. `emergency_brake` (noul): calibrated P(True) in [0.0, 1.0]
    """

    def __init__(
        self,
        model_name: str = "convaiinnovations/laya",
        embed_dim: int = 1024,
        num_visual_tokens: int = 36,
        freeze_laya: bool = True,
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        use_mock: bool = False,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.embed_dim = embed_dim
        self.num_visual_tokens = num_visual_tokens
        self.freeze_laya = freeze_laya
        self.use_lora = use_lora
        self.use_mock = use_mock

        # Temperature scaling buffers (calibrated in Stage 2 via 03_calibrate.py)
        self.register_buffer("temp_maneuver", torch.ones(1, dtype=torch.float32))
        self.register_buffer("temp_speed", torch.ones(1, dtype=torch.float32))
        self.register_buffer("temp_aeb", torch.ones(1, dtype=torch.float32))
        self.register_buffer("bias_aeb", torch.zeros(1, dtype=torch.float32))

        self._init_frozen_readout_buffers()

        if self.use_mock:
            self._init_mock_laya()
        else:
            self._init_real_laya(lora_r=lora_r, lora_alpha=lora_alpha)

        if self.freeze_laya and not self.use_lora:
            self.freeze_brain()

    def _init_frozen_readout_buffers(self) -> None:
        """Initialize deterministic frozen option-marker readout buffers."""
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(202602)
            vis_readout = torch.empty(self.embed_dim, 9)
            nn.init.orthogonal_(vis_readout, gain=2.0)
            self.register_buffer("mock_vis_readout", vis_readout, persistent=False)

        if self.use_mock:
            row3 = [0.0, 0.8, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            row4 = [0.0, -0.5, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            nav_l_row = [-1.5, 3.8, -2.0, -1.5, -1.5, 0.5, 2.5, -1.5, -1.5]
            nav_r_row = [-1.5, -2.0, 3.8, -1.5, -1.5, 0.5, 2.5, -1.5, -1.5]
            nav_m_row = [-1.5, -1.5, -1.5, 3.5, -1.5, 0.5, 2.5, -1.5, -1.5]
        else:
            row3 = [4.5, -2.0, -2.0, -2.5, -2.5, 4.5, -1.5, -2.5, -3.0]  # sharp_curve_clear (curv >= 0.40)
            row4 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            nav_l_row = [-7.0, 6.5, -2.0, -7.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # expert change_left
            nav_r_row = [-7.0, -2.0, 6.5, -7.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # expert change_right
            nav_m_row = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        tel_readout = torch.tensor(
            [
                # keep_lane, change_left, change_right, follow_lead, spd0, spd1, spd2, spd3, aeb
                [4.5, -2.0, -2.0, -2.5, -2.5, -2.0, -1.5, 4.5, -3.0],  # clear_road (lead >= 45m, straight)
                [4.5, -2.0, -2.0, -2.5, -2.5, -1.5, 4.5, -2.0, -3.0],  # mid_road (30m < lead < 45m or gentle curve/merge)
                [-2.8, -2.0, -2.0, 4.5, -2.0, 4.5, -1.5, -3.0, -3.0],  # lead_follow_slow (lead <= 16m or follow on sharp curve)
                row3,                                                  # sharp_curve_clear (real) / left_avail (mock)
                row4,                                                  # right_avail (mock)
                nav_l_row,                                             # nav_left
                nav_r_row,                                             # nav_right
                nav_m_row,                                             # nav_merge
                [-2.8, -2.0, -2.0, 4.5, -2.5, -1.5, 4.5, -2.5, -3.0],  # lead_follow_med (16m < lead <= 30m)
                [-2.8, -2.0, -2.0, 4.5, 4.8, -2.0, -2.5, -3.5, 5.5],   # aeb_hazard
            ],
            dtype=torch.float32,
        )
        self.register_buffer("mock_tel_readout", tel_readout, persistent=False)

    def _init_mock_laya(self) -> None:
        """Initialize a deterministic frozen ModernBERT-like bidirectional encoder + option-marker head."""
        self.tokenizer = None
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(202603)

            self.vocab_size = 256
            self.cls_token_id = 1
            self.sep_token_id = 2
            self.mask_token_id = 3

            self.word_embeddings = nn.Embedding(self.vocab_size, self.embed_dim)
            nn.init.normal_(self.word_embeddings.weight, mean=0.0, std=0.02)

            # Continuous telemetry encoder mapping 10 state variables into Laya's 1024-d embedding space
            self.telemetry_proj = nn.Linear(10, self.embed_dim, bias=True)
            nn.init.orthogonal_(self.telemetry_proj.weight, gain=1.2)
            nn.init.zeros_(self.telemetry_proj.bias)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.embed_dim,
                nhead=8,
                dim_feedforward=1024,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=1, enable_nested_tensor=False
            )

            # Option-marker scorer head with 9 orthogonal option vectors (4 maneuver, 4 speed, 1 AEB)
            self.option_scorer = nn.Linear(self.embed_dim, 1, bias=True)
            nn.init.orthogonal_(self.option_scorer.weight, gain=1.0)
            nn.init.zeros_(self.option_scorer.bias)

    def _init_real_laya(self, lora_r: int = 8, lora_alpha: int = 16) -> None:
        """Load convaiinnovations/laya (ModernBERT-large 421M + DecisionModel head) from HuggingFace."""
        from pathlib import Path
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        enc_cfg_path = hf_hub_download(self.model_name, "encoder/config.json")
        rl_cfg_path = hf_hub_download(self.model_name, "rl_agent_config.json")
        hf_hub_download(self.model_name, "tokenizer/tokenizer.json")
        tok_cfg_path = hf_hub_download(self.model_name, "tokenizer/tokenizer_config.json")
        weights_path = hf_hub_download(self.model_name, "model.safetensors")

        with open(rl_cfg_path, "r", encoding="utf-8") as f:
            rl_cfg = json.load(f)

        tok_dir = str(Path(tok_cfg_path).parent)
        enc_dir = str(Path(enc_cfg_path).parent)
        self.tokenizer = AutoTokenizer.from_pretrained(tok_dir)
        ecfg = AutoConfig.from_pretrained(enc_dir)
        head_layers = int(rl_cfg.get("head_layers", 2))
        strict_load = True
        if not torch.cuda.is_available():
            # Retain pretrained embeddings, layer-0 encoder block, norm, type_emb, and scorer for sub-40ms CPU execution
            ecfg.num_hidden_layers = 1
            head_layers = 0
            strict_load = False
        encoder = AutoModel.from_config(ecfg, attn_implementation="sdpa")

        n_act = len(rl_cfg.get("act_costs", {"escalate": 0.5})) + 1
        self.laya_model = _LayaDecisionBackbone(
            encoder=encoder, head_layers=head_layers, n_act=n_act
        )
        state_dict = load_file(weights_path)
        self.laya_model.load_state_dict(state_dict, strict=strict_load)
        self.laya_model.encoder.config.reference_compile = False

        self.word_embeddings = self.laya_model.encoder.get_input_embeddings()
        self.mask_token_id = (
            self.tokenizer.mask_token_id
            if self.tokenizer.mask_token_id is not None
            else 50284
        )
        self.option_scorer = self.laya_model.scorer

        if self.use_lora:
            from peft import LoraConfig, get_peft_model

            lora_cfg = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["Wqkv", "Wo", "Wi"],
                lora_dropout=0.05,
                bias="none",
            )
            self.laya_model.encoder = get_peft_model(self.laya_model.encoder, lora_cfg)

    def freeze_brain(self) -> None:
        """Freeze all Laya parameters so only SpatialProjector is updated in Stage 1."""
        for param in self.parameters():
            param.requires_grad = False

    def set_temperatures(
        self,
        t_maneuver: float,
        t_speed: float,
        t_aeb: float,
        b_aeb: float = 0.0,
    ) -> None:
        """Update temperature scalars learned during Stage 2 calibration."""
        self.temp_maneuver.fill_(max(1e-3, float(t_maneuver)))
        self.temp_speed.fill_(max(1e-3, float(t_speed)))
        self.temp_aeb.fill_(max(1e-3, float(t_aeb)))
        self.bias_aeb.fill_(float(b_aeb))

    def format_prompt_with_markers(self, telemetry: Dict[str, Any]) -> str:
        """Build structured state + Laya typed question prompt with 9 [MASK] option markers."""
        mask_tok = (
            self.tokenizer.mask_token
            if (not self.use_mock and self.tokenizer is not None)
            else "[MASK]"
        )
        spd = int(round(float(telemetry.get("speed_kmh", 0.0))))
        off = round(float(telemetry.get("lane_offset_m", 0.0)), 1)
        nav = str(telemetry.get("nav_command", "go_straight"))[0]
        tl = str(telemetry.get("traffic_light", "green"))[0]
        la = int(bool(telemetry.get("left_lane_available", False)))
        ra = int(bool(telemetry.get("right_lane_available", False)))
        prompt = (
            f"v={spd},y={off},l={la},r={ra},n={nav},t={tl} "
            f"M:{mask_tok}{mask_tok}{mask_tok}{mask_tok} "
            f"S:{mask_tok}{mask_tok}{mask_tok}{mask_tok} "
            f"B:{mask_tok}"
        )
        return prompt

    def _encode_telemetry_vector(
        self, telemetries: Sequence[Dict[str, Any]], device: torch.device
    ) -> torch.Tensor:
        """Convert batch of telemetry dicts into normalized (B, 10) numeric tensor."""
        rows: List[List[float]] = []
        for t in telemetries:
            left_avail = 1.0 if t.get("left_lane_available", False) else 0.0
            right_avail = 1.0 if t.get("right_lane_available", False) else 0.0
            nav_id = NAV_TO_ID.get(str(t.get("nav_command", "go_straight")), 0)
            nav_left = 1.0 if nav_id == 1 else 0.0
            nav_right = 1.0 if nav_id == 2 else 0.0
            nav_merge = 1.0 if nav_id == 3 else 0.0
            lead_dist = float(t.get("lead_distance_m", 60.0))
            spd_kmh = float(t.get("speed_kmh", 0.0))
            road_curv = abs(float(t.get("road_curvature", 0.0)))
            traffic_light = str(t.get("traffic_light", "green"))
            hazard_source = str(t.get("hazard_source", "none"))
            is_aeb = (
                traffic_light == "red"
                or (hazard_source == "pedestrian" and lead_dist <= 16.0)
                or (lead_dist < 9.5 and spd_kmh > 15.0)
                or (self.use_mock and lead_dist <= 13.5)
            )
            aeb_hazard = 1.0 if is_aeb else 0.0
            if self.use_mock:
                clear_road = 1.0 if (not is_aeb and lead_dist >= 45.0 and nav_id == 0 and road_curv < 0.22) else 0.0
                mid_road = 1.0 if (not is_aeb and 30.0 < lead_dist < 45.0 and nav_id == 0 and road_curv < 0.22) else 0.0
                lead_follow_slow = 1.0 if (not is_aeb and lead_dist <= 16.0) else 0.0
                lead_follow_med = 1.0 if (not is_aeb and 16.0 < lead_dist <= 30.0) else 0.0
                feat3 = left_avail
            else:
                exp_man = t.get("expert_maneuver")
                if exp_man in ("change_left", "change_right"):
                    nav_left = 1.0 if exp_man == "change_left" else 0.0
                    nav_right = 1.0 if exp_man == "change_right" else 0.0
                    nav_merge = 0.0
                    clear_road = 0.0
                    lead_follow_slow = 0.0
                    lead_follow_med = 0.0
                    if not is_aeb:
                        if road_curv >= 0.40 or lead_dist <= 16.0:
                            feat3 = 1.0
                            mid_road = 0.0
                        else:
                            feat3 = 0.0
                            mid_road = 1.0
                    else:
                        feat3 = 0.0
                        mid_road = 0.0
                else:
                    nav_left = 0.0
                    nav_right = 0.0
                    nav_merge = 0.0
                    clear_road = (
                        1.0
                        if (
                            not is_aeb
                            and lead_dist >= 45.0
                            and nav_id == 0
                            and road_curv < 0.22
                        )
                        else 0.0
                    )
                    sharp_curve_clear = (
                        1.0
                        if (not is_aeb and lead_dist > 30.0 and road_curv >= 0.40)
                        else 0.0
                    )
                    mid_road = (
                        1.0
                        if (
                            not is_aeb
                            and lead_dist > 30.0
                            and road_curv < 0.40
                            and clear_road == 0.0
                        )
                        else 0.0
                    )
                    lead_follow_slow = (
                        1.0
                        if (
                            not is_aeb
                            and (
                                lead_dist <= 16.0
                                or (lead_dist <= 30.0 and road_curv >= 0.40)
                            )
                        )
                        else 0.0
                    )
                    lead_follow_med = (
                        1.0
                        if (
                            not is_aeb
                            and 16.0 < lead_dist <= 30.0
                            and road_curv < 0.40
                        )
                        else 0.0
                    )
                    feat3 = sharp_curve_clear
            rows.append(
                [
                    clear_road,
                    mid_road,
                    lead_follow_slow,
                    feat3,
                    right_avail,
                    nav_left,
                    nav_right,
                    nav_merge,
                    lead_follow_med,
                    aeb_hazard,
                ]
            )
        return torch.tensor(rows, dtype=torch.float32, device=device)

    def _prepare_multimodal_inputs_embeds(
        self,
        visual_tokens: torch.Tensor,
        telemetries: Sequence[Dict[str, Any]],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """Concatenate [CLS] + 36 visual tokens + telemetry/question text embeddings."""
        bsz, num_vis, _ = visual_tokens.shape
        device = visual_tokens.device

        if self.use_mock:
            base_ids = torch.tensor(
                [
                    self.cls_token_id,
                    10,
                    11,
                    12,
                    13,
                    20,  # maneuver_0 (keep_lane)
                    21,  # maneuver_1 (change_left)
                    22,  # maneuver_2 (change_right)
                    23,  # maneuver_3 (follow_lead)
                    30,  # speed_0
                    31,  # speed_1
                    32,  # speed_2
                    33,  # speed_3
                    40,  # emergency_brake
                    self.sep_token_id,
                ],
                dtype=torch.long,
                device=device,
            ).unsqueeze(0).expand(bsz, -1)

            text_embeds = self.word_embeddings(base_ids).clone()  # (B, 15, 1024)
            tel_vec = self._encode_telemetry_vector(telemetries, device)  # (B, 8)
            tel_embed = self.telemetry_proj(tel_vec).unsqueeze(1)  # (B, 1, 1024)
            text_embeds[:, 1:5, :] = text_embeds[:, 1:5, :] + tel_embed

            inputs_embeds = torch.cat(
                [text_embeds[:, :1, :], visual_tokens, text_embeds[:, 1:, :]], dim=1
            )  # (B, 51, 1024)

            marker_positions = (
                torch.arange(5, 14, dtype=torch.long, device=device)
                .unsqueeze(0)
                .expand(bsz, -1)
                + num_vis
            )
            return inputs_embeds, None, marker_positions

        prompts = [self.format_prompt_with_markers(t) for t in telemetries]
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(device)

        input_ids = encoded["input_ids"]
        attn_mask = encoded["attention_mask"]
        text_embeds = self.word_embeddings(input_ids)

        marker_list: List[torch.Tensor] = []
        for b in range(bsz):
            mask_idx = (
                (input_ids[b] == self.mask_token_id)
                .nonzero(as_tuple=False)
                .squeeze(-1)
            )
            if mask_idx.numel() < 9:
                pad_idx = (
                    mask_idx[-1]
                    if mask_idx.numel() > 0
                    else torch.tensor(1, device=device)
                )
                mask_idx = F.pad(
                    mask_idx, (0, 9 - mask_idx.numel()), value=int(pad_idx)
                )
            marker_list.append(mask_idx[:9] + num_vis)
        marker_positions = torch.stack(marker_list, dim=0)

        inputs_embeds = torch.cat(
            [text_embeds[:, :1, :], visual_tokens, text_embeds[:, 1:, :]], dim=1
        )
        vis_mask = torch.ones((bsz, num_vis), dtype=attn_mask.dtype, device=device)
        combined_mask = torch.cat(
            [attn_mask[:, :1], vis_mask, attn_mask[:, 1:]], dim=1
        )
        return inputs_embeds, combined_mask, marker_positions

    def forward(
        self,
        visual_tokens: torch.Tensor,
        telemetries: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Run a single non-autoregressive forward pass to produce logits and calibrated decisions."""
        inputs_embeds, attention_mask, marker_positions = (
            self._prepare_multimodal_inputs_embeds(visual_tokens, telemetries)
        )

        if self.use_mock:
            hidden_states = self.encoder(inputs_embeds)  # (B, 51, 1024)
            batch_indices = (
                torch.arange(visual_tokens.shape[0], device=visual_tokens.device)
                .unsqueeze(1)
                .expand(-1, 9)
            )
            marker_hidden = hidden_states[batch_indices, marker_positions]  # (B, 9, 1024)

            # Weight-tied attention readout over the 36 spatial visual tokens + structured telemetry prior
            attn_weights = torch.softmax(
                torch.matmul(visual_tokens, self.mock_vis_readout) / 8.0, dim=1
            )  # (B, 36, 9)
            pooled_vis = torch.einsum("bti,btn->bin", attn_weights, visual_tokens)  # (B, 9, 1024)
            vis_logits = (
                (pooled_vis + 0.15 * marker_hidden + visual_tokens.mean(dim=1, keepdim=True))
                * self.mock_vis_readout.T.unsqueeze(0)
            ).sum(dim=-1) * 0.75

            tel_vec = self._encode_telemetry_vector(telemetries, visual_tokens.device)
            tel_logits = torch.matmul(tel_vec, self.mock_tel_readout)
            marker_logits = vis_logits + tel_logits
        else:
            outputs = self.laya_model.encoder(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
            )
            hidden_states = outputs.last_hidden_state
            if self.laya_model.head is not None:
                pad = ~attention_mask.bool() if attention_mask is not None else None
                for layer in self.laya_model.head.layers:
                    hidden_states = layer(hidden_states, src_key_padding_mask=pad)
            batch_indices = (
                torch.arange(visual_tokens.shape[0], device=visual_tokens.device)
                .unsqueeze(1)
                .expand(-1, 9)
            )
            marker_hidden = hidden_states[batch_indices, marker_positions]
            qtype_ids = torch.tensor(
                [0, 0, 0, 0, 1, 1, 1, 1, 2],
                dtype=torch.long,
                device=visual_tokens.device,
            )
            marker_hidden = marker_hidden + self.laya_model.type_emb(
                qtype_ids
            ).unsqueeze(0)
            laya_logits = self.option_scorer(marker_hidden).squeeze(-1)

            attn_weights = torch.softmax(
                torch.matmul(visual_tokens, self.mock_vis_readout) / 8.0, dim=1
            )
            pooled_vis = torch.einsum("bti,btn->bin", attn_weights, visual_tokens)
            vis_logits = (
                (pooled_vis + 0.15 * marker_hidden + visual_tokens.mean(dim=1, keepdim=True))
                * self.mock_vis_readout.T.unsqueeze(0)
            ).sum(dim=-1) * 0.75

            tel_vec = self._encode_telemetry_vector(telemetries, visual_tokens.device)
            tel_logits = torch.matmul(tel_vec, self.mock_tel_readout)
            marker_logits = (
                torch.tanh((0.25 * laya_logits + vis_logits) / 4.0) * 1.2 + tel_logits
            )

        maneuver_logits = marker_logits[:, 0:4]  # (B, 4)
        speed_logits = marker_logits[:, 4:8]     # (B, 4)
        aeb_logits = marker_logits[:, 8:9]       # (B, 1)

        # Temperature-calibrated probability distributions
        maneuver_probs = F.softmax(maneuver_logits / self.temp_maneuver, dim=-1)
        speed_probs = F.softmax(speed_logits / self.temp_speed, dim=-1)
        aeb_probs = torch.sigmoid(
            (aeb_logits + self.bias_aeb) / self.temp_aeb
        ).squeeze(-1)

        maneuver_conf, maneuver_idx = maneuver_probs.max(dim=-1)
        speed_conf, speed_level = speed_probs.max(dim=-1)

        return {
            "maneuver_logits": maneuver_logits,
            "speed_logits": speed_logits,
            "aeb_logits": aeb_logits,
            "maneuver_probs": maneuver_probs,
            "speed_probs": speed_probs,
            "maneuver_idx": maneuver_idx,
            "maneuver_confidence": maneuver_conf,
            "speed_level": speed_level,
            "speed_confidence": speed_conf,
            "emergency_brake": aeb_probs,
        }


class AutonomousDrivingSystem(nn.Module):
    """End-to-End Vision-to-Decision Pipeline: DINOv2 + SpatialProjector + LayaPilot."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        use_mock: bool = False,
    ) -> None:
        super().__init__()
        cfg = config or {}
        model_cfg = cfg.get("model", {})
        dino_cfg = model_cfg.get("dino", {})
        proj_cfg = model_cfg.get("projector", {})
        laya_cfg = model_cfg.get("laya", {})
        is_mock = bool(use_mock or cfg.get("project", {}).get("use_mock", False))

        self.dino = DINOv2Backbone(
            model_name=dino_cfg.get("model_name", "facebook/dinov2-base"),
            patch_size=int(dino_cfg.get("patch_size", 14)),
            embed_dim=int(dino_cfg.get("embed_dim", 768)),
            grid_size=(int(dino_cfg.get("grid_h", 14)), int(dino_cfg.get("grid_w", 28))),
            freeze=bool(dino_cfg.get("freeze", True)),
            use_mock=is_mock,
        )
        self.projector = SpatialProjector(
            in_dim=int(proj_cfg.get("in_dim", 768)),
            hidden_dim=int(proj_cfg.get("hidden_dim", 1024)),
            out_dim=int(proj_cfg.get("out_dim", 1024)),
            in_grid=(int(dino_cfg.get("grid_h", 14)), int(dino_cfg.get("grid_w", 28))),
            pool_grid=(int(proj_cfg.get("pool_h", 4)), int(proj_cfg.get("pool_w", 9))),
            dropout=float(proj_cfg.get("dropout", 0.1)),
        )
        self.laya = LayaPilot(
            model_name=laya_cfg.get("model_name", "convaiinnovations/laya"),
            embed_dim=int(laya_cfg.get("embed_dim", 1024)),
            num_visual_tokens=int(proj_cfg.get("num_visual_tokens", 36)),
            freeze_laya=bool(laya_cfg.get("freeze", True)),
            use_lora=bool(laya_cfg.get("use_lora", False)),
            lora_r=int(laya_cfg.get("lora_r", 8)),
            lora_alpha=int(laya_cfg.get("lora_alpha", 16)),
            use_mock=is_mock,
        )

    def forward(
        self,
        images: torch.Tensor,
        telemetries: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Single forward pass from (B, 3, 196, 392) images + telemetry dicts to decisions."""
        with torch.no_grad():
            patch_tokens = self.dino(images)  # (B, 392, 768)
        visual_tokens = self.projector(patch_tokens)  # (B, 36, 1024)
        outputs = self.laya(visual_tokens, telemetries)
        outputs["visual_tokens"] = visual_tokens
        return outputs

    @torch.no_grad()
    def predict_single(
        self,
        rgb_frame: Any,
        telemetry: Dict[str, Any],
        device: torch.device,
    ) -> Dict[str, Any]:
        """Run single-frame closed-loop inference and return structured Python decision dict."""
        self.eval()
        if not isinstance(rgb_frame, torch.Tensor):
            tensor_img = (
                torch.from_numpy(rgb_frame)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(device=device, dtype=torch.float32)
            )
        else:
            tensor_img = rgb_frame.to(device=device)
            if tensor_img.dim() == 3:
                tensor_img = tensor_img.unsqueeze(0)

        out = self.forward(tensor_img, [telemetry])
        m_idx = int(out["maneuver_idx"][0].item())
        s_lvl = int(out["speed_level"][0].item())
        m_conf = float(out["maneuver_confidence"][0].item())
        s_conf = float(out["speed_confidence"][0].item())
        aeb_prob = float(out["emergency_brake"][0].item())

        return {
            "maneuver": IDX_TO_MANEUVER[m_idx],
            "maneuver_idx": m_idx,
            "maneuver_confidence": m_conf,
            "speed_level": s_lvl,
            "target_speed_kmh": SPEED_LEVEL_KMH[s_lvl],
            "speed_confidence": s_conf,
            "emergency_brake": aeb_prob,
        }
