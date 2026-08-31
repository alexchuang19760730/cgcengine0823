"""
Complementary Logit Correction for FusionRoute

Core formula: z_fuse = z_expert + c
  z_expert: logits from the selected expert model
  c: complementary correction vector (from router)
  z_fuse: corrected logits

This fixes per-token errors made by individual experts.
The correction is small but effective - trained via CDPO.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class ComplementaryLogit(nn.Module):
    """
    Complementary logit correction head.
    
    Takes the router's hidden state and produces a correction vector
    that is added to the expert's logits.
    
    Architecture:
        hidden_state [batch, seq, hidden] 
            → Linear(hidden, hidden) → ReLU 
            → Linear(hidden, vocab_size) 
            → correction c [batch, seq, vocab_size]
        
        z_fuse = z_expert + α * c
    """
    
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        alpha: float = 0.1,  # correction strength
    ):
        super().__init__()
        self.alpha = alpha
        
        self.correction_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, vocab_size),
        )
        
        # Initialize small (corrections should be subtle)
        for layer in self.correction_head:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.01)
                nn.init.zeros_(layer.bias)
    
    def forward(
        self,
        hidden_state: torch.Tensor,  # [batch, seq, hidden]
        expert_logits: torch.Tensor,  # [batch, seq, vocab]
    ) -> torch.Tensor:
        """
        Apply complementary logit correction.
        
        Returns:
            corrected_logits: [batch, seq, vocab]
        """
        correction = self.correction_head(hidden_state)  # [batch, seq, vocab]
        corrected_logits = expert_logits + self.alpha * correction
        return corrected_logits


class FusionRouteDecoder:
    """
    Complete FusionRoute decode step: route → expert → correct.
    
    Usage:
        decoder = FusionRouteDecoder(router, experts, complementary)
        next_token = decoder.step(input_ids, kv_caches)
    """
    
    def __init__(self, router, experts, complementary, kv_translator=None):
        """
        Args:
            router: Qwen3Router
            experts: list of model instances
            complementary: ComplementaryLogit
            kv_translator: RidgeKVMapper (optional, for KV Translation)
        """
        self.router = router
        self.experts = experts
        self.complementary = complementary
        self.kv_translator = kv_translator
        self.current_expert_idx = None
    
    @torch.no_grad()
    def step(
        self,
        input_ids: torch.Tensor,
        kv_caches: list = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Single decode step: route → select expert → decode → correct.
        
        Returns:
            next_token: [batch, 1]
            info: dict with routing decisions and stats
        """
        # 1. Get routing scores from router
        _, router_scores = self.router(input_ids)  # [batch, seq, n_experts]
        
        # 2. Select expert (last token's scores)
        scores_last = router_scores[:, -1, :]  # [batch, n_experts]
        expert_idx = torch.argmax(scores_last, dim=-1)  # [batch]
        
        # 3. Check if expert changed → KV Translation needed
        expert_changed = False
        if self.current_expert_idx is not None:
            expert_changed = not torch.all(expert_idx == self.current_expert_idx)
        
        # 4. KV Translation if expert changed
        if expert_changed and self.kv_translator is not None and kv_caches is not None:
            # TODO: implement KV cache translation between experts
            # For now, log the transition
            pass
        
        self.current_expert_idx = expert_idx
        
        # 5. Run selected expert
        expert = self.experts[expert_idx[0].item()]
        expert_outputs = expert(input_ids, past_key_values=kv_caches, use_cache=True)
        expert_logits = expert_outputs.logits[:, -1:, :]  # [batch, 1, vocab]
        
        # 6. Get router hidden state for correction
        with torch.no_grad():
            router_out, _ = self.router(input_ids)
            router_hidden = router_out.hidden_states[-1][:, -1:, :]  # [batch, 1, hidden]
        
        # 7. Apply complementary logit correction
        corrected_logits = self.complementary(router_hidden, expert_logits)
        
        # 8. Sample next token
        probs = torch.softmax(corrected_logits, dim=-1)
        next_token = torch.multinomial(probs[:, 0, :], num_samples=1)  # [batch, 1]
        
        info = {
            "expert_idx": expert_idx[0].item(),
            "expert_changed": expert_changed.item() if isinstance(expert_changed, torch.Tensor) else expert_changed,
            "scores": scores_last[0].cpu().tolist(),
        }
        
        return next_token, info
