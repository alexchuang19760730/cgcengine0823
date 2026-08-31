"""
Token-Level Router for FusionRoute + MoT

Adapted from xiongny/FusionRoute for Qwen3.6/Ornith-1.5 family.
Original: Router(LlamaForCausalLM) with weight_proj = Linear(hidden, n)

Architecture:
  1. Load base Qwen3 model (frozen)
  2. Add weight_proj: Linear(hidden_size → n_experts)
  3. Forward: hidden_states → scores [batch, seq_len, n_experts]
  4. Train with CDPO: learn which expert to select per token

Key difference from original:
  - Qwen3 uses Qwen3MoeForCausalLM (MoE) or Qwen3ForCausalLM (dense)
  - KV head structure: 2 KV heads, 256 head_dim (different from Llama)
  - Supports MTP: draft tokens can be predicted alongside routing scores
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Qwen3MoeForCausalLM,
    Qwen3ForCausalLM,
)


class Qwen3Router(nn.Module):
    """
    Token-level router for Qwen3.6 family models.
    
    Based on FusionRoute's Router but adapted for Qwen3 architecture.
    Adds a linear projection from hidden states to expert selection scores.
    
    Usage:
        router = Qwen3Router.from_pretrained(
            "Qwen/Qwen3.6-35B-A3B",
            n_experts=2,  # Qwen3.6 + Ornith-1.5
            freeze_base=True,
        )
        outputs, scores = router(input_ids)
        # scores: [batch, seq_len, 2] - token-level routing
    """
    
    def __init__(self, base_model, n_experts: int = 2, freeze_base: bool = True):
        super().__init__()
        self.base_model = base_model
        self.n_experts = n_experts
        
        # Get hidden size from model config
        hidden_size = base_model.config.hidden_size
        
        # Routing projection: hidden → expert scores
        self.weight_proj = nn.Linear(hidden_size, n_experts)
        
        # Initialize with small weights (near-uniform routing initially)
        nn.init.normal_(self.weight_proj.weight, std=0.01)
        nn.init.zeros_(self.weight_proj.bias)
        
        # Optionally freeze base model (only train router head)
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
            # Unfreeze only weight_proj
            for param in self.weight_proj.parameters():
                param.requires_grad = True
    
    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        n_experts: int = 2,
        freeze_base: bool = True,
        torch_dtype=torch.float16,
        device_map: str = "auto",
        **kwargs,
    ):
        """
        Load a Qwen3 model and wrap it as a router.
        
        Args:
            model_name: HF model repo (e.g., "Qwen/Qwen3.6-35B-A3B")
            n_experts: number of expert models to route between
            freeze_base: freeze base model weights (only train router head)
        """
        print(f"Loading base model: {model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
            **kwargs,
        )
        
        model_type = getattr(base_model.config, "model_type", "")
        print(f"  model_type: {model_type}")
        print(f"  hidden_size: {base_model.config.hidden_size}")
        print(f"  num_layers: {getattr(base_model.config, 'num_hidden_layers', 'N/A')}")
        
        router = cls(base_model, n_experts=n_experts, freeze_base=freeze_base)
        
        # Print stats
        total_params = sum(p.numel() for p in router.parameters())
        trainable_params = sum(p.numel() for p in router.parameters() if p.requires_grad)
        print(f"  Total params: {total_params:,}")
        print(f"  Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
        
        return router
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: get model outputs + routing scores.
        
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
        
        Returns:
            outputs: model outputs (logits, hidden_states, etc.)
            scores: [batch, seq_len, n_experts] routing scores
        """
        # Get hidden states from base model
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs,
        )
        
        # Last hidden state → routing scores
        last_hidden = outputs.hidden_states[-1]  # [batch, seq_len, hidden_size]
        scores = self.weight_proj(last_hidden)     # [batch, seq_len, n_experts]
        
        return outputs, scores
    
    def get_expert_weights(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get soft expert weights (softmax of routing scores).
        
        Returns:
            weights: [batch, seq_len, n_experts] - probability distribution
        """
        with torch.no_grad():
            _, scores = self.forward(input_ids, attention_mask)
            weights = torch.softmax(scores, dim=-1)
        return weights
    
    def get_expert_selection(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get hard expert selection (argmax of routing scores).
        
        Returns:
            selections: [batch, seq_len] - expert index per token
        """
        with torch.no_grad():
            _, scores = self.forward(input_ids, attention_mask)
            selections = torch.argmax(scores, dim=-1)
        return selections
    
    def save_router(self, path: str):
        """Save only the router weights (not the base model)."""
        import json
        from pathlib import Path
        
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save router weights
        torch.save(
            self.weight_proj.state_dict(),
            save_dir / "weight_proj.pt"
        )
        
        # Save config
        config = {
            "n_experts": self.n_experts,
            "hidden_size": self.base_model.config.hidden_size,
            "base_model_name": getattr(
                self.base_model.config, "_name_or_path", "unknown"
            ),
        }
        with open(save_dir / "router_config.json", "w") as f:
            json.dump(config, f)
        
        print(f"Saved router to {save_dir}")
    
    @classmethod
    def load_router(
        cls,
        path: str,
        base_model_name: str = None,
        freeze_base: bool = True,
        **kwargs,
    ):
        """Load a saved router."""
        import json
        from pathlib import Path
        
        save_dir = Path(path)
        
        with open(save_dir / "router_config.json") as f:
            config = json.load(f)
        
        if base_model_name is None:
            base_model_name = config["base_model_name"]
        
        router = cls.from_pretrained(
            base_model_name,
            n_experts=config["n_experts"],
            freeze_base=freeze_base,
            **kwargs,
        )
        
        # Load router weights
        router.weight_proj.load_state_dict(
            torch.load(save_dir / "weight_proj.pt", weights_only=True)
        )
        
        print(f"Loaded router from {save_dir}")
        return router


def create_dual_expert_router(
    model_a: str = "Qwen/Qwen3.6-35B-A3B",
    model_b: str = "ornith-ai/Ornith-1.5-35B-A3B",
    freeze_base: bool = True,
):
    """
    Create a 2-expert router: Qwen3.6 (general) + Ornith-1.5 (code/agent).
    
    Returns:
        router: Qwen3Router with n_experts=2
        expert_names: ["Qwen3.6-35B", "Ornith-1.5-35B"]
    """
    # Use Qwen3.6 as the router base (smaller, faster)
    router = Qwen3Router.from_pretrained(
        model_a,
        n_experts=2,
        freeze_base=freeze_base,
    )
    
    expert_names = [
        model_a.split("/")[-1],  # "Qwen3.6-35B-A3B"
        model_b.split("/")[-1],  # "Ornith-1.5-35B-A3B"
    ]
    
    print(f"\nDual-expert router created:")
    print(f"  Expert 0: {expert_names[0]} (general)")
    print(f"  Expert 1: {expert_names[1]} (code/agent)")
    
    return router, expert_names
