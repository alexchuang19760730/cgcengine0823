#!/usr/bin/env python3
"""
CDPO Training for FusionRoute Router (CPU-only, self-contained)

Trains router weight_proj + complementary logit head using mock experts.
No transformers/LLM dependencies required — only PyTorch + NumPy.

Usage:
  python train_cdpo.py --n_samples 1000 --n_epochs 10
  python train_cdpo.py --n_samples 100 --n_epochs 3 --hidden_size 256 --vocab_size 1000 --seq_len 64
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
from dataclasses import dataclass


# ============================================================================
# Inlined modules (no external deps)
# ============================================================================

class MockExpert(nn.Module):
    """Mock expert: embedding + linear projection. CPU-friendly."""
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.proj = nn.Linear(hidden_size, vocab_size)
        nn.init.normal_(self.embedding.weight, std=0.01)
        nn.init.normal_(self.proj.weight, std=0.01)
        nn.init.zeros_(self.proj.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.proj(self.embedding(input_ids))


class ComplementaryLogit(nn.Module):
    """z_fuse = z_expert + alpha * correction_head(hidden)."""
    def __init__(self, hidden_size: int, vocab_size: int, alpha: float = 0.1):
        super().__init__()
        self.alpha = alpha
        self.correction_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, vocab_size),
        )
        for m in self.correction_head:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, hidden_state: torch.Tensor, expert_logits: torch.Tensor) -> torch.Tensor:
        return expert_logits + self.alpha * self.correction_head(hidden_state)


class SimpleRouter(nn.Module):
    """Lightweight router: embedding + linear → n_experts scores."""
    def __init__(self, vocab_size: int, hidden_size: int, n_experts: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.proj = nn.Linear(hidden_size, n_experts)
        nn.init.normal_(self.proj.weight, std=0.01)
        nn.init.zeros_(self.proj.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.proj(self.embedding(input_ids))


# ============================================================================
# Synthetic preference data
# ============================================================================

@dataclass
class PreferenceSample:
    prompt: str
    chosen: str
    rejected: str
    label: int  # 1 = DPO pair, 0 = SFT


def generate_synthetic_preferences(n_samples: int = 1000, seed: int = 42) -> List[PreferenceSample]:
    rng = np.random.RandomState(seed)
    prompts = [
        "Write a Python function to compute factorial.",
        "Explain the difference between TCP and UDP.",
        "Implement binary search in Python.",
        "What is the time complexity of quicksort?",
        "Write a SQL query to find duplicates.",
        "Explain how garbage collection works.",
        "Implement a linked list in Python.",
        "What is the CAP theorem?",
        "Write a function to merge sorted arrays.",
        "Explain virtual memory.",
        "Implement a thread-safe queue.",
        "What is consistent hashing?",
        "Write a decorator that retries on failure.",
        "Explain the observer pattern.",
        "Implement a bloom filter.",
        "What is event sourcing?",
        "Write a parser for nested parentheses.",
        "Explain copy-on-write semantics.",
        "Implement a priority queue.",
        "What is the saga pattern?",
    ]
    chosen_suffixes = [
        "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)",
        "TCP is connection-oriented with guaranteed delivery; UDP is connectionless.",
        "def bs(a, t):\n lo,hi=0,len(a)-1\n while lo<=hi:\n  m=(lo+hi)//2\n  if a[m]==t: return m\n  elif a[m]<t: lo=m+1\n  else: hi=m-1\n return -1",
        "Quicksort: O(n log n) average, O(n^2) worst case.",
        "SELECT col, COUNT(*) FROM t GROUP BY col HAVING COUNT(*)>1;",
        "GC reclaims unreachable memory via mark-sweep or generational collection.",
        "class Node: ...\nclass LinkedList: ...",
        "CAP: Consistency, Availability, Partition tolerance — pick two.",
        "def merge(a,b):\n r=[]; i=j=0\n while i<len(a) and j<len(b):\n  if a[i]<=b[j]: r.append(a[i]); i+=1\n  else: r.append(b[j]); j+=1\n return r+a[i:]+b[j:]",
        "Virtual memory maps logical addresses to physical RAM + swap.",
        "Use threading.Lock with collections.deque.",
        "Consistent hashing maps keys to a ring, minimizing redistribution.",
        "def retry(n=3):\n def deco(f):\n  def w(*a,**k):\n   for i in range(n):\n    try: return f(*a,**k)\n    except: pass\n  return w\n return deco",
        "Observer: subject notifies observers on state change.",
        "Bloom filter: probabilistic set with false positives, no false negatives.",
        "Event sourcing stores state changes as immutable events.",
        "def parse(s):\n stack=[]\n for c in s:\n  if c=='(': stack.append(c)\n  elif c==')': stack.pop()\n return len(stack)==0",
        "COW shares pages until modification, then copies.",
        "def push(h,i): h.append(i); _up(h,len(h)-1)",
        "Saga = sequence of local txns with compensating actions.",
    ]
    rejected_suffixes = [
        "factorial = lambda n: n * factorial(n-1)",  # missing base case
        "TCP is faster than UDP.",  # wrong
        "def bs(a,t): return a.index(t)",  # O(n)
        "Quicksort is always O(n log n).",  # wrong
        "SELECT * FROM t;",  # wrong
        "GC deletes all objects.",  # wrong
        "class Node: pass",  # incomplete
        "CAP means all three.",  # wrong
        "return sorted(a+b)",  # O(n log n) not O(n)
        "Virtual memory is just RAM.",  # wrong
        "global q = []",  # not thread-safe
        "Just use a dict.",  # misses point
        "def retry(f): return f()",  # no retry
        "Observer = polling.",  # wrong
        "Bloom filter = exact.",  # wrong
        "Event sourcing = CRUD.",  # wrong
        "def parse(s): return True",  # no parsing
        "COW = always copies.",  # opposite
        "def push(h,i): h.append(i)",  # no ordering
        "Saga = 2PC.",  # wrong
    ]
    samples = []
    for i in range(n_samples):
        idx = i % len(prompts)
        samples.append(PreferenceSample(
            prompt=prompts[idx] + (f" (v{i//len(prompts)+1})" if i >= len(prompts) else ""),
            chosen=chosen_suffixes[idx],
            rejected=rejected_suffixes[idx],
            label=1,
        ))
    # SFT samples
    for i in range(n_samples // 4):
        idx = i % len(prompts)
        samples.append(PreferenceSample(
            prompt=prompts[idx],
            chosen=chosen_suffixes[idx],
            rejected=chosen_suffixes[idx],
            label=0,
        ))
    rng.shuffle(samples)
    return samples


# ============================================================================
# CDPO Trainer
# ============================================================================

class CDPOTrainer:
    def __init__(self, router, complementary, expert_a, expert_b,
                 lr=1e-4, beta=0.1, sft_weight=0.5, device="cpu"):
        self.router = router
        self.complementary = complementary
        self.expert_a = expert_a
        self.expert_b = expert_b
        self.beta = beta
        self.sft_weight = sft_weight
        self.device = device

        for p in list(expert_a.parameters()) + list(expert_b.parameters()):
            p.requires_grad = False

        trainable = list(router.parameters()) + list(complementary.parameters())
        self.optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
        self.step_count = 0
        self.loss_history = []

    def get_combined_logits(self, input_ids, hard=False):
        router_scores = self.router(input_ids)  # [batch, seq, 2]
        if hard:
            idx = torch.argmax(router_scores, dim=-1)
            one_hot = F.one_hot(idx, num_classes=2).float()
            s_a, s_b = one_hot[..., 0:1], one_hot[..., 1:2]
        else:
            probs = F.softmax(router_scores, dim=-1)
            s_a, s_b = probs[..., 0:1], probs[..., 1:2]

        with torch.no_grad():
            la = self.expert_a(input_ids)
            lb = self.expert_b(input_ids)
            mv = min(la.shape[-1], lb.shape[-1])
            la, lb = la[..., :mv], lb[..., :mv]

        combined = la * s_a + lb * s_b
        labels = torch.roll(input_ids, shifts=-1, dims=1)
        labels[:, -1] = -100
        pred_a = torch.argmax(la, dim=-1)
        pred_b = torch.argmax(lb, dim=-1)
        labels[pred_a == pred_b] = -100
        return combined, labels

    def _log_probs(self, logits, labels):
        lp = F.log_softmax(logits[:, :-1, :], dim=-1)
        lab = labels[:, :-1].clone()
        lab[lab == -100] = 0
        tok_lp = torch.gather(lp, -1, lab.unsqueeze(-1)).squeeze(-1)
        mask = (labels[:, :-1] != -100).float()
        return (tok_lp * mask).sum(-1)

    def train_step(self, input_ids, label_tensor):
        self.router.train()
        self.complementary.train()
        self.optimizer.zero_grad()

        total_loss = torch.tensor(0.0, device=self.device)
        metrics = {}

        dpo_mask = (label_tensor == 1)
        sft_mask = (label_tensor == 0)

        if dpo_mask.sum() > 0:
            ids = input_ids[dpo_mask]
            cl, lab = self.get_combined_logits(ids, hard=False)
            with torch.no_grad():
                rl, _ = self.get_combined_logits(ids, hard=True)
            c_lp = self._log_probs(cl, lab)
            r_lp = self._log_probs(rl, lab)
            dpo_loss = -F.logsigmoid(self.beta * (c_lp - r_lp)).mean()
            total_loss = total_loss + dpo_loss
            metrics["dpo_loss"] = dpo_loss.item()

        if sft_mask.sum() > 0:
            ids = input_ids[sft_mask]
            cl, lab = self.get_combined_logits(ids, hard=False)
            sft_loss = nn.CrossEntropyLoss(ignore_index=-100)(
                cl[:, :-1].contiguous().view(-1, cl.shape[-1]),
                lab[:, :-1].contiguous().view(-1),
            )
            total_loss = total_loss + self.sft_weight * sft_loss
            metrics["sft_loss"] = sft_loss.item()

        if total_loss.requires_grad:
            total_loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.router.parameters()) + list(self.complementary.parameters()), 1.0)
            self.optimizer.step()

        metrics["total_loss"] = total_loss.item()
        self.step_count += 1
        self.loss_history.append(metrics["total_loss"])
        return metrics

    def save(self, path):
        d = Path(path); d.mkdir(parents=True, exist_ok=True)
        torch.save(self.router.state_dict(), d / "router.pt")
        torch.save(self.complementary.state_dict(), d / "complementary.pt")
        with open(d / "stats.json", "w") as f:
            json.dump({"steps": self.step_count, "loss": self.loss_history[-100:],
                        "beta": self.beta, "sft_weight": self.sft_weight}, f, indent=2)
        print(f"Saved to {d}")


# ============================================================================
# Main
# ============================================================================

def train_cdpo(n_samples=1000, n_epochs=10, batch_size=8, lr=1e-4,
               beta=0.1, sft_weight=0.5, hidden_size=2048, vocab_size=151936,
               seq_len=256, output_dir="fusion_route_training/", device="cpu"):
    print("=" * 60)
    print("FusionRoute CDPO Training (CPU-only)")
    print("=" * 60)
    print(f"  Samples={n_samples} Epochs={n_epochs} Batch={batch_size}")
    print(f"  lr={lr} beta={beta} sft_weight={sft_weight}")
    print(f"  hidden={hidden_size} vocab={vocab_size} seq={seq_len}")
    print()

    samples = generate_synthetic_preferences(n_samples)
    dpo_n = sum(1 for s in samples if s.label == 1)
    sft_n = sum(1 for s in samples if s.label == 0)
    print(f"  Data: {len(samples)} total ({dpo_n} DPO, {sft_n} SFT)")

    expert_a = MockExpert(vocab_size, hidden_size)
    expert_b = MockExpert(vocab_size, hidden_size)
    router = SimpleRouter(vocab_size, hidden_size, 2)
    comp = ComplementaryLogit(hidden_size, vocab_size, 0.1)

    trainer = CDPOTrainer(router, comp, expert_a, expert_b, lr, beta, sft_weight, device)

    total_params = sum(p.numel() for p in router.parameters()) + sum(p.numel() for p in comp.parameters())
    print(f"  Trainable params: {total_params:,}")
    print(f"  Steps/epoch: {len(samples)//batch_size}")
    print()

    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(n_epochs):
        t0 = time.time()
        losses = []
        idx = np.random.permutation(len(samples))
        for bs in range(0, len(idx), batch_size):
            batch = [samples[i] for i in idx[bs:bs+batch_size]]
            ids = []
            labs = []
            for s in batch:
                toks = [ord(c) % vocab_size for c in (s.prompt + " " + s.chosen)[:seq_len]]
                toks += [0] * (seq_len - len(toks))
                ids.append(toks)
                labs.append(s.label)
            input_ids = torch.tensor(ids, dtype=torch.long, device=device)
            label_t = torch.tensor(labs, dtype=torch.long, device=device)
            m = trainer.train_step(input_ids, label_t)
            losses.append(m["total_loss"])

        avg = np.mean(losses)
        dt = time.time() - t0
        print(f"  Epoch {epoch+1}/{n_epochs}: loss={avg:.4f} time={dt:.1f}s step={trainer.step_count}")
        if avg < best_loss:
            best_loss = avg
            trainer.save(str(out / "best"))
        if (epoch + 1) % 5 == 0:
            trainer.save(str(out / f"ep{epoch+1}"))

    trainer.save(str(out / "final"))
    print(f"\nDone. Best loss={best_loss:.4f} Output={output_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="FusionRoute CDPO (CPU-only)")
    p.add_argument("--n_samples", type=int, default=1000)
    p.add_argument("--n_epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--sft_weight", type=float, default=0.5)
    p.add_argument("--hidden_size", type=int, default=2048)
    p.add_argument("--vocab_size", type=int, default=151936)
    p.add_argument("--seq_len", type=int, default=256)
    p.add_argument("--output_dir", default="fusion_route_training/")
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    train_cdpo(**vars(a))
