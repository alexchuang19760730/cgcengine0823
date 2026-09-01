#!/usr/bin/env python3
"""
Extract coding prompts from Freebuff conversation history.
Generates CDPO-ready training data for FusionRoute+MoT.

Usage:
  python extract_prompts.py --db <path-to-desktop-v2.db> --output prompts.jsonl
  
  # With quality scoring
  python extract_prompts.py --db <path> --output prompts.jsonl --score
"""

import argparse
import json
import os
import re
import sqlite3
import hashlib
from datetime import datetime
from pathlib import Path


def extract_text_from_parts(parts_json: str) -> str:
    """Extract text content from Freebuff message parts."""
    try:
        parts = json.loads(parts_json)
    except (json.JSONDecodeError, TypeError):
        return ""
    
    texts = []
    for part in parts:
        if isinstance(part, dict):
            if part.get("kind") == "text":
                texts.append(part.get("text", ""))
            elif part.get("type") == "text":
                texts.append(part.get("text", ""))
    return "\n".join(texts)


def classify_prompt(text: str) -> str:
    """Classify a coding prompt into a category."""
    text_lower = text.lower()
    
    # Code generation
    if any(kw in text_lower for kw in [
        "写一个", "写代码", "write a function", "write a class",
        "implement", "create a", "生成代码", "生成一个",
        "def ", "class ", "function ", "async function"
    ]):
        return "code_generation"
    
    # Debugging
    if any(kw in text_lower for kw in [
        "debug", "fix", "修复", "为什么", "报错", "error",
        "失败", "fail", "不工作", "doesn't work", "bug",
        "crash", "崩溃", "exception", "traceback"
    ]):
        return "debugging"
    
    # Refactoring
    if any(kw in text_lower for kw in [
        "refactor", "重构", "优化", "optimize", "改进",
        "improve", "clean up", "整理", "rename", "重命名"
    ]):
        return "refactoring"
    
    # Architecture / Design
    if any(kw in text_lower for kw in [
        "架构", "architecture", "设计", "design", "方案",
        "plan", "白皮书", "whitepaper", "技术方案",
        "system design", "如何", "how to"
    ]):
        return "architecture"
    
    # Code analysis / explanation
    if any(kw in text_lower for kw in [
        "分析", "analyze", "解释", "explain", "看看",
        "check", "review", "检查", "what does", "什么意思"
    ]):
        return "code_analysis"
    
    # Testing
    if any(kw in text_lower for kw in [
        "test", "测试", "unit test", "benchmark", "跑一下",
        "run", "verify", "验证", "E2E", "end to end"
    ]):
        return "testing"
    
    # DevOps / Build
    if any(kw in text_lower for kw in [
        "build", "编译", "compile", "deploy", "部署",
        "install", "安装", "config", "配置", "git",
        "commit", "push", "branch"
    ]):
        return "devops"
    
    # Data / ML
    if any(kw in text_lower for kw in [
        "model", "模型", "train", "训练", "dataset",
        "数据", "GGUF", "llama", "transformer", "KV cache",
        "embedding", "inference", "推理"
    ]):
        return "ml_inference"
    
    return "general"


def extract_coding_tasks(text: str) -> list:
    """Extract specific coding task descriptions from a message."""
    tasks = []
    
    # Split by common delimiters
    lines = text.split("\n")
    current_task = []
    
    for line in lines:
        # Skip empty lines and very short lines
        stripped = line.strip()
        if not stripped or len(stripped) < 5:
            if current_task:
                tasks.append("\n".join(current_task))
                current_task = []
            continue
        
        # Skip common non-task lines
        if any(stripped.startswith(p) for p in ["```", "---", "===", "#", "│", "├", "└"]):
            if current_task:
                tasks.append("\n".join(current_task))
                current_task = []
            continue
        
        current_task.append(stripped)
    
    if current_task:
        tasks.append("\n".join(current_task))
    
    return tasks


def compute_quality_score(text: str, role: str, context: dict) -> float:
    """
    Compute a quality score for a prompt.
    Higher = more valuable for CDPO training.
    """
    score = 0.0
    
    # Length bonus (not too short, not too long)
    if 50 < len(text) < 5000:
        score += 0.3
    elif 20 < len(text) < 50:
        score += 0.1
    
    # Has code block bonus
    if "```" in text or "def " in text or "class " in text:
        score += 0.2
    
    # Has specific technical terms
    technical_terms = [
        "python", "javascript", "typescript", "rust", "go",
        "api", "http", "json", "sql", "git", "docker",
        "pytorch", "transformer", "gguf", "llama",
        "function", "class", "import", "async", "await",
        "bug", "error", "fix", "test"
    ]
    tech_count = sum(1 for term in technical_terms if term in text.lower())
    score += min(tech_count * 0.05, 0.3)
    
    # Has question mark (inquiry = good training signal)
    if "?" in text or "？" in text:
        score += 0.1
    
    # Has file path (specific = good)
    if re.search(r'[/\\][\w.-]+\.\w+', text):
        score += 0.1
    
    # Negative: too generic
    generic_phrases = ["继续", "continue", "好的", "ok", "yes", "no", "是", "否"]
    if text.strip().lower() in generic_phrases:
        score -= 0.5
    
    return max(0.0, min(1.0, score))


def extract_conversations(db_path: str, output_path: str, score: bool = False):
    """Main extraction pipeline."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all threads
    cursor.execute("""
        SELECT t.id, t.title, t.model
        FROM threads t
        WHERE EXISTS (SELECT 1 FROM messages m WHERE m.thread_id = t.id)
    """)
    threads = cursor.fetchall()
    
    print(f"Found {len(threads)} threads with messages")
    
    prompts = []
    seen_hashes = set()
    
    for thread_id, title, model in threads:
        # Get all messages in this thread, ordered by timestamp
        cursor.execute("""
            SELECT seq, role, parts_json, ts
            FROM messages
            WHERE thread_id = ?
            ORDER BY ts
        """, (thread_id,))
        messages = cursor.fetchall()
        
        # Build conversation context
        conversation = []
        for seq, role, parts_json, ts in messages:
            text = extract_text_from_parts(parts_json)
            if text.strip():
                conversation.append({
                    "seq": seq,
                    "role": role,
                    "text": text.strip(),
                    "ts": ts
                })
        
        # Extract user prompts with context
        for i, msg in enumerate(conversation):
            if msg["role"] != "user":
                continue
            
            text = msg["text"]
            
            # Skip very short prompts
            if len(text) < 10:
                continue
            
            # Skip meta/system messages
            if text.startswith("<system>") or text.startswith("<since_"):
                continue
            
            # Compute hash for dedup
            text_hash = hashlib.md5(text.encode()).hexdigest()[:12]
            if text_hash in seen_hashes:
                continue
            seen_hashes.add(text_hash)
            
            # Get assistant response (if exists)
            response_text = ""
            for j in range(i + 1, min(i + 5, len(conversation))):
                if conversation[j]["role"] == "assistant":
                    response_text = conversation[j]["text"][:500]
                    break
            
            # Get previous context (3 messages before)
            context_msgs = []
            for j in range(max(0, i - 3), i):
                context_msgs.append({
                    "role": conversation[j]["role"],
                    "text": conversation[j]["text"][:200]
                })
            
            # Classify
            category = classify_prompt(text)
            
            # Quality score
            q_score = compute_quality_score(text, "user", {
                "thread_title": title,
                "category": category
            }) if score else None
            
            prompt_entry = {
                "id": f"freebuff_{thread_id[:8]}_{seq}",
                "thread_id": thread_id,
                "thread_title": title,
                "model": model,
                "category": category,
                "prompt": text[:2000],  # Cap length
                "response_preview": response_text,
                "context": context_msgs,
                "timestamp": msg["ts"],
                "hash": text_hash,
            }
            
            if score:
                prompt_entry["quality_score"] = round(q_score, 3)
            
            prompts.append(prompt_entry)
    
    conn.close()
    
    # Sort by quality score if scored
    if score:
        prompts.sort(key=lambda x: x.get("quality_score", 0), reverse=True)
    
    # Write output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for p in prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    
    # Summary
    print(f"\n=== Extraction Summary ===")
    print(f"Total prompts: {len(prompts)}")
    
    # Category distribution
    categories = {}
    for p in prompts:
        cat = p["category"]
        categories[cat] = categories.get(cat, 0) + 1
    print(f"\nCategory distribution:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")
    
    # Quality score distribution (if scored)
    if score:
        scores = [p["quality_score"] for p in prompts]
        print(f"\nQuality scores:")
        print(f"  Mean: {sum(scores)/len(scores):.3f}")
        print(f"  Min: {min(scores):.3f}")
        print(f"  Max: {max(scores):.3f}")
        high_quality = sum(1 for s in scores if s > 0.5)
        print(f"  High quality (>0.5): {high_quality}/{len(scores)}")
    
    print(f"\nOutput: {output_path}")
    
    return prompts


def generate_cdpo_pairs(prompts_path: str, output_path: str):
    """
    Generate CDPO chosen/rejected pairs from extracted prompts.
    
    Strategy:
    - For each prompt, create a "chosen" version (good context, clear task)
    - Create a "rejected" version (bad context, ambiguous task)
    """
    with open(prompts_path, encoding="utf-8") as f:
        prompts = [json.loads(line) for line in f if line.strip()]
    
    cdpo_pairs = []
    
    for prompt in prompts:
        text = prompt["prompt"]
        category = prompt["category"]
        
        # Chosen: the original prompt (clear, specific)
        chosen = {
            "prompt": text,
            "category": category,
            "expert_hint": "qwen36" if category in ["general", "devops"] else "ornith",
            "quality": "chosen"
        }
        
        # Rejected: perturbed version (ambiguous, missing context)
        rejected_variants = [
            # Too short / ambiguous
            text[:20] + "..." if len(text) > 20 else text,
            # With noise prefix
            "嗯 " + text,
            # Without context
            text.split("\n")[-1] if "\n" in text else text,
        ]
        
        for rejected_text in rejected_variants:
            if rejected_text != text:  # Skip if same as chosen
                rejected = {
                    "prompt": rejected_text,
                    "category": category,
                    "expert_hint": "ornith" if category in ["general", "devops"] else "qwen36",
                    "quality": "rejected"
                }
                
                cdpo_pairs.append({
                    "chosen": chosen,
                    "rejected": rejected,
                    "source_id": prompt["id"]
                })
    
    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in cdpo_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    
    print(f"Generated {len(cdpo_pairs)} CDPO pairs from {len(prompts)} prompts")
    print(f"Output: {output_path}")
    
    return cdpo_pairs


def main():
    parser = argparse.ArgumentParser(description="Extract coding prompts from Freebuff")
    parser.add_argument("--db", default=r"C:\Users\alexchuang\Desktop\fastprefill\.freebuff\desktop-v2.db",
                       help="Path to Freebuff SQLite database")
    parser.add_argument("--output", default=r"D:\alex\flashkv0516\cgcengine_full\src\fusion_route\training_data\freebuff_prompts.jsonl",
                       help="Output JSONL path")
    parser.add_argument("--score", action="store_true",
                       help="Add quality scores to prompts")
    parser.add_argument("--cdpo", action="store_true",
                       help="Also generate CDPO pairs")
    args = parser.parse_args()
    
    # Extract prompts
    prompts = extract_conversations(args.db, args.output, args.score)
    
    # Generate CDPO pairs if requested
    if args.cdpo:
        cdpo_path = args.output.replace(".jsonl", "_cdpo.jsonl")
        generate_cdpo_pairs(args.output, cdpo_path)
    
    return prompts


if __name__ == "__main__":
    main()
