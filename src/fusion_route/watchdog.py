#!/usr/bin/env python3
"""
Freebuff Session Watchdog
监控 desktop-v2.db 变化，自动提取新增 coding prompts。

用法:
  # 后台运行（推荐）
  python watchdog.py --daemon
  
  # 前台运行（调试）
  python watchdog.py --verbose
  
  # 只检查一次
  python watchdog.py --once

原理:
  1. 每 30 秒检查 DB 文件修改时间 + 消息数量
  2. 检测到新消息后，等 60 秒（session 可能还在进行）
  3. 60 秒内无新消息 → 触发提取
  4. 提取完成后，记录已处理的最大 seq，避免重复

支持多账号:
  - 每个 Freebuff 实例有独立的 desktop-v2.db
  - 通过 --db 指定不同路径，或自动发现所有实例
"""

import argparse
import hashlib
import json
import os
import signal
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path


# Default paths
FREEBUFF_DB_PATTERNS = [
    r"C:\Users\*\Desktop\fastprefill\.freebuff\desktop-v2.db",
    r"C:\Users\*\Documents\*\.freebuff\desktop-v2.db",
    r"C:\Users\*\Desktop\*\.freebuff\desktop-v2.db",
]

EXTRACT_SCRIPT = os.path.join(os.path.dirname(__file__), "extract_prompts.py")
STATE_FILE = os.path.join(os.path.dirname(__file__), "training_data", "watchdog_state.json")


class WatchdogState:
    """Persistent state for tracking what's been extracted."""
    
    def __init__(self, state_file: str):
        self.state_file = state_file
        self.state = self._load()
    
    def _load(self) -> dict:
        try:
            with open(self.state_file, encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"dbs": {}, "last_run": None, "total_extracted": 0}
    
    def save(self):
        os.makedirs(os.path.dirname(self.state_file) or ".", exist_ok=True)
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2, ensure_ascii=False)
    
    def get_last_seq(self, db_path: str) -> int:
        return self.state.get("dbs", {}).get(db_path, {}).get("last_seq", 0)
    
    def set_last_seq(self, db_path: str, seq: int, count: int):
        if "dbs" not in self.state:
            self.state["dbs"] = {}
        self.state["dbs"][db_path] = {
            "last_seq": seq,
            "last_extracted": datetime.now().isoformat(),
            "count": count
        }
        self.state["last_run"] = datetime.now().isoformat()
        self.state["total_extracted"] = self.state.get("total_extracted", 0) + count
        self.save()


def find_freebuff_dbs() -> list:
    """Auto-discover Freebuff DB instances."""
    import glob
    dbs = []
    for pattern in FREEBUFF_DB_PATTERNS:
        dbs.extend(glob.glob(pattern))
    return list(set(dbs))  # deduplicate


def get_message_count(db_path: str) -> tuple:
    """Get total message count and max seq from a Freebuff DB."""
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*), COALESCE(MAX(seq), 0) FROM messages")
        count, max_seq = cursor.fetchone()
        conn.close()
        return count, max_seq
    except Exception as e:
        return 0, 0


def extract_new_prompts(db_path: str, last_seq: int, output_path: str) -> tuple:
    """Extract only new prompts since last_seq."""
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cursor = conn.cursor()
        
        # Get new messages
        cursor.execute("""
            SELECT seq, thread_id, role, parts_json, ts
            FROM messages
            WHERE seq > ? AND role = 'user'
            ORDER BY seq
        """, (last_seq,))
        
        new_messages = cursor.fetchall()
        conn.close()
        
        if not new_messages:
            return 0, last_seq
        
        # Load existing prompts for dedup
        existing_hashes = set()
        if os.path.exists(output_path):
            with open(output_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        existing_hashes.add(d.get("hash", ""))
                    except:
                        pass
        
        # Extract and classify
        sys.path.insert(0, os.path.dirname(__file__))
        from extract_prompts import classify_prompt, compute_quality_score, extract_text_from_parts
        
        new_prompts = []
        max_seq = last_seq
        
        for seq, thread_id, role, parts_json, ts in new_messages:
            text = extract_text_from_parts(parts_json)
            if not text.strip() or len(text.strip()) < 10:
                continue
            
            # Skip meta messages
            if text.startswith("<system>") or text.startswith("<since_"):
                continue
            
            text_hash = hashlib.md5(text.encode()).hexdigest()[:12]
            if text_hash in existing_hashes:
                continue
            
            category = classify_prompt(text)
            q_score = compute_quality_score(text, "user", {"category": category})
            
            new_prompts.append({
                "id": f"watchdog_{thread_id[:8]}_{seq}",
                "thread_id": thread_id,
                "category": category,
                "prompt": text[:2000],
                "timestamp": ts,
                "hash": text_hash,
                "quality_score": round(q_score, 3),
            })
            
            existing_hashes.add(text_hash)
            max_seq = max(max_seq, seq)
        
        # Append to output
        if new_prompts:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "a", encoding="utf-8") as f:
                for p in new_prompts:
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")
        
        return len(new_prompts), max_seq
        
    except Exception as e:
        print(f"[ERROR] Extract failed: {e}", file=sys.stderr)
        return 0, last_seq


def run_watchdog(db_path: str, output_path: str, state: WatchdogState, 
                 poll_interval: int = 30, quiet_period: int = 60, verbose: bool = False):
    """Main watchdog loop."""
    print(f"[WATCHDOG] Monitoring: {db_path}")
    print(f"[WATCHDOG] Output: {output_path}")
    print(f"[WATCHDOG] Poll: {poll_interval}s, Quiet: {quiet_period}s")
    print(f"[WATCHDOG] Press Ctrl+C to stop\n")
    
    last_count = 0
    last_change_time = time.time()
    extracting = False
    
    while True:
        try:
            current_count, current_max_seq = get_message_count(db_path)
            
            if verbose:
                now = datetime.now().strftime("%H:%M:%S")
                print(f"[{now}] Messages: {current_count}, Max seq: {current_max_seq}")
            
            # Detect new messages
            if current_count > last_count:
                last_change_time = time.time()
                if verbose:
                    print(f"  → New messages detected! (+{current_count - last_count})")
                last_count = current_count
                extracting = False
            
            # Check if quiet period has passed (no new messages for quiet_period seconds)
            quiet_elapsed = time.time() - last_change_time
            if (not extracting and 
                current_count > 0 and 
                quiet_elapsed > quiet_period and
                current_count > state.get_last_seq(db_path)):
                
                print(f"\n[TRIGGER] Session quiet for {quiet_elapsed:.0f}s, extracting...")
                extracting = True
                
                last_seq = state.get_last_seq(db_path)
                new_count, new_max_seq = extract_new_prompts(db_path, last_seq, output_path)
                
                if new_count > 0:
                    state.set_last_seq(db_path, new_max_seq, new_count)
                    print(f"[DONE] Extracted {new_count} new prompts (total: {state.state.get('total_extracted', 0)})")
                else:
                    print(f"[SKIP] No new valid prompts found")
                
                # Regenerate CDPO pairs
                try:
                    cdpo_path = output_path.replace(".jsonl", "_cdpo.jsonl")
                    from extract_prompts import generate_cdpo_pairs
                    pairs = generate_cdpo_pairs(output_path, cdpo_path)
                    print(f"[CDPO] Regenerated {len(pairs)} pairs")
                except Exception as e:
                    print(f"[WARN] CDPO generation failed: {e}")
                
                extracting = False
                last_count = current_count
            
            time.sleep(poll_interval)
            
        except KeyboardInterrupt:
            print("\n[STOP] Watchdog stopped")
            break
        except Exception as e:
            print(f"[ERROR] {e}", file=sys.stderr)
            time.sleep(poll_interval)


def main():
    parser = argparse.ArgumentParser(description="Freebuff Session Watchdog")
    parser.add_argument("--db", help="Path to desktop-v2.db (auto-discover if omitted)")
    parser.add_argument("--output", default=r"D:\alex\flashkv0516\cgcengine_full\src\fusion_route\training_data\freebuff_prompts.jsonl")
    parser.add_argument("--poll", type=int, default=30, help="Poll interval in seconds")
    parser.add_argument("--quiet", type=int, default=60, help="Quiet period before trigger")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--daemon", action="store_true", help="Run as background daemon")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    
    # Find DB
    if args.db:
        dbs = [args.db]
    else:
        dbs = find_freebuff_dbs()
        if not dbs:
            print("[ERROR] No Freebuff DB found. Use --db to specify path.")
            sys.exit(1)
    
    print(f"[INIT] Found {len(dbs)} Freebuff DB(s)")
    for db in dbs:
        count, max_seq = get_message_count(db)
        print(f"  → {db} ({count} messages, max_seq={max_seq})")
    
    # Load state
    state = WatchdogState(STATE_FILE)
    
    if args.once:
        # Run extraction once for each DB
        for db in dbs:
            last_seq = state.get_last_seq(db)
            new_count, new_max_seq = extract_new_prompts(db, last_seq, args.output)
            if new_count > 0:
                state.set_last_seq(db, new_max_seq, new_count)
                print(f"[{db}] Extracted {new_count} new prompts")
            else:
                print(f"[{db}] No new prompts")
        return
    
    # Daemon mode - fork to background
    if args.daemon:
        print("[DAEMON] Starting in background...")
        # On Windows, use subprocess
        import subprocess
        cmd = [sys.executable, __file__, 
               "--db", dbs[0], 
               "--output", args.output,
               "--poll", str(args.poll),
               "--quiet", str(args.quiet)]
        if args.verbose:
            cmd.append("--verbose")
        
        # Write log file
        log_path = os.path.join(os.path.dirname(args.output), "watchdog.log")
        with open(log_path, "w") as log:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        print(f"[DAEMON] PID: {proc.pid}, Log: {log_path}")
        return
    
    # Foreground mode
    for db in dbs:
        run_watchdog(db, args.output, state, 
                    poll_interval=args.poll, 
                    quiet_period=args.quiet,
                    verbose=args.verbose)


if __name__ == "__main__":
    main()
