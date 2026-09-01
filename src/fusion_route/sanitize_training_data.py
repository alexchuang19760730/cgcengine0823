#!/usr/bin/env python3
"""
Sanitize training data by removing secrets, tokens, and credentials.
Run before committing to GitHub.
"""

import json
import re
import os


# Patterns to detect and redact
SECRET_PATTERNS = [
    (r'hf_[A-Za-z0-9]{20,}', '[HF_TOKEN_REDACTED]'),
    (r'sk-[A-Za-z0-9]{20,}', '[OPENAI_KEY_REDACTED]'),
    (r'ghp_[A-Za-z0-9]{20,}', '[GITHUB_TOKEN_REDACTED]'),
    (r'xoxb-[A-Za-z0-9-]+', '[SLACK_TOKEN_REDACTED]'),
    (r'AKIA[A-Z0-9]{16}', '[AWS_KEY_REDACTED]'),
    (r'password\s*[=:]\s*["\']?[^\s"\']+', '[PASSWORD_REDACTED]'),
    (r'token\s*[=:]\s*["\']?[^\s"\']+', '[TOKEN_REDACTED]'),
    (r'api[_-]?key\s*[=:]\s*["\']?[^\s"\']+', '[API_KEY_REDACTED]'),
    (r'secret\s*[=:]\s*["\']?[^\s"\']+', '[SECRET_REDACTED]'),
    (r'-----BEGIN (RSA |EC |DSA )?PRIVATE KEY-----', '[PRIVATE_KEY_REDACTED]'),
]


def sanitize_text(text: str) -> tuple:
    """Redact secrets from text. Returns (sanitized_text, count_of_redactions)."""
    count = 0
    for pattern, replacement in SECRET_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            count += len(matches)
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text, count


def sanitize_jsonl(input_path: str, output_path: str):
    """Sanitize a JSONL file."""
    total_redactions = 0
    
    with open(input_path, encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        
        for line_num, line in enumerate(fin, 1):
            try:
                data = json.loads(line)
                file_redactions = 0
                
                # Recursively sanitize all string values
                def sanitize_obj(obj):
                    nonlocal file_redactions
                    if isinstance(obj, str):
                        sanitized, count = sanitize_text(obj)
                        file_redactions += count
                        return sanitized
                    elif isinstance(obj, dict):
                        return {k: sanitize_obj(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [sanitize_obj(item) for item in obj]
                    return obj
                
                data = sanitize_obj(data)
                total_redactions += file_redactions
                
                fout.write(json.dumps(data, ensure_ascii=False) + '\n')
                
                if file_redactions > 0:
                    print(f'  Line {line_num}: {file_redactions} redactions')
                    
            except json.JSONDecodeError:
                # Keep invalid lines as-is
                fout.write(line)
    
    return total_redactions


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Sanitize training data')
    parser.add_argument('--input', default=r'src/fusion_route/training_data/freebuff_prompts_cdpo.jsonl')
    parser.add_argument('--output', default=r'src/fusion_route/training_data/freebuff_prompts_cdpo.jsonl')
    args = parser.parse_args()
    
    # If output is same as input, backup first
    if args.input == args.output:
        backup = args.input + '.bak'
        import shutil
        shutil.copy2(args.input, backup)
        print(f'Backup: {backup}')
    
    print(f'Sanitizing: {args.input}')
    count = sanitize_jsonl(args.input, args.output)
    print(f'\nTotal redactions: {count}')
    
    # Also sanitize the prompts file
    prompts_in = args.input.replace('_cdpo.jsonl', '.jsonl')
    if os.path.exists(prompts_in):
        prompts_out = prompts_in + '.sanitized'
        print(f'\nSanitizing: {prompts_in}')
        count2 = sanitize_jsonl(prompts_in, prompts_out)
        print(f'Total redactions: {count2}')
        
        if count2 > 0:
            # Replace original
            os.replace(prompts_out, prompts_in)
            print(f'Replaced: {prompts_in}')
        else:
            os.remove(prompts_out)
            print('No secrets found, keeping original')


if __name__ == '__main__':
    main()
