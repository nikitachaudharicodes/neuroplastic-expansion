#!/usr/bin/env python3
import os
import subprocess
import shutil
from pathlib import Path

def run_cmd(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.returncode == 0, result.stdout.strip()

def main():
    branches = ['gradprune', 'consolidation-ne', 'adaptive-expansion']
    output_dir = Path('all_results')
    output_dir.mkdir(exist_ok=True)
    
    # Get current branch
    _, current = run_cmd('git branch --show-current')
    print(f"Current branch: {current}\n")
    
    for branch in branches:
        print(f"📂 Processing: {branch}")
        
        # Checkout branch
        success, _ = run_cmd(f'git checkout {branch}')
        if not success:
            print(f"  ✗ Failed to checkout")
            continue
        
        # Copy results
        results = Path('results')
        if results.exists():
            count = 0
            for item in results.iterdir():
                if item.is_dir():
                    dest = output_dir / item.name
                    if dest.exists():
                        shutil.rmtree(dest)
                    shutil.copytree(item, dest)
                    count += 1
            print(f"  ✓ Copied {count} experiment(s)")
        else:
            print(f"  ⚠ No results/ directory found")
    
    # Return to original branch
    run_cmd(f'git checkout {current}')
    print(f"\n✓ Done! Returned to {current}")
    
    # List what we got
    print(f"\n=== Collected Results ===")
    if output_dir.exists():
        for item in sorted(output_dir.iterdir()):
            print(f"  • {item.name}")
    
    print(f"\nRun TensorBoard:")
    print(f"  tensorboard --logdir {output_dir.absolute()}")

if __name__ == '__main__':
    main()