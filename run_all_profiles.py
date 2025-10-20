#!/usr/bin/env python3
"""
Batch script to run OriFlake with all three detection profiles on a single image.
Usage: python run_all_profiles.py [image_path]
"""

import os
import sys
import subprocess
from pathlib import Path

def run_profile(profile_name, image_path, output_dir):
    """Run OriFlake with a specific profile."""
    print(f"\n=== Running {profile_name} profile ===")
    cmd = [
        "python", "-m", "oriflake.main",
        "--profile", profile_name,
        "--input", image_path,
        "--output", output_dir
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"[OK] {profile_name} profile completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[FAIL] {profile_name} profile failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    # Default image path
    default_image = "images/testImg/20251016 r1 small S 140 Mo 700/5.png"
    
    # Get image path from command line or use default
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = default_image
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        print(f"Usage: python run_all_profiles.py [image_path]")
        print(f"Default: {default_image}")
        sys.exit(1)
    
    # Output directory
    output_dir = "OriFlake_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing image: {image_path}")
    print(f"Output directory: {output_dir}")
    
    # Run all three profiles
    profiles = ["loose", "balanced", "strict"]
    results = {}
    
    for profile in profiles:
        success = run_profile(profile, image_path, output_dir)
        results[profile] = success
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    
    for profile, success in results.items():
        status = "[OK] SUCCESS" if success else "[FAIL] FAILED"
        print(f"{profile:8}: {status}")
    
    # List output files
    print(f"\nOutput files in {output_dir}:")
    for file in sorted(os.listdir(output_dir)):
        if file.endswith(('.png', '.csv')):
            print(f"  {file}")
    
    print(f"\nAll profiles completed!")

if __name__ == "__main__":
    main()
