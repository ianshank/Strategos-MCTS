#!/usr/bin/env python3
"""
HuggingFace Authentication Setup.

This script helps you authenticate with HuggingFace Hub
to access gated datasets like PRIMUS.

Usage:
    python scripts/setup_hf_auth.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def setup_huggingface_auth():
    """Set up HuggingFace authentication."""
    print("=" * 60)
    print("HUGGINGFACE AUTHENTICATION SETUP")
    print("=" * 60)

    try:
        from huggingface_hub import HfFolder, login

        # Check current status
        current_token = HfFolder.get_token()
        if current_token:
            print(f"[INFO] Token already configured: {current_token[:10]}...")
            print("\nTo update token, you can:")
            print("1. Use login() function with new token")
            print("2. Delete token file at: " + str(HfFolder.path_token))
            return

        print("\nNo HuggingFace token found. Let's set one up!\n")

        print("PREREQUISITES:")
        print("1. Create HuggingFace account at https://huggingface.co/join")
        print("2. Accept PRIMUS dataset terms:")
        print("   https://huggingface.co/datasets/trendmicro-ailab/Primus-Seed")
        print("3. Generate token at https://huggingface.co/settings/tokens")

        print("\n" + "-" * 60)
        token = input("\nPaste your HuggingFace token (hf_...): ").strip()

        if not token:
            print("[ERROR] No token provided")
            return

        if not token.startswith("hf_"):
            print("[WARNING] Token doesn't start with 'hf_'. Are you sure it's correct?")

        # Save token
        print("\nSaving token...")
        login(token=token)
        print("[OK] Token saved successfully!")

        # Verify
        print("\nVerifying authentication...")
        from huggingface_hub import whoami

        user_info = whoami()
        print(f"[OK] Authenticated as: {user_info.get('name', 'Unknown')}")
        print(f"     Email: {user_info.get('email', 'Not provided')}")

        # Test PRIMUS access
        print("\nTesting PRIMUS dataset access...")
        try:
            from datasets import load_dataset

            # Just try to load dataset info (not full download)
            dataset = load_dataset(
                "trendmicro-ailab/Primus-Seed",
                split="train",
                streaming=True,  # Use streaming to avoid full download
            )

            # Get first sample to verify access
            sample = next(iter(dataset))
            print(f"[OK] PRIMUS access verified! Sample keys: {list(sample.keys())}")

        except Exception as e:
            if "gated" in str(e) or "access" in str(e).lower():
                print("[WARN] Token saved, but PRIMUS access denied.")
                print("       Make sure you've accepted the dataset terms at:")
                print("       https://huggingface.co/datasets/trendmicro-ailab/Primus-Seed")
            else:
                print(f"[WARN] Couldn't verify PRIMUS access: {e}")

        print("\n" + "=" * 60)
        print("SETUP COMPLETE")
        print("=" * 60)
        print("\nYou can now run:")
        print("  python scripts/validate_datasets.py")
        print("\nTo load PRIMUS in your code:")
        print("  from src.data.dataset_loader import PRIMUSLoader")
        print("  loader = PRIMUSLoader()")
        print("  samples = loader.load_seed(max_samples=1000)")

    except ImportError:
        print("[ERROR] huggingface_hub not installed.")
        print("Run: pip install huggingface_hub")
    except Exception as e:
        print(f"[ERROR] Setup failed: {e}")


if __name__ == "__main__":
    setup_huggingface_auth()
