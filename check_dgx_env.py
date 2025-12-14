import torch
import sys
import os

def check_environment():
    print("="*40)
    print("   NVIDIA DGX Spark - Environment Check")
    print("="*40)

    # 1. Check Python
    print(f"\n[Python] {sys.version.split()[0]}")

    # 2. Check CUDA/GPU
    print("\n[GPU Status]")
    if torch.cuda.is_available():
        print(f"✅ CUDA is available")
        print(f"   Device Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            
        # Simple Tensor Test
        try:
            x = torch.tensor([1.0, 2.0]).cuda()
            print("   Tensor Test: Passed (Allocated on GPU)")
        except Exception as e:
            print(f"   Tensor Test: Failed ({e})")
    else:
        print("❌ CUDA is NOT available. Models will run on CPU (Slow).")

    # 3. Check Directories
    print("\n[Project Structure]")
    required_files = ["main.py", "requirements.txt", "osha_rules.json"]
    for f in required_files:
        exists = os.path.exists(f)
        status = "✅" if exists else "❌"
        print(f"   {status} {f}")

    print("\n" + "="*40)

if __name__ == "__main__":
    try:
        check_environment()
    except ImportError:
        print("❌ Error: torch not installed. Run 'pip install -r requirements.txt' first.")
