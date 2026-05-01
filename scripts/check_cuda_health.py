import os
import sys
import subprocess
import torch

def check():
    print("=== System Environment ===")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')}")
    print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'NOT SET')}")
    
    print("\n=== NVIDIA System Check ===")
    try:
        smi = subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT).decode()
        print("nvidia-smi: SUCCESS")
        # Print first few lines of smi to see driver version
        print("\n".join(smi.split('\n')[:3]))
    except Exception as e:
        print(f"nvidia-smi: FAILED ({e})")

    print("\n=== PyTorch CUDA Check ===")
    cuda_available = torch.cuda.is_available()
    print(f"torch.cuda.is_available(): {cuda_available}")
    
    if not cuda_available:
        try:
            # Trigger the internal error to see if we get more detail
            torch.cuda.init()
        except Exception as e:
            print(f"torch.cuda.init() Error: {e}")
        return

    print(f"Device Count: {torch.cuda.device_count()}")
    print(f"Current Device: {torch.cuda.current_device()}")
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    
    print("\n=== Functional Test ===")
    try:
        x = torch.ones(1).cuda()
        print(f"Tensor allocation: SUCCESS (Value: {x.item()})")
        y = x + x
        print("Tensor math: SUCCESS")
    except Exception as e:
        print(f"Functional Test: FAILED ({e})")

if __name__ == "__main__":
    check()
