import sys
import importlib
import torch

print(f"Python version: {sys.version}\n")

if torch.cuda.is_available():
    print(f"CUDA available: True")
    print(f"CUDA version: {torch.version.cuda}\n")
else:
    print("CUDA available: False\n")

libraries = ["torch", "numpy"]
for lib in libraries:
    module = importlib.import_module(lib)
    version = getattr(module, '__version__', 'Version not available')
    print(f"{lib}: {version}")