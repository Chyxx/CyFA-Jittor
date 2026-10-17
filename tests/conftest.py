import importlib.abc
import sys


class NoBridge(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in ("torch", "triton"):
            raise ImportError("Native tests forbid Torch and Triton")


sys.meta_path.insert(0, NoBridge())
from cyclic_flow_attention.utils import setup_cuda

setup_cuda(42)
