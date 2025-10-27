import os, time, json, psutil, torch
from contextlib import contextmanager

def device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def mem_gb():
    vmem = psutil.virtual_memory()
    return dict(total_gb=vmem.total/1e9, used_gb=vmem.used/1e9, avail_gb=vmem.available/1e9)

@contextmanager
def timer(name):
    t0=time.time(); yield; dt=time.time()-t0
    print(json.dumps({"timer_s":{name:round(dt,3)}}))
