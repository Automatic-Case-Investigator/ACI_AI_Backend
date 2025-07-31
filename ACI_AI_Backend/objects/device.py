import torch

DEVICE = "cpu"

def get_freest_gpu():
    if not torch.cuda.is_available():
        return None

    num_gpus = torch.cuda.device_count()
    min_allocated_mem = float('inf')
    best_gpu = 0

    for i in range(num_gpus):
        torch.cuda.reset_peak_memory_stats(i)
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated(i)
        if allocated < min_allocated_mem:
            min_allocated_mem = allocated
            best_gpu = i

    return best_gpu

def update_current_device():
    gpu_id = get_freest_gpu()
    global DEVICE
    DEVICE = torch.device(f"cuda:{gpu_id}" if gpu_id is not None else "cpu")
