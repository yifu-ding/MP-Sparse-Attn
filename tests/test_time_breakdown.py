import torch
import torch.profiler
from torch.autograd import profiler as autograd_profiler

# BLKQ = 128
# BLKK = 128
# pack_along_lastdim = True
# dual_scale = False
# quant_granularity = "tokenwise"
# data = torch.randn(24, 4096, 128, device='cuda')
# scale = torch.randn(24, 4096, 4, device='cuda')
# torch.cuda.synchronize()
# q = torch.randn(1, 24, 4096, 128, device='cuda')
# k = torch.randn(1, 24, 4096, 128, device='cuda')

def test_time_breakdown(func, *args, **kwargs):

    for _ in range(5):  # warmup
        func(*args, **kwargs)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        
        with autograd_profiler.record_function("all"):
            for _ in range(10):  # 重复执行
                func(*args, **kwargs)
        prof.step()


    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=1000,  # 默认是10，调大点
        max_name_column_width=200,  # 默认是64，调大一点以显示完整 kernel 名称
        top_level_events_only=True
    ))

    