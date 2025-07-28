import torch
import torch.profiler
from ours.mxfp import MXFP4Tensor, MXScaleTensor
from torch.autograd import profiler as autograd_profiler
from ours.quant_kernels import quant_mxfp4

BLKQ = 128
BLKK = 128
pack_along_lastdim = True
dual_scale = False
quant_granularity = "tokenwise"
data = torch.randn(24, 4096, 128, device='cuda')
scale = torch.randn(24, 4096, 4, device='cuda')
torch.cuda.synchronize()
q = torch.randn(1, 24, 4096, 128, device='cuda')
k = torch.randn(1, 24, 4096, 128, device='cuda')

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    

    for _ in range(1):

        with autograd_profiler.record_function("all"):
            mxfp = MXFP4Tensor(data=data)
            # torch.cuda.synchronize()

        # with autograd_profiler.record_function("to_packed"):
            packed = mxfp.to_packed_tensor(dim=2)
            # torch.cuda.synchronize()
        
        # with autograd_profiler.record_function("cvt_scale"):
            scale_cvt = MXScaleTensor(scale.to(torch.float32))        
            torch.cuda.synchronize()

            mxfp = MXFP4Tensor(data=data)
            # torch.cuda.synchronize()

        # with autograd_profiler.record_function("to_packed"):
            packed = mxfp.to_packed_tensor(dim=2)
            # torch.cuda.synchronize()
        
        # with autograd_profiler.record_function("cvt_scale"):
            scale_cvt = MXScaleTensor(scale.to(torch.float32))        
            torch.cuda.synchronize()
        
        # q_fp, q_scale, k_fp, k_scale, q_scale_2, k_scale_2 = quant_mxfp4(q, k, BLKQ=BLKQ, BLKK=BLKK, \
        #     pack_along_lastdim=pack_along_lastdim, dual_scale=dual_scale, quant_granularity=quant_granularity)

        prof.step()

# print(prof.key_averages().table(sort_by="cuda_time_total"))

print(prof.key_averages().table(
    sort_by="cuda_time_total",
    row_limit=1000,  # 默认是10，调大点
    max_name_column_width=200,  # 默认是64，调大一点以显示完整 kernel 名称
    top_level_events_only=True
))