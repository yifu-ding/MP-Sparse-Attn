import os
import torch

# 定义文件路径
input_path = os.path.join('./saved_files', 'input_tensors.pth')
quant_path = os.path.join('./saved_files', 'quant_tensors.pth')
scale_path = os.path.join('./saved_files', 'triton_q_qk.pth')

# 读取.pth文件
input_tensors = torch.load(input_path, map_location=torch.device('cpu'))
quant_tensors = torch.load(quant_path, map_location=torch.device('cpu'))
triton_q_qk = torch.load(scale_path, map_location=torch.device('cpu'))

# 从quant_tensors字典中提取各个tensor
a_quant = quant_tensors['a_quant']  # torch.Size([1, 4, 129, 128])
a_scale = quant_tensors['a_scale']  # torch.Size([1, 4, 2, 1, 32, 4, 4])
b_quant = quant_tensors['b_quant']  # torch.Size([1, 4, 129, 128])
b_scale = quant_tensors['b_scale']  # torch.Size([1, 4, 2, 1, 32, 4, 4])

scale_q0 = triton_q_qk['scale_q0']  # torch.Size([128, 4])
scale_q1 = triton_q_qk['scale_q1']  # torch.Size([128, 4])
qk0 = triton_q_qk['qk0']  # torch.Size([128, 128])
qk1 = triton_q_qk['qk1']  # torch.Size([128, 128])
scale_q0_ptr = triton_q_qk['scale_q0_ptr']  # torch.Size([1, 512])
scale_q1_ptr = triton_q_qk['scale_q1_ptr']  # torch.Size([1, 512])
# a_scale = triton_q_qk['a_scale']  # torch.Size([1, 4, 2, 1, 32, 4, 4])
# b_scale = triton_q_qk['b_scale']  # torch.Size([1, 4, 2, 1, 32, 4, 4])

query = input_tensors['q'] # torch.Size([1, 4, 129, 128])
key = input_tensors['k'] # torch.Size([1, 4, 129, 128])
value = input_tensors['v'] # torch.Size([1, 4, 129, 128])

import pdb; pdb.set_trace() 