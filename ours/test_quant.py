import torch
from spas_sage_attn.quant_per_block import per_block_int8
from quant_mxint8 import quant_fpxint8, quant_mxfp8e5, quant_mxfp4

from block_scaled_matmul import initialize_block_scaled_from_tensor, block_scaled_matmul
from batched_block_scaled_matmul import initialize_block_scaled_batched_from_tensor, block_scaled_batched_matmul

def test_per_block_int8():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Generate random input data (fp16)
    batch_size = 1
    num_heads = 1
    seq_len = 256
    head_dim = 128
    BLKQ = 128
    
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')
    
    # Quantize to int8
    q_int8, q_scale, k_int8, k_scale = per_block_int8(q, k, BLKQ=BLKQ)
    # import pdb; pdb.set_trace()
    
    # Dequantize
    # q_scale: (b, h, nblock, 1) -> squeeze(-1) -> (b, h, nblock)
    q_scale_squeezed = q_scale.squeeze(-1)
    # repeat_interleave to seq_len
    q_scale_expanded = q_scale_squeezed.repeat_interleave(BLKQ, dim=2)[..., :seq_len]
    # (b, h, seq_len, 1) for broadcasting
    q_scale_expanded = q_scale_expanded.unsqueeze(-1)
    q_dequant = q_int8.float() * q_scale_expanded
    
    # Calculate quantization error
    mse_loss = torch.nn.functional.mse_loss(q_dequant, q.float() * (head_dim ** -0.5) )
    max_error = torch.max(torch.abs(q_dequant - q.float() * (head_dim ** -0.5) ))
    
    print(f"MSE loss after quantization: {mse_loss.item():.6f}")
    print(f"Maximum absolute error: {max_error.item():.6f}")
    
    # Verify quantized values are within int8 range
    assert torch.all(q_int8 >= -128) and torch.all(q_int8 <= 127), "Quantized values exceed int8 range"

    # print(f"q: {q[0, 0, :, :]}")
    # print(f"q_dequant: {q_dequant[0, 0, :, :]}")
    
    # Calculate QK before quantization
    qk_ref = torch.matmul(q.float(), k.float().transpose(-2, -1)) * (head_dim ** -0.5) 
    
    # Calculate QK after quantization
    # First calculate matrix multiplication of int8 q and k
    qk_int8 = torch.matmul(q_int8.float(), k_int8.float().transpose(-2, -1))
    
    # Expand scale to corresponding sequence length
    k_scale_squeezed = k_scale.squeeze(-1)
    k_scale_expanded = k_scale_squeezed.repeat_interleave(BLKQ, dim=2)[..., :seq_len]
    
    # Calculate final qk, multiply with scales of q and k
    qk_quant = qk_int8 * q_scale_expanded * k_scale_expanded.unsqueeze(-1)
    
    # Calculate QK quantization error
    qk_mse_loss = torch.nn.functional.mse_loss(qk_quant, qk_ref)
    qk_max_error = torch.max(torch.abs(qk_quant - qk_ref))
    
    print(f"QK MSE loss after quantization: {qk_mse_loss.item():.6f}")
    print(f"QK maximum absolute error: {qk_max_error.item():.6f}")

    # Calculate QK after softmax
    
    # Apply softmax
    qk_ref_softmax = torch.nn.functional.softmax(qk_ref, dim=-1)
    qk_quant_softmax = torch.nn.functional.softmax(qk_quant, dim=-1)
    
    # Calculate error after softmax
    softmax_mse_loss = torch.nn.functional.mse_loss(qk_quant_softmax, qk_ref_softmax)
    softmax_max_error = torch.max(torch.abs(qk_quant_softmax - qk_ref_softmax))
    
    print(f"MSE loss after softmax: {softmax_mse_loss.item():.6f}")
    print(f"Maximum absolute error after softmax: {softmax_max_error.item():.6f}")

    # import pdb; pdb.set_trace()
    
    return mse_loss.item(), max_error.item()

def test_quant_fpxint8():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Generate random input data (fp16)
    batch_size = 1
    num_heads = 1
    seq_len = 256
    head_dim = 128
    BLKQ = 128
    
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')
    
    # Quantize to int8
    q_int8, q_scale, k_int8, k_scale = quant_fpxint8(q, k, BLKQ=BLKQ)
    
    # Dequantize
    # q_scale: (b, h, nblock, head_dim//32, 1) or (b, h, nblock, head_dim//32)
    # if q_scale.shape[-1] == 1:
    #     q_scale = q_scale.squeeze(-1)  # (b, h, nblock, head_dim//32)
    # Expand to (b, h, nblock, head_dim)
    q_scale_expanded = q_scale.repeat_interleave(32, dim=-1)  # (b, h, nblock, head_dim)
    # Expand to (b, h, seq_len, head_dim)
    # q_scale_expanded = q_scale_expanded.repeat_interleave(BLKQ, dim=2)[..., :seq_len, :]  # (b, h, seq_len, head_dim)
    print('q_int8.shape:', q_int8.shape)
    print('q_scale_expanded.shape:', q_scale_expanded.shape)
    # import pdb; pdb.set_trace()
    q_dequant = q_int8.to(torch.float32) * (q_scale_expanded.to(torch.float32))
    

    # Calculate quantization error
    mse_loss = torch.nn.functional.mse_loss(q_dequant, q.float() * (head_dim ** -0.5))
    max_error = torch.max(torch.abs(q_dequant - q.float() * (head_dim ** -0.5)))
    
    print(f"MSE loss after quantization: {mse_loss.item():.6f}")
    print(f"maximum absolute error: {max_error.item():.6f}")
    
    # Verify quantized values are within int8 range
    assert torch.all(q_int8 >= -128) and torch.all(q_int8 <= 127), "Quantized values exceed int8 range"
    
    # Calculate QK before quantization
    qk_ref = torch.matmul(q.float(), k.float().transpose(-2, -1)) * (head_dim ** -0.5) 
    
    # Calculate QK after quantization
    # First dequantize k
    k_scale = k_scale.squeeze(-1) if k_scale.shape[-1] == 1 else k_scale
    k_scale_expanded = k_scale.repeat_interleave(32, dim=-1)[..., :head_dim]
    # k_scale_expanded = k_scale_expanded.repeat_interleave(BLKQ, dim=2)[..., :seq_len, :]
    k_dequant = k_int8.to(torch.float32) * (k_scale_expanded.to(torch.float32))
    
    # Calculate matrix multiplication of dequantized q and k
    qk_quant = torch.matmul(q_dequant, k_dequant.transpose(-2, -1)) 
    
    # Calculate QK quantization error
    qk_mse_loss = torch.nn.functional.mse_loss(qk_quant, qk_ref)
    qk_max_error = torch.max(torch.abs(qk_quant - qk_ref))
    
    print(f"QK MSE loss after quantization: {qk_mse_loss.item():.6f}")
    print(f"QK maximum absolute error: {qk_max_error.item():.6f}")
    
    # Calculate QK after softmax
    qk_ref_softmax = torch.nn.functional.softmax(qk_ref, dim=-1)
    qk_quant_softmax = torch.nn.functional.softmax(qk_quant, dim=-1)
    
    # Calculate error after softmax
    softmax_mse_loss = torch.nn.functional.mse_loss(qk_quant_softmax, qk_ref_softmax)
    softmax_max_error = torch.max(torch.abs(qk_quant_softmax - qk_ref_softmax))
    
    print(f"MSE loss after softmax: {softmax_mse_loss.item():.6f}")
    print(f"maximum absolute error after softmax: {softmax_max_error.item():.6f}")    
    
    return mse_loss.item(), max_error.item()

def test_quant_mxfp8e5(q, k, batch_size = 1, num_heads = 1, seq_len = 256, head_dim = 128, BLKQ = 128):

    # Quantize to int8 (MX format)
    q_fp8, q_scale, k_fp8, k_scale = quant_mxfp8e5(q, k, BLKQ=BLKQ)

    q_scale_expanded = q_scale.repeat_interleave(32, dim=-1)
    k_scale_expanded = k_scale.repeat_interleave(32, dim=-1)
    q_dequant = q_fp8.float() * q_scale_expanded
    
    # Calculate quantization error
    mse_loss = torch.nn.functional.mse_loss(q_dequant, q.float() * (head_dim ** -0.5) )
    max_error = torch.max(torch.abs(q_dequant - q.float() * (head_dim ** -0.5) ))
    
    print(f"MX format MSE loss after quantization: {mse_loss.item():.6f}")
    print(f"MX format maximum absolute error: {max_error.item():.6f}")
    
    # Calculate QK before quantization
    qk_ref = torch.matmul(q.float(), k.float().transpose(-2, -1)) * (head_dim ** -0.5)
    
    # Calculate QK after quantization
    # First dequantize q and k separately
    # q_dequant = q_fp8.float() * q_scale_expanded
    k_dequant = k_fp8.float() * k_scale_expanded
    
    # Then calculate QK
    qk_quant = torch.matmul(q_dequant, k_dequant.transpose(-2, -1)) 
    # print(f"qk_quant in manual fp8: {qk_quant}")
    # Calculate QK quantization error
    qk_mse_loss = torch.nn.functional.mse_loss(qk_quant, qk_ref)
    qk_max_error = torch.max(torch.abs(qk_quant - qk_ref)/(qk_ref+1e-6))
    
    print(f"MX format QK MSE loss after quantization: {qk_mse_loss.item():.6f}")
    print(f"MX format QK maximum absolute error (relative): {qk_max_error.item():.6f}")
    
    # Print some scale statistics for analysis
    # print(f"q_scale range: [{q_scale.min().item():.6f}, {q_scale.max().item():.6f}]")
    # print(f"q_scale mean: {q_scale.mean().item():.6f}")
    # print(f"q_scale standard deviation: {q_scale.std().item():.6f}")
    
    # Calculate QK after softmax
    qk_ref_softmax = torch.nn.functional.softmax(qk_ref, dim=-1)
    qk_quant_softmax = torch.nn.functional.softmax(qk_quant, dim=-1)
    
    # Calculate error after softmax
    softmax_mse_loss = torch.nn.functional.mse_loss(qk_quant_softmax, qk_ref_softmax)
    softmax_max_error = torch.max(torch.abs(qk_quant_softmax - qk_ref_softmax)/(qk_ref_softmax+1e-6))
    
    print(f"MSE loss after softmax: {softmax_mse_loss.item():.6f}")
    print(f"maximum absolute error after softmax (relative): {softmax_max_error.item():.6f}")
    
    # return mse_loss.item(), max_error.item()
    return qk_quant


def test_quant_mxfp4(q, k, 
                     batch_size = 1,
                     num_heads = 1,
                     seq_len = 256,
                     head_dim = 128,
                     BLKQ = 128):

    # Quantize to float4 (MX format)
    q_fp4, q_scale, k_fp4, k_scale = quant_mxfp4(q, k, BLKQ=BLKQ)
    # Convert uint8 to float4 format
    # Extract sign, exp, mantissa from uint8
    sign = (q_fp4 >> 3) & 0x1  # Highest bit is sign bit
    exp = (q_fp4 >> 1) & 0x3   # Next 2 bits are exponent bits
    mantissa = q_fp4 & 0x1  # Last bit is mantissa bit
    
    # Rebuild float value
    # 1. Calculate mantissa part
    mantissa_value = torch.where(exp == 0, mantissa.float() * 0.5, 1.0 + mantissa.float() * 0.5)
    # 2. Calculate exponent part: 2^exp
    bias = 1.0
    exp_value = torch.pow(2.0, exp.float() - bias)
    # 3. Apply sign bit
    q_fp4_float = mantissa_value * exp_value
    q_fp4_float = torch.where(sign.bool(), -q_fp4_float, q_fp4_float)
    # import pdb; pdb.set_trace()
    # Dequantize
    # q_scale: (b, h, seq_len, head_dim//32)
    # First repeat_interleave 32 in last dimension to get (b, h, seq_len, head_dim)
    q_scale_expanded = q_scale.repeat_interleave(32, dim=-1)
    k_scale_expanded = k_scale.repeat_interleave(32, dim=-1)
    q_dequant = q_fp4_float.float() * q_scale_expanded
    
    # import pdb; pdb.set_trace()
    
    # Calculate quantization error
    mse_loss = torch.nn.functional.mse_loss(q_dequant, q.float() * (head_dim ** -0.5))
    max_error = torch.max(torch.abs(q_dequant - q.float() * (head_dim ** -0.5)))
    
    print(f"Float4 MX format MSE loss after quantization: {mse_loss.item():.6f}")
    print(f"Float4 MX format maximum absolute error: {max_error.item():.6f}")

    # Calculate QK 
    qk_ref = torch.matmul(q.float(), k.float().transpose(-2, -1)) * (head_dim ** -0.5)

    sign = (k_fp4 >> 3) & 0x1  # Highest bit is sign bit
    exp = (k_fp4 >> 1) & 0x3   # Next 2 bits are exponent bits
    mantissa = k_fp4 & 0x1  # Last bit is mantissa bit
    
    mantissa_value = torch.where(exp == 0, mantissa.float() * 0.5, 1.0 + mantissa.float() * 0.5)
    bias = 1.0
    exp_value = torch.pow(2.0, exp.float() - bias)
    k_fp4_float = mantissa_value * exp_value
    k_fp4_float = torch.where(sign.bool(), -k_fp4_float, k_fp4_float)
    k_dequant = k_fp4_float * k_scale_expanded
    
    # Then calculate QK
    qk_quant = torch.matmul(q_dequant, k_dequant.transpose(-2, -1))
    
    # Calculate QK quantization error
    qk_mse_loss = torch.nn.functional.mse_loss(qk_quant, qk_ref)
    qk_max_error = torch.max(torch.abs(qk_quant - qk_ref) / (qk_ref+1e-6))
    
    print(f"Float4 MX format QK MSE loss after quantization: {qk_mse_loss.item():.6f}")
    print(f"Float4 MX format QK maximum absolute error (relative): {qk_max_error.item():.6f}")
    # print(f"qk_quant in manual fp4: {qk_quant}")
    
    # Calculate error after softmax
    qk_ref_softmax = torch.nn.functional.softmax(qk_ref, dim=-1)
    qk_quant_softmax = torch.nn.functional.softmax(qk_quant, dim=-1)
    
    softmax_mse_loss = torch.nn.functional.mse_loss(qk_quant_softmax, qk_ref_softmax)
    softmax_max_error = torch.max(torch.abs(qk_quant_softmax - qk_ref_softmax) / (qk_ref_softmax+1e-6))
    
    print(f"Float4 MX format MSE loss after softmax: {softmax_mse_loss.item():.6f}")
    print(f"Float4 MX format maximum absolute error after softmax (relative): {softmax_max_error.item():.6f}")

    # return mse_loss.item(), max_error.item()
    # return k_fp4, k_scale


def test_quant_mxfpx_block_scaled(q, k, block_scale_type="mxfp4", head_dim=256, batch_size=1, num_heads=1, seq_len=256, BLKQ=128):
    # Quantize to float4 (MX format)
    if block_scale_type == "mxfp4":
        q_fp4, q_scale, k_fp4, k_scale = quant_mxfp4(q, k, BLKQ=BLKQ)
        q_quant = q_fp4
        k_quant = k_fp4
    else:
        q_fp8, q_scale, k_fp8, k_scale = quant_mxfp8e5(q, k, BLKQ=BLKQ)
        q_quant = q_fp8
        k_quant = k_fp8

    q_quant = q_quant.reshape(seq_len, head_dim)
    k_quant = k_quant.reshape(seq_len, head_dim)
    q_scale = q_scale.reshape(seq_len//128, 4, 32, head_dim//32//4, 4).permute(0, 3, 2, 1, 4).contiguous()
    k_scale = k_scale.reshape(seq_len//128, 4, 32, head_dim//32//4, 4).permute(0, 3, 2, 1, 4).contiguous()
    
    q_packed, q_scale, k_packed, k_scale, configs, (reference, q_dequant, k_dequant) = \
        initialize_block_scaled_from_tensor(q_quant, k_quant, q_scale, k_scale, block_scale_type=block_scale_type, compute_reference=True)
    # q_scale2_expanded = q_scale2.repeat_interleave(32, dim=-1)
    # q_dequant2 = q_fp8.float() * q_scale2_expanded
    # import pdb; pdb.set_trace()
    
    output = block_scaled_matmul(q_packed, q_scale, k_packed, k_scale, torch.float16, seq_len, seq_len, head_dim, configs)
    torch.testing.assert_close(reference, output.to(torch.float32), atol=1e-3, rtol=1e-3)
    print(f"✅ (pass {block_scale_type} block scaled)")
    
    q_dequant = q_dequant.reshape(q.shape)
    mse_loss = torch.nn.functional.mse_loss(q_dequant, q.float() * (head_dim ** -0.5))
    max_error = torch.max(torch.abs(q_dequant - q.float() * (head_dim ** -0.5)))
    
    print(f"{block_scale_type} MX format MSE loss after quantization: {mse_loss.item():.6f}")
    print(f"{block_scale_type} MX format maximum absolute error: {max_error.item():.6f}")

    qk_ref = torch.matmul(q.float(), k.float().transpose(-2, -1)) * (head_dim ** -0.5)
    
    qk_quant = output.view(qk_ref.shape)
    qk_mse_loss = torch.nn.functional.mse_loss(qk_quant, qk_ref)
    qk_max_error = torch.max(torch.abs(qk_quant - qk_ref) / (qk_ref+1e-6))

    print(f"{block_scale_type} MX format QK MSE loss after quantization: {qk_mse_loss.item():.6f}")
    print(f"{block_scale_type} MX format QK maximum absolute error (relative): {qk_max_error.item():.6f}")
    # print(f"qk_quant in kernel {block_scale_type}: {qk_quant}")
    
    # Calculate QK after softmax
    qk_ref_softmax = torch.nn.functional.softmax(qk_ref, dim=-1)
    qk_quant_softmax = torch.nn.functional.softmax(qk_quant, dim=-1)
    
    # Calculate error after softmax
    softmax_mse_loss = torch.nn.functional.mse_loss(qk_quant_softmax, qk_ref_softmax)
    softmax_max_error = torch.max(torch.abs(qk_quant_softmax - qk_ref_softmax) / (qk_ref_softmax+1e-6))
    
    print(f"{block_scale_type} MX format MSE loss after softmax: {softmax_mse_loss.item():.6f}")
    print(f"{block_scale_type} MX format maximum absolute error after softmax (relative): {softmax_max_error.item():.6f}")
    # return k_fp42, k_scale2
    return reference

# multi-batch, multi-head
def test_batched_quant_mxfpx_block_scaled(q, k, block_scale_type="mxfp4", head_dim=256, batch_size=1, num_heads=1, seq_len=256, BLKQ=128):
    # Quantize to float4 (MX format)
    if block_scale_type == "mxfp4":
        q_fp4, q_scale, k_fp4, k_scale = quant_mxfp4(q, k, BLKQ=BLKQ)
        q_quant = q_fp4
        k_quant = k_fp4
    else:
        q_fp8, q_scale, k_fp8, k_scale = quant_mxfp8e5(q, k, BLKQ=BLKQ)
        q_quant = q_fp8
        k_quant = k_fp8

    q_quant = q_quant.reshape(batch_size, num_heads, seq_len, head_dim)
    k_quant = k_quant.reshape(batch_size, num_heads, seq_len, head_dim)
    q_scale = q_scale.reshape(batch_size, num_heads, seq_len//128, 4, 32, head_dim//32//4, 4).permute(0, 1, 2, 5, 4, 3, 6).contiguous()
    k_scale = k_scale.reshape(batch_size, num_heads, seq_len//128, 4, 32, head_dim//32//4, 4).permute(0, 1, 2, 5, 4, 3, 6).contiguous()
    
    q_packed, q_scale, k_packed, k_scale, configs, (reference, q_dequant, k_dequant) = \
        initialize_block_scaled_batched_from_tensor(q_quant, k_quant, q_scale, k_scale, block_scale_type=block_scale_type, compute_reference=True)
    
    output = block_scaled_batched_matmul(q_packed, q_scale, k_packed, k_scale, torch.float16, batch_size, num_heads, seq_len, seq_len, head_dim, configs)
    torch.testing.assert_close(reference, output.to(torch.float32), atol=1e-3, rtol=1e-3)
    print(f"✅ (pass {block_scale_type} block scaled)")
    
    q_dequant = q_dequant.reshape(q.shape)
    mse_loss = torch.nn.functional.mse_loss(q_dequant, q.float() * (head_dim ** -0.5))
    max_error = torch.max(torch.abs(q_dequant - q.float() * (head_dim ** -0.5)))
    
    print(f"{block_scale_type} MX format MSE loss after quantization: {mse_loss.item():.6f}")
    print(f"{block_scale_type} MX format maximum absolute error: {max_error.item():.6f}")

    qk_ref = torch.matmul(q.float(), k.float().transpose(-2, -1)) * (head_dim ** -0.5)
    
    qk_quant = output.view(qk_ref.shape)
    qk_mse_loss = torch.nn.functional.mse_loss(qk_quant, qk_ref)
    qk_max_error = torch.max(torch.abs(qk_quant - qk_ref) / (qk_ref+1e-6))

    print(f"{block_scale_type} MX format QK MSE loss after quantization: {qk_mse_loss.item():.6f}")
    print(f"{block_scale_type} MX format QK maximum absolute error (relative): {qk_max_error.item():.6f}")
    # print(f"qk_quant in kernel {block_scale_type}: {qk_quant}")
    
    # Calculate QK after softmax
    qk_ref_softmax = torch.nn.functional.softmax(qk_ref, dim=-1)
    qk_quant_softmax = torch.nn.functional.softmax(qk_quant, dim=-1)
    
    # Calculate error after softmax
    softmax_mse_loss = torch.nn.functional.mse_loss(qk_quant_softmax, qk_ref_softmax)
    softmax_max_error = torch.max(torch.abs(qk_quant_softmax - qk_ref_softmax) / (qk_ref_softmax+1e-6))
    
    print(f"{block_scale_type} MX format MSE loss after softmax: {softmax_mse_loss.item():.6f}")
    print(f"{block_scale_type} MX format maximum absolute error after softmax (relative): {softmax_max_error.item():.6f}")
    # return k_fp42, k_scale2
    return reference
    
if __name__ == "__main__":
    torch.manual_seed(42)
    batch_size = 2
    num_heads = 4
    seq_len = 256
    head_dim = 256
    BLKQ = 128
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')
    qk_ref = torch.matmul(q.float(), k.float().transpose(-2, -1)) * (head_dim ** -0.5)
    # print(f"qk_ref in fp32: {qk_ref}")
    # print("Testing per_block_int8:")
    # test_per_block_int8()
    # print("\nTesting quant_fpxint8:")
    # test_quant_fpxint8()
    print("\n*********** Testing test_quant_mxfp8e5: ***********")
    test_quant_mxfp8e5(q, k, batch_size = batch_size, num_heads = num_heads, seq_len = seq_len, head_dim=head_dim, BLKQ=BLKQ) 

    # print("\n*********** Testing quant_mxfp4: ***********")
    # test_quant_mxfp4(q, k, batch_size = batch_size, num_heads = num_heads, seq_len = seq_len, head_dim=head_dim, BLKQ=BLKQ)

    block_scale_type = "mxfp8"
    # print(f"\n*********** Testing quant_mxfp4_block_scaled {block_scale_type}: ***********")
    assert ("fp4" in block_scale_type and head_dim>=256 and head_dim%128==0) or ("fp8" in block_scale_type and head_dim>=128 and head_dim%128==0), \
        "FP4: head_dim must be >=256 and divisible by 128 \nFP8: head_dim must be >=128 and divisible by 128"
    # test_quant_mxfpx_block_scaled(q, k, block_scale_type=block_scale_type, head_dim=head_dim, batch_size=batch_size, \
    #     num_heads=num_heads, seq_len=seq_len, BLKQ=BLKQ)
    
    print(f"\n************ Testing batched_quant_mxfp4_block_scaled {block_scale_type}: ************")
    test_batched_quant_mxfpx_block_scaled(q, k, block_scale_type=block_scale_type, head_dim=head_dim, batch_size=batch_size, \
        num_heads=num_heads, seq_len=seq_len, BLKQ=BLKQ)

