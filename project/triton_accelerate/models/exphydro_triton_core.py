import torch
import triton
import triton.language as tl
from typing import Tuple, Optional

# ==========================================
# Triton 前向传播 Kernel (ExpHydro)
# ==========================================

@triton.jit
def exphydro_forward_kernel(
    # --- 输入指针 ---
    p_ptr, t_ptr, lday_ptr,
    snow_in_ptr, soil_in_ptr,
    # --- 参数指针 ---
    tmin_ptr, tmax_ptr, df_ptr, smax_ptr, qmax_ptr, f_ptr,
    # --- 输出指针 ---
    q_out_ptr, snow_out_ptr, soil_out_ptr,
    # --- 配置 ---
    nearzero: tl.constexpr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # 1. Load Data
    p = tl.load(p_ptr + offsets, mask=mask, other=0.0)
    t_val = tl.load(t_ptr + offsets, mask=mask, other=0.0)
    lday = tl.load(lday_ptr + offsets, mask=mask, other=0.0)
    
    snow = tl.load(snow_in_ptr + offsets, mask=mask, other=0.0)
    soil = tl.load(soil_in_ptr + offsets, mask=mask, other=0.0)
    
    tmin = tl.load(tmin_ptr + offsets, mask=mask, other=0.0)
    tmax = tl.load(tmax_ptr + offsets, mask=mask, other=0.0)
    df = tl.load(df_ptr + offsets, mask=mask, other=0.0)
    smax = tl.load(smax_ptr + offsets, mask=mask, other=0.0)
    qmax = tl.load(qmax_ptr + offsets, mask=mask, other=0.0)
    f = tl.load(f_ptr + offsets, mask=mask, other=0.0)

    # ==========================
    # Bucket 1: Snow & Melt
    # ==========================
    
    # 降水分离: If T < Tmin -> Snow
    is_snow = t_val < tmin
    snowfall = tl.where(is_snow, p, 0.0)
    rainfall = tl.where(is_snow, 0.0, p)
    
    # 融雪: Melt = min(Snow, Df * (T - Tmax)) if T > Tmax
    temp_excess = t_val - tmax
    is_warm = temp_excess > 0.0
    
    pot_melt = df * tl.where(is_warm, temp_excess, 0.0)
    
    # 融雪受限于现有积雪
    # snow_mid = snow + snowfall (通常融雪发生在降雪之后，或者同时)
    # 这里为了简单，假设融雪作用于 (snow_prev + snowfall)
    snow_avail = snow + snowfall
    
    is_melt_limited = pot_melt < snow_avail
    melt = tl.where(is_melt_limited, pot_melt, snow_avail)
    
    # 更新积雪
    snow_new = snow_avail - melt
    
    # PET 计算 (Hamon 公式 + Melt)
    # es = 0.611 * exp(17.3 * T / (T + 237.3))
    es_exponent = (17.3 * t_val) / (t_val + 237.3)
    es = 0.611 * tl.exp(es_exponent)
    
    # term1 = 29.8 * Lday * 24 * es / (T + 273.2)
    hamon = (29.8 * 24.0 * lday * es) / (t_val + 273.2)
    pet = hamon + melt

    # ==========================
    # Bucket 2: Soil & Flow
    # ==========================
    
    # 蒸发: E = PET * min(1.0, Soil / Smax)
    # 这里的 Soil 是上一时刻的 Soil
    eps = 1e-6
    soil_ratio = soil / (smax + eps)
    # 限制 ratio 在 [0, 1]
    soil_ratio_sat = tl.minimum(tl.maximum(soil_ratio, 0.0), 1.0)
    
    evap = pet * soil_ratio_sat
    
    # 基流: Q_base = Qmax * exp(-f * max(0, Smax - Soil))
    deficit = tl.maximum(smax - soil, 0.0)
    baseflow = qmax * tl.exp(-f * deficit)
    
    # 地表径流: Q_surf = max(0, Soil - Smax)
    surfaceflow = tl.maximum(soil - smax, 0.0)
    
    # 总径流
    flow = baseflow + surfaceflow
    
    # 更新土壤水
    # dS = (Rain + Melt) - (Evap + Flow)
    soil_new = soil + (rainfall + melt) - (evap + flow)
    soil_new = tl.maximum(soil_new, nearzero) # 物理约束

    # Store Outputs
    tl.store(q_out_ptr + offsets, flow, mask=mask)
    tl.store(snow_out_ptr + offsets, snow_new, mask=mask)
    tl.store(soil_out_ptr + offsets, soil_new, mask=mask)


# ==========================================
# Triton 反向传播 Kernel (手动梯度推导)
# ==========================================

@triton.jit
def exphydro_backward_kernel(
    # --- Forward Inputs (Recompute needed) ---
    p_ptr, t_ptr, lday_ptr,
    snow_prev_ptr, soil_prev_ptr,
    # --- Parameters ---
    tmin_ptr, tmax_ptr, df_ptr, smax_ptr, qmax_ptr, f_ptr,
    # --- Gradients In (from Next Step & Loss) ---
    d_q_total_ptr, d_snow_next_ptr, d_soil_next_ptr,
    # --- Gradients Out ---
    d_snow_prev_ptr, d_soil_prev_ptr,
    d_tmin_ptr, d_tmax_ptr, d_df_ptr, d_smax_ptr, d_qmax_ptr, d_f_ptr,
    # --- Config ---
    nearzero: tl.constexpr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    eps = 1e-6

    # 1. Load Inputs & Params
    p = tl.load(p_ptr + offsets, mask=mask, other=0.0)
    t_val = tl.load(t_ptr + offsets, mask=mask, other=0.0)
    lday = tl.load(lday_ptr + offsets, mask=mask, other=0.0)
    
    snow_prev = tl.load(snow_prev_ptr + offsets, mask=mask, other=0.0)
    soil_prev = tl.load(soil_prev_ptr + offsets, mask=mask, other=0.0)
    
    tmin = tl.load(tmin_ptr + offsets, mask=mask, other=0.0)
    tmax = tl.load(tmax_ptr + offsets, mask=mask, other=0.0)
    df = tl.load(df_ptr + offsets, mask=mask, other=0.0)
    smax = tl.load(smax_ptr + offsets, mask=mask, other=0.0)
    qmax = tl.load(qmax_ptr + offsets, mask=mask, other=0.0)
    f = tl.load(f_ptr + offsets, mask=mask, other=0.0)

    # 2. Load Incoming Gradients
    d_q = tl.load(d_q_total_ptr + offsets, mask=mask, other=0.0)
    d_snow_out = tl.load(d_snow_next_ptr + offsets, mask=mask, other=0.0)
    d_soil_out = tl.load(d_soil_next_ptr + offsets, mask=mask, other=0.0)

    # Init Param Grads accumulators
    d_tmin = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    d_tmax = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    d_df = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    d_smax = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    d_qmax = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    d_f = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # =========================================================
    # PHASE 1: RECOMPUTE FORWARD (为了获取 mask 和中间变量)
    # =========================================================
    
    # Partitioning
    is_snow = t_val < tmin
    # rain = p if not snow else 0
    # snow_in = p if snow else 0
    
    # Melt
    temp_excess = t_val - tmax
    is_warm = temp_excess > 0.0
    pot_melt = df * tl.where(is_warm, temp_excess, 0.0)
    
    snow_avail = snow_prev + tl.where(is_snow, p, 0.0)
    is_melt_limited = pot_melt < snow_avail
    melt = tl.where(is_melt_limited, pot_melt, snow_avail)
    
    # PET
    es_denom = t_val + 237.3
    es = 0.611 * tl.exp((17.3 * t_val) / es_denom)
    hamon = (29.8 * 24.0 * lday * es) / (t_val + 273.2)
    pet = hamon + melt # Note: melt is part of pet in this model definition

    # Soil Fluxes
    # Evap
    soil_ratio_raw = soil_prev / (smax + eps)
    soil_ratio_sat = tl.minimum(tl.maximum(soil_ratio_raw, 0.0), 1.0)
    evap = pet * soil_ratio_sat
    
    # Baseflow
    deficit_arg = smax - soil_prev
    is_deficit_pos = deficit_arg > 0.0
    deficit = tl.where(is_deficit_pos, deficit_arg, 0.0)
    baseflow = qmax * tl.exp(-f * deficit)
    
    # Surfaceflow
    surf_arg = soil_prev - smax
    is_surf_pos = surf_arg > 0.0
    # surfaceflow = max(0, soil - smax)
    
    # Soil Update Logic Check:
    # soil_new = soil + rain + melt - evap - baseflow - surfaceflow
    # 这里我们需要计算 soil_new 是否被 nearzero 截断，因为这会影响梯度反传
    soil_temp = soil_prev + tl.where(is_snow, 0.0, p) + melt - evap - baseflow - tl.where(is_surf_pos, surf_arg, 0.0)
    is_soil_clamped = soil_temp < nearzero

    # =========================================================
    # PHASE 2: BACKWARD GRADIENTS
    # =========================================================
    
    # 1. Soil State Gradients
    # dL/dSoil_new comes from next step.
    # If soil was clamped to nearzero, gradient is killed (or handled by mask)
    d_soil_curr = tl.where(is_soil_clamped, 0.0, d_soil_out)
    
    # Q = Baseflow + Surfaceflow
    d_baseflow = d_q - d_soil_curr # -d_soil because flow reduces soil
    d_surfaceflow = d_q - d_soil_curr
    
    # 2. Surface Flow Gradients
    # Q_surf = max(0, Soil - Smax)
    # Gradient passes only if Soil > Smax
    d_surf_arg = tl.where(is_surf_pos, d_surfaceflow, 0.0)
    
    d_soil_prev = d_surf_arg # Contribution from surf
    d_smax = d_smax - d_surf_arg
    
    # 3. Baseflow Gradients
    # Q_base = Qmax * exp(-f * deficit)
    # d_Qbase / d_Qmax = exp(...) = baseflow / Qmax
    d_qmax = d_qmax + d_baseflow * (baseflow / (qmax + eps))
    
    # d_Qbase / d_exponent = Q_base
    d_exponent = d_baseflow * baseflow
    
    # exponent = -f * deficit
    d_f = d_f + d_exponent * (-deficit)
    d_deficit = d_exponent * (-f)
    
    # deficit = max(0, Smax - Soil)
    d_def_arg = tl.where(is_deficit_pos, d_deficit, 0.0)
    d_smax = d_smax + d_def_arg
    d_soil_prev = d_soil_prev - d_def_arg # Contribution from baseflow
    
    # 4. Evap Gradients
    # E = PET * ratio
    # d_soil_new / d_Evap = -1
    d_evap = -d_soil_curr
    
    d_pet = d_evap * soil_ratio_sat
    d_ratio_sat = d_evap * pet
    
    # ratio_sat = clamp(soil/smax, 0, 1)
    mask_ratio = (soil_ratio_raw > 0.0) & (soil_ratio_raw < 1.0)
    d_ratio_raw = tl.where(mask_ratio, d_ratio_sat, 0.0)
    
    # ratio_raw = soil / smax
    d_soil_prev = d_soil_prev + d_ratio_raw * (1.0 / (smax + eps))
    d_smax = d_smax + d_ratio_raw * (-soil_prev / ((smax + eps)*(smax + eps)))
    
    # Direct contribution of soil_prev to soil_new (identity)
    d_soil_prev = d_soil_prev + d_soil_curr
    
    # 5. PET & Melt Interaction
    # pet = hamon + melt
    # melt contributes to:
    #   1. soil_new (d_soil_curr)
    #   2. pet (d_pet)
    #   3. snow_new (d_snow_out * -1)
    
    d_melt = d_soil_curr + d_pet - d_snow_out
    
    # 6. Melt Logic Gradients
    # melt = min(pot_melt, snow_avail)
    # is_melt_limited: pot_melt < snow_avail
    
    d_pot_melt = tl.where(is_melt_limited, d_melt, 0.0)
    d_snow_avail = tl.where(is_melt_limited, 0.0, d_melt)
    
    # pot_melt = Df * max(T-Tmax, 0)
    d_df_term = tl.where(is_warm, d_pot_melt, 0.0) # passes if warm
    
    d_df = d_df + d_df_term * temp_excess
    # d(temp_excess)/dtmax = -1
    d_tmax = d_tmax + d_df_term * df * (-1.0)
    
    # 7. Snow Update Gradients
    # snow_new = snow_avail - melt (melt gradient handled above)
    # snow_avail = snow_prev + snow_input
    # Direct contribution to snow_new
    d_snow_avail = d_snow_avail + d_snow_out
    
    d_snow_prev = d_snow_avail
    d_snow_input = d_snow_avail
    
    # 8. Precip Partitioning
    # if is_snow: snow_input = p, rain = 0
    # else:       snow_input = 0, rain = p
    
    # rain contributes to soil_new (d_soil_curr)
    d_rain = d_soil_curr
    
    # threshold gradients (Strict logic -> 0 gradient for Tmin)
    # d_p = d_rain (if rain) + d_snow_input (if snow)
    # d_p is not stored but good to verify logic
    
    # Tmin logic: strictly speaking gradient is 0 or dirac. 
    # In strict mode, we leave d_tmin as 0.
    
    # 9. Store Param Gradients (Accumulate? No, calculate per step output)
    # Note: Autograd expects grad w.r.t parameters.
    # Since param pointers are scalar-like (broadcasted), we store results.
    tl.store(d_snow_prev_ptr + offsets, d_snow_prev, mask=mask)
    tl.store(d_soil_prev_ptr + offsets, d_soil_prev, mask=mask)
    
    tl.store(d_tmin_ptr + offsets, d_tmin, mask=mask)
    tl.store(d_tmax_ptr + offsets, d_tmax, mask=mask)
    tl.store(d_df_ptr + offsets, d_df, mask=mask)
    tl.store(d_smax_ptr + offsets, d_smax, mask=mask)
    tl.store(d_qmax_ptr + offsets, d_qmax, mask=mask)
    tl.store(d_f_ptr + offsets, d_f, mask=mask)
    
class ExpHydroTritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, 
                p_seq, t_seq, lday_seq,
                snow_init, soil_init,
                tmin, tmax, df, smax, qmax, f,
                nearzero=1e-6):
        
        T_steps, Batch = p_seq.shape
        device = p_seq.device
        
        # 1. 预分配输出
        q_out = torch.empty((T_steps, Batch), device=device, dtype=torch.float32)
        
        # 2. 必须保存每一步的状态用于反向传播 (Checkpointing)
        # 为了高性能反向传播，我们需要 T 时刻的状态来计算 T-1 的梯度
        # 这里我们保存所有中间状态。如果显存不够，可以只存 init，在 backward 里重算
        # 这里为了演示清晰性，保存所有 step 的 snow/soil 状态
        
        # 状态列表 (包含 T+1 个状态，第0个是init)
        snow_states = torch.empty((T_steps + 1, Batch), device=device)
        soil_states = torch.empty((T_steps + 1, Batch), device=device)
        
        snow_states[0] = snow_init
        soil_states[0] = soil_init
        
        # 3. Forward Loop (Python Driver -> Triton Kernel)
        # 这里的循环开销相比于 PyTorch Autograd 的开销微乎其微
        grid = lambda meta: (triton.cdiv(Batch, meta['BLOCK_SIZE']),)
        
        for t in range(T_steps):
            exphydro_forward_kernel[grid](
                p_seq[t], t_seq[t], lday_seq[t],
                snow_states[t], soil_states[t],
                tmin, tmax, df, smax, qmax, f,
                q_out[t], snow_states[t+1], soil_states[t+1],
                nearzero,
                Batch,
                BLOCK_SIZE=1024
            )
            
        # 4. Save Context
        ctx.save_for_backward(p_seq, t_seq, lday_seq, snow_states, soil_states)
        ctx.params = (tmin, tmax, df, smax, qmax, f)
        ctx.nearzero = nearzero
        
        return q_out

    @staticmethod
    def backward(ctx, grad_q_out):
        # grad_q_out: [T, B]
        
        p_seq, t_seq, lday_seq, snow_states, soil_states = ctx.saved_tensors
        tmin, tmax, df, smax, qmax, f = ctx.params
        nearzero = ctx.nearzero
        
        T_steps, Batch = p_seq.shape
        
        # 1. Init Param Grads accumulators
        d_tmin = torch.zeros_like(tmin)
        d_tmax = torch.zeros_like(tmax)
        d_df = torch.zeros_like(df)
        d_smax = torch.zeros_like(smax)
        d_qmax = torch.zeros_like(qmax)
        d_f = torch.zeros_like(f)
        
        # Next step state gradients (initially 0 at T_end)
        d_snow_next = torch.zeros_like(snow_states[0])
        d_soil_next = torch.zeros_like(soil_states[0])
        
        # 临时梯度 buffer (用于从 kernel 读回梯度)
        # 为了减少显存分配，可以复用 buffer
        d_tmin_step = torch.empty_like(tmin)
        d_tmax_step = torch.empty_like(tmax)
        d_df_step = torch.empty_like(df)
        d_smax_step = torch.empty_like(smax)
        d_qmax_step = torch.empty_like(qmax)
        d_f_step = torch.empty_like(f)
        
        # 输入数据的梯度 (通常不需要，除非做气象数据归因)
        # d_p = torch.zeros_like(p_seq)
        
        grid = lambda meta: (triton.cdiv(Batch, meta['BLOCK_SIZE']),)
        
        # 2. Backward Loop (Reverse Time)
        for t in range(T_steps - 1, -1, -1):
            
            # 这里的 snow_states[t] 就是 snow_prev
            # d_snow_next 是从 t+1 时刻传回来的 dL/dSnow_new
            
            exphydro_backward_kernel[grid](
                p_seq[t], t_seq[t], lday_seq[t],
                snow_states[t], soil_states[t],
                tmin, tmax, df, smax, qmax, f,
                
                grad_q_out[t], d_snow_next, d_soil_next,
                
                d_snow_next, d_soil_next, # Update directly in place (prev becomes next for t-1)
                d_tmin_step, d_tmax_step, d_df_step, d_smax_step, d_qmax_step, d_f_step,
                
                nearzero,
                Batch,
                BLOCK_SIZE=1024
            )
            
            # Accumulate Parameter Grads
            d_tmin += d_tmin_step
            d_tmax += d_tmax_step
            d_df += d_df_step
            d_smax += d_smax_step
            d_qmax += d_qmax_step
            d_f += d_f_step
            
        # 返回所有输入的梯度
        # p, t, lday 不需要梯度 (None)
        # snow_init, soil_init 的梯度就是循环结束后的 d_snow_next, d_soil_next
        return (None, None, None, 
                d_snow_next, d_soil_next, 
                d_tmin, d_tmax, d_df, d_smax, d_qmax, d_f, 
                None)

# 封装调用函数
def exphydro_run_triton(p, t, lday, snow_init, soil_init, params, nearzero=1e-6):
    """
    ExpHydro Triton 入口
    params: dict of tensors (tmin, tmax, df, smax, qmax, f)
    """
    return ExpHydroTritonFunction.apply(
        p, t, lday,
        snow_init, soil_init,
        params['tmin'], params['tmax'], params['df'], 
        params['smax'], params['qmax'], params['f'],
        nearzero
    )