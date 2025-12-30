
XAJ_PARAMS_BOUNDS = {
    "k": [0.0, 2.0],
    "b": [0.0, 2.0],
    "im": [0.0, 0.6],
    "um": [0.0, 300.0],
    "lm": [0.0, 400.0],
    "dm": [0.0, 400.0],
    "c": [0.0, 1.0],
    "sm": [0.0, 300.0],
    "ex": [0.1, 2.5],
    "ki": [0.0, 1.0],
    "kg": [0.0, 1.0],
    "a": [0.0, 1.0],
    "theta": [0.0, 1.0],
    "ci": [0.0, 1.0],
    "cg": [0.0, 1.0],
}

# @torch.jit.script
# def xaj_timestep_loop(
#     P: torch.Tensor,
#     PET: torch.Tensor,
#     wu: torch.Tensor,
#     wl: torch.Tensor,
#     wd: torch.Tensor,
#     s: torch.Tensor,
#     fr: torch.Tensor,
#     qi: torch.Tensor,
#     qg: torch.Tensor,
#     k: torch.Tensor,
#     b: torch.Tensor,
#     im: torch.Tensor,
#     um: torch.Tensor,
#     lm: torch.Tensor,
#     dm: torch.Tensor,
#     c: torch.Tensor,
#     sm: torch.Tensor,
#     ex: torch.Tensor,
#     ki: torch.Tensor,
#     kg: torch.Tensor,
#     a: torch.Tensor,
#     theta: torch.Tensor,
#     ci: torch.Tensor,
#     cg: torch.Tensor,
#     kernel_size: int,
#     nearzero: float,
# ) -> tuple[
#     torch.Tensor,
#     torch.Tensor,
#     torch.Tensor,
#     torch.Tensor,
#     torch.Tensor,
#     torch.Tensor,
#     torch.Tensor,
#     torch.Tensor,
#     torch.Tensor,
#     torch.Tensor,
#     torch.Tensor,
# ]:
#     """
#     XAJ 模型单步循环（JIT 版本，HF 源汇分配）

#     Parameters
#     ----------
#     P : torch.Tensor
#         降水 (T, B, E)
#     PET : torch.Tensor
#         潜在蒸散 (T, B, E)
#     wu, wl, wd : torch.Tensor
#         三层土壤初始含水 (B, E)
#     s, fr : torch.Tensor
#         自由水蓄水与产流面积状态 (B, E)
#     qi, qg : torch.Tensor
#         壤中流与基流线性水库初始出流 (B, E)
#     k, b, im, um, lm, dm, c, sm, ex, ki, kg, a, theta, ci, cg : torch.Tensor
#         参数，可为 (B, E) 或 (T, B, E)
#     kernel_size : int
#         单位线长度
#     nearzero : float
#         防止除零的小量

#     Returns
#     -------
#     tuple
#         q_out, et_out, rs_out, ri_out, rg_out, wu, wl, wd, s, fr, q_res
#         其中 q_res 为线性水库最终状态 (qi, qg)
#     """

#     n_steps = P.shape[0]
#     n_grid = P.shape[1]
#     nmul = P.shape[2]
#     device = P.device

#     zero = torch.tensor(0.0, dtype=torch.float32, device=device)
#     one = torch.tensor(1.0, dtype=torch.float32, device=device)

#     # 输出张量
#     q_out = torch.zeros((n_steps, n_grid, nmul), dtype=torch.float32, device=device)
#     et_out = torch.zeros_like(q_out)
#     rs_out = torch.zeros_like(q_out)
#     ri_out = torch.zeros_like(q_out)
#     rg_out = torch.zeros_like(q_out)

#     # 参数动态标记
#     def _is_dyn(x: torch.Tensor) -> bool:
#         return x.dim() == 3

#     dyn_flags = {
#         "k": _is_dyn(k),
#         "b": _is_dyn(b),
#         "im": _is_dyn(im),
#         "um": _is_dyn(um),
#         "lm": _is_dyn(lm),
#         "dm": _is_dyn(dm),
#         "c": _is_dyn(c),
#         "sm": _is_dyn(sm),
#         "ex": _is_dyn(ex),
#         "ki": _is_dyn(ki),
#         "kg": _is_dyn(kg),
#         "a": _is_dyn(a),
#         "theta": _is_dyn(theta),
#         "ci": _is_dyn(ci),
#         "cg": _is_dyn(cg),
#     }

#     # 单位线权重（简单指数核，按 batch 广播）
#     uh = torch.zeros((kernel_size, n_grid, nmul), dtype=torch.float32, device=device)
#     theta_now = theta if not dyn_flags["theta"] else theta[0]
#     a_now = a if not dyn_flags["a"] else a[0]
#     for i in range(kernel_size):
#         uh[i] = a_now * torch.pow(torch.clamp(one - theta_now, min=nearzero), i)
#     uh_sum = torch.clamp(uh.sum(0, keepdim=True), min=nearzero)
#     uh = uh / uh_sum
#     surf_buf = torch.zeros_like(uh)

#     for t in range(n_steps):
#         def pick(name: str, tensor: torch.Tensor):
#             return tensor[t] if dyn_flags[name] else tensor

#         k_t = pick("k", k)
#         b_t = pick("b", b)
#         im_t = pick("im", im)
#         um_t = pick("um", um)
#         lm_t = pick("lm", lm)
#         dm_t = pick("dm", dm)
#         c_t = pick("c", c)
#         sm_t = pick("sm", sm)
#         ex_t = pick("ex", ex)
#         ki_t = pick("ki", ki)
#         kg_t = pick("kg", kg)
#         a_t = pick("a", a)
#         theta_t = pick("theta", theta)
#         ci_t = pick("ci", ci)
#         cg_t = pick("cg", cg)

#         # degree to hydrological inputs
#         prcp = torch.clamp(P[t], min=zero)
#         pet = torch.clamp(PET[t] * k_t, min=zero)

#         wm = um_t + lm_t + dm_t
#         wu_prev = wu
#         wl_prev = wl
#         wd_prev = wd
#         w0 = wu_prev + wl_prev + wd_prev

#         # evaporation three-layer
#         eu = torch.where(wu + prcp >= pet, pet, wu + prcp)
#         ed = torch.where(
#             (wl < c_t * lm_t) & (wl < c_t * (pet - eu)),
#             c_t * (pet - eu) - wl,
#             zero,
#         )
#         el = torch.where(
#             wu + prcp >= pet,
#             zero,
#             torch.where(
#                 wl >= c_t * lm_t,
#                 (pet - eu) * wl / torch.clamp(lm_t, min=nearzero),
#                 torch.where(wl >= c_t * (pet - eu), c_t * (pet - eu), wl),
#             ),
#         )
#         e = eu + el + ed
#         prcp_diff = prcp - e
#         pe = torch.clamp(prcp_diff, min=zero)

#         # runoff generation
#         wmm = wm * (one + b_t)
#         a_coef = wmm * (
#             one
#             - torch.pow(
#                 torch.clamp(one - torch.clamp(w0 / torch.clamp(wm, min=nearzero), max=one), min=zero),
#                 one / (one + b_t),
#             )
#         )

#         inner_term = one - torch.clamp((a_coef + pe) / torch.clamp(wmm, min=nearzero), max=one)
#         r_cal = torch.where(
#             pe > zero,
#             torch.where(
#                 pe + a_coef < wmm,
#                 pe - (wm - w0) + wm * torch.pow(torch.clamp(inner_term, min=zero), one + b_t),
#                 pe - (wm - w0),
#             ),
#             zero,
#         )
#         r = torch.clamp(r_cal, min=zero)
#         rim = torch.clamp(pe * im_t, min=zero)

#         # update soil moisture storages
#         wu = torch.where(
#             pe > zero,
#             torch.where(wu_prev + pe - r < um_t, wu_prev + pe - r, um_t),
#             torch.where(wu_prev + pe > zero, wu_prev + pe, zero),
#         )
#         wd_new = torch.where(
#             pe > zero,
#             torch.where(
#                 wu_prev + wl_prev + pe - r > um_t + lm_t,
#                 wu_prev + wl_prev + wd_prev + pe - r - um_t - lm_t,
#                 wd_prev,
#             ),
#             wd_prev - ed,
#         )
#         wd = torch.clamp(wd_new, min=zero)
#         wl = torch.where(
#             pe > zero,
#             wu_prev + wl_prev + wd_prev + pe - r - wu - wd,
#             wl_prev - el,
#         )
#         wl = torch.clamp(wl, min=zero, max=lm_t)
#         wd = torch.clamp(wd, min=zero, max=dm_t)

#         # source separation (HF)
#         ms = sm_t * (one + ex_t)
#         fr_new = torch.where(r > zero, r / torch.clamp(pe, min=nearzero), fr)
#         fr_new = torch.clamp(fr_new, min=nearzero)

#         ss = torch.where(fr_new > zero, fr * s / torch.clamp(fr_new, min=nearzero), s)
#         ss = torch.clamp(ss, max=sm_t - nearzero)
#         au = ms * (
#             one
#             - torch.pow(
#                 torch.clamp(one - ss / torch.clamp(sm_t, min=nearzero), min=zero),
#                 one / (one + ex_t),
#             )
#         )

#         rs = torch.where(
#             pe + au < ms,
#             fr_new * (
#                 pe
#                 - sm_t
#                 + ss
#                 + sm_t
#                 * torch.pow(
#                     torch.clamp(
#                         one
#                         - torch.clamp(pe + au, max=ms) / torch.clamp(ms, min=nearzero),
#                         min=zero,
#                     ),
#                     one + ex_t,
#                 )
#             ),
#             fr_new * (pe + ss - sm_t),
#         )
#         rs = torch.clamp(rs, min=zero, max=r)
#         s_tmp = ss + (r - rs) / torch.clamp(fr_new, min=nearzero)
#         s_tmp = torch.clamp(s_tmp, max=sm_t)

#         ri = ki_t * s_tmp * fr_new
#         rg = kg_t * s_tmp * fr_new
#         s = torch.clamp(s_tmp * (one - ki_t - kg_t), min=nearzero)
#         fr = fr_new

#         # impervious correction
#         rss = rs * (one - im_t)
#         ris = ri * (one - im_t)
#         rgs = rg * (one - im_t)

#         # update UH kernel when parameters are dynamic
#         if dyn_flags["a"] or dyn_flags["theta"]:
#             for i in range(kernel_size):
#                 uh[i] = a_t * torch.pow(torch.clamp(one - theta_t, min=nearzero), i)
#             uh_sum = torch.clamp(uh.sum(0, keepdim=True), min=nearzero)
#             uh = uh / uh_sum

#         # surface convolution buffer
#         surf_buf = torch.roll(surf_buf, shifts=1, dims=0)
#         surf_buf[0] = torch.clamp(rss + rim, min=zero)
#         qs_fast = (surf_buf * uh).sum(0)

#         # linear reservoirs
#         qi = ci_t * qi + (one - ci_t) * ris
#         qg = cg_t * qg + (one - cg_t) * rgs

#         q = qs_fast + qi + qg

#         q_out[t] = q
#         et_out[t] = e
#         rs_out[t] = rs
#         ri_out[t] = ri
#         rg_out[t] = rg

#     return (
#         q_out,
#         et_out,
#         rs_out,
#         ri_out,
#         rg_out,
#         wu,
#         wl,
#         wd,
#         s,
#         fr,
#         torch.stack([qi, qg], dim=0),
#     )


