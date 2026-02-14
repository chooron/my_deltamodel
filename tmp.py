import pandas as pd
import numpy as np
import itertools

# ==========================================
# 1. 基础数据库准备 (保持不变)
# ==========================================
borides_lib = {
    'WB2':  {'metal': 'W',  'm_mass': 183.84, 'molar_mass': 205.46, 'density': 10.77},
    'ZrB2': {'metal': 'Zr', 'm_mass': 91.22,  'molar_mass': 112.84, 'density': 6.10},
    'NbB2': {'metal': 'Nb', 'm_mass': 92.91,  'molar_mass': 114.53, 'density': 6.97},
    'TiB2': {'metal': 'Ti', 'm_mass': 47.87,  'molar_mass': 69.49,  'density': 4.52},
    'CrB2': {'metal': 'Cr', 'm_mass': 52.00,  'molar_mass': 73.62,  'density': 5.20}
}

metals_lib = {
    'Cu': {'atomic_mass': 63.55, 'radius': 128},
    'W':  {'atomic_mass': 183.84, 'radius': 139},
    'Mo': {'atomic_mass': 95.96, 'radius': 139},
    'Nb': {'atomic_mass': 92.91, 'radius': 146},
    'Zr': {'atomic_mass': 91.22, 'radius': 160},
    'Ti': {'atomic_mass': 47.87, 'radius': 147},
    'Ta': {'atomic_mass': 180.95, 'radius': 146},
    'Cr': {'atomic_mass': 52.00, 'radius': 128},
    'Hf': {'atomic_mass': 178.49, 'radius': 159},
    'Sc': {'atomic_mass': 44.96, 'radius': 162},
    'V':  {'atomic_mass': 50.94, 'radius': 134},
    'Ce': {'atomic_mass': 140.12, 'radius': 182},
    'La': {'atomic_mass': 138.91, 'radius': 187}
}

for boride, data in borides_lib.items():
    metal_symbol = data['metal']
    if metal_symbol in metals_lib:
        borides_lib[boride]['radius'] = metals_lib[metal_symbol]['radius']

CU_density = 8.96 

miedema_data = {
    'Cu': [0, -38.34, 79.94, 67.09, 9.18, -78.1, -32.83, 6.58, 48.94, -58.43, -82.57, 18.97, -66.21, -63.03],
    'B': [-32.74, 0, -135.06, -147.86, -213.65, -258.63, -227.44, -211.6, -181.48, -242.62, -196.53, -181.93, -162.9, -157.34],
    'W': [101.06, -205.16, 0, -0.9, -32.9, -33.55, -22.83, -29.08, 4.25, -24.06, 33.99, -3.42, 95.44, 105.29],
    'Mo': [82.54, -128.66, -0.88, 0, -22.01, -22.33, -14.05, -19.15, 1.69, -14.52, 38.69, 0.04, 94.83, 103.89],
    'Nb': [11.6, -325, -33.63, -23.01, 0, 14.98, 8.09, 0.11, -32.29, 14.75, 65.99, -4.52, 112.55, 120.21],
    'Zr': [-109.98, -438.96, -38.86, -26.46, 16.98, 0, -0.97, 11.78, -57.66, -0.81, 15.77, -17.06, 40.38, 44.95],
    'Ti': [-39.83, -332.33, -22.57, -14.21, 7.82, -0.83, 0, 5.5, -32.73, 0.56, 27.31, -7.08, 59.06, 64.54],
    'Ta': [8.32, -322.08, -29.75, -20.03, 0.11, 10.4, 5.69, 0, -30.18, 10.96, 58.86, -4.4, 104.1, 111.57],
    'Cr': [50.55, -146.18, 3.48, 1.41, -25.93, -40.86, -27.18, -24.22, 0, -31.89, 2.33, -7.56, 45.3, 52.32],
    'Hf': [-81.41, -407.31, -27.46, -16.95, 16.47, -0.8, 0.64, 12.23, -44.35, 0, 20.04, -10.16, 48.55, 53.68],
    'Sc': [-114.81, -319.54, 38.61, 45.19, 75.24, 16.17, 32.47, 67.07, 3.24, 20.76, 0, 32.67, 5.78, 7.61],
    'V': [20.66, -238.34, -2.99, 0.04, -3.87, -12.89, -6.28, -3.77, -8.07, -7.79, 24.71, 0, 62.96, 69.31],
    'Ce': [-116.18, -334.12, 137.01, 140, 162.23, 52.37, 88.78, 149.96, 79.67, 63.59, 7.29, 105.23, 0, 0.16],
    'La': [-113.58, -331.44, 155.28, 157.55, 178.01, 59.88, 99.67, 165.12, 94.51, 72.24, 9.86, 119.01, 0.16, 0]
}
elements_order = ['Cu', 'B', 'W', 'Mo', 'Nb', 'Zr', 'Ti', 'Ta', 'Cr', 'Hf', 'Sc', 'V', 'Ce', 'La']
miedema_df = pd.DataFrame(miedema_data, index=elements_order, columns=elements_order)

# ==========================================
# 2. 核心计算与验证函数
# ==========================================
def evaluate_recipe(total_boride_wt_percent, boride_composition):
    """验证配方是否满足所有条件，如果不满足返回False，满足返回具体数据"""
    
    # --- 需求1: 计算密度 ---
    w_boride_total_frac = total_boride_wt_percent / 100.0
    w_cu_frac = 1.0 - w_boride_total_frac
    
    sum_wi_over_rho_i = 0
    for boride, wt_percent in boride_composition.items():
        w_i = wt_percent / 100.0
        rho_i = borides_lib[boride]['density']
        sum_wi_over_rho_i += w_i / rho_i

    rho_boride_mixture = 1.0 / sum_wi_over_rho_i
    rho_composite = 1.0 / (w_boride_total_frac / rho_boride_mixture + w_cu_frac / CU_density)
    
    # 【一票否决】密度不在 8.0 - 8.5 直接淘汰，提升搜索速度
    if not (8.0 <= rho_composite <= 8.5):
        return False, None

    # --- 需求2: 计算溶解焓 ---
    total_mass = 100.0
    m_metal_in_boride_total = 0
    metal_masses = {}
    total_boride_mass = total_mass * w_boride_total_frac
    
    for boride, wt_percent in boride_composition.items():
        mass_of_this_boride = total_boride_mass * (wt_percent / 100.0)
        info = borides_lib[boride]
        metal = info['metal']
        mass_of_metal_from_boride = mass_of_this_boride * (info['m_mass'] / info['molar_mass'])
        m_metal_in_boride_total += mass_of_metal_from_boride
        metal_masses[metal] = metal_masses.get(metal, 0) + mass_of_metal_from_boride

    m_cu = (total_mass / 2.0) - m_metal_in_boride_total
    metal_masses['Cu'] = m_cu

    # 转原子分数
    metal_moles = {m: mass / metals_lib[m]['atomic_mass'] for m, mass in metal_masses.items() if mass > 0}
    total_moles = sum(metal_moles.values())
    metal_atomic_fractions = {m: moles / total_moles for m, moles in metal_moles.items()}

    # 累加能量矩阵
    delta_H_mix = 0
    active_metals = list(metal_atomic_fractions.keys())
    
    for i in range(len(active_metals)):
        for j in range(len(active_metals)):
            if i != j:
                m1, m2 = active_metals[i], active_metals[j]
                avg_h = (miedema_df.loc[m1, m2] + miedema_df.loc[m2, m1]) / 2.0
                c1, c2 = metal_atomic_fractions[m1], metal_atomic_fractions[m2]
                delta_H_mix += 4 * avg_h * c1 * c2
                
    # 【一票否决】溶解焓 >= 0 直接淘汰
    if delta_H_mix >= 0:
        return False, None

    # --- 需求3: 计算相对原子尺寸 ---
    r_bar = sum(c_i * metals_lib[m]['radius'] for m, c_i in metal_atomic_fractions.items())
    sum_term_for_delta = sum(c_i * (1 - metals_lib[m]['radius'] / r_bar)**2 for m, c_i in metal_atomic_fractions.items())
    delta = np.sqrt(sum_term_for_delta) * 100  # 转化为百分比
    
    # 【一票否决】尺寸畸变 > 4% 直接淘汰
    if delta > 4.0:
        return False, None

    # 全部通过，返回结果
    metrics = {
        'density': rho_composite,
        'enthalpy': delta_H_mix,
        'delta': delta
    }
    return True, metrics

# ==========================================
# 3. 辅助生成配方组合的函数
# ==========================================
def generate_proportions(num_components, step=10):
    """递归生成所有总和为100的比例分配（例如：生成 10,90; 20,80 等）"""
    if num_components == 1:
        yield [100.0]
        return
    
    def get_combos(n, target):
        if n == 1:
            yield [float(target)]
            return
        for i in range(step, int(target) - step * (n - 2), step):
            for rest in get_combos(n - 1, target - i):
                yield [float(i)] + rest
                
    yield from get_combos(num_components, 100)

# ==========================================
# 4. 主搜索程序
# ==========================================
if __name__ == "__main__":
    print("正在全面搜索所有可能的配方组合，请稍等...")
    
    available_borides = list(borides_lib.keys())
    valid_recipes = []
    
    # 搜索范围设定：
    # 总添加量 (Wt%): 1% 到 30%，步长 1%
    total_wt_range = range(1, 31) 
    
    # 考虑单种硼化物、2种混合、3种混合
    for num_mix in [1, 2, 3]: 
        for boride_combo in itertools.combinations(available_borides, num_mix):
            # 内部比例步长 10% (比如 10:90, 20:80)
            for proportions in generate_proportions(num_mix, step=10): 
                composition = dict(zip(boride_combo, proportions))
                
                # 遍历总质量分数
                for total_wt in total_wt_range:
                    success, metrics = evaluate_recipe(total_wt, composition)
                    if success:
                        valid_recipes.append({
                            'Total_Wt%': total_wt,
                            'Composition': composition,
                            'Density': metrics['density'],
                            'Enthalpy': metrics['enthalpy'],
                            'Delta': metrics['delta']
                        })

    # ==========================================
    # 5. 输出格式化报告
    # ==========================================
    if not valid_recipes:
        print("\n未找到满足所有条件（密度8.0-8.5，焓<0，尺寸畸变<=4%）的配方！")
        print("建议：放宽参数要求。")
    else:
        # 按热力学最稳定（溶解焓最小/最负）进行排序
        valid_recipes.sort(key=lambda x: x['Enthalpy'])
        
        print(f"\n搜索完成！共找到 {len(valid_recipes)} 个符合所有条件的黄金配方。")
        print("以下按【热力学稳定性 (溶解焓越负越好)】为您推荐前 20 个最佳配方：")
        print("="*95)
        print(f"{'配方组成 (硼化物 & 相对比例)':<45} | {'总添加量(wt%)':<12} | {'密度(g/cm3)':<10} | {'溶解焓(kJ/mol)':<12} | {'尺寸差(δ%)':<10}")
        print("-" * 95)
        
        # 只打印前20个，避免刷屏
        for recipe in valid_recipes[:20]:
            comp_str = " + ".join([f"{k}({v}%)" for k, v in recipe['Composition'].items()])
            print(f"{comp_str:<45} | {recipe['Total_Wt%']:<14} | {recipe['Density']:<11.3f} | {recipe['Enthalpy']:<14.2f} | {recipe['Delta']:<10.2f}")
        
        print("="*95)
        print("提示：'配方组成'括号里的百分比是硼化物之间的相对比例。")
        print("例如：TiB2(50.0%) + CrB2(50.0%)，总添加量 12%，意味着在整个铜合金里：TiB2占6%，CrB2占6%，铜占88%。")