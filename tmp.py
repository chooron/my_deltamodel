import sys
from pathlib import Path

def delete_npy_files():
    # ================= 配置区域 =================
    # 目标文件夹路径 (使用 r 前缀防止转义字符问题)
    target_path = r"/workspace/my_deltamodel/project/diff_compare/output/camels_559/train1989-1998/no_multi/Calibrate_E50_R365_B100_n16_noLn_noWU_42"
    
    # 模拟运行开关：
    # True  = 只打印要删除的文件，不执行删除（推荐首次运行使用）
    # False = 真正执行删除操作
    DRY_RUN = False 
    # ===========================================

    root_dir = Path(target_path)

    # 检查路径是否存在
    if not root_dir.exists():
        print(f"错误：找不到路径 -> {target_path}")
        return

    print(f"正在扫描目录: {root_dir}")
    print(f"模式: {'模拟运行 (不删除)' if DRY_RUN else '执行删除'}")
    print("-" * 50)

    count = 0
    # rglob('*') 递归查找所有文件
    # .suffix 检查文件后缀
    files_to_process = list(root_dir.rglob("*.npy"))

    if not files_to_process:
        print("未找到任何 .npy 文件。")
        return

    for file_path in files_to_process:
        try:
            if DRY_RUN:
                print(f"[将删除] {file_path}")
            else:
                file_path.unlink() # 执行删除
                print(f"[已删除] {file_path}")
            
            count += 1
        except Exception as e:
            print(f"[出错] 无法删除 {file_path}: {e}")

    print("-" * 50)
    print(f"处理完成。共{'发现' if DRY_RUN else '删除'} {count} 个 .npy 文件。")

if __name__ == "__main__":
    delete_npy_files()