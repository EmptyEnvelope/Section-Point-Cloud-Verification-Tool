import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox
import laspy

# ================= 核心算法部分 =================

def process_chunk(sorted_pts, global_N, immunity_count=10):
    """
    核心处理引擎：基于栈和前瞻机制判断异常点。
    """
    N_chunk = len(sorted_pts)
    valid_stack = []
    anomaly_orig_indices = []

    for i in range(N_chunk):
        curr_pt = sorted_pts[i]
        
        while True:
            if not valid_stack:
                valid_stack.append(curr_pt)
                break
                
            prev_pt = valid_stack[-1]
            
            dt = curr_pt[3] - prev_pt[3]
            dz_abs = abs(curr_pt[2] - prev_pt[2])
            
            # 使用了你调整过的参数
            threshold_dz = 0.15 if dt <= 0.075 else 0.20
            
            if dz_abs > threshold_dz:
                # 前瞻趋势验证 (Look-ahead)，使用了你调整的参数 10
                look_ahead_k = 10
                future_zs = [sorted_pts[j][2] for j in range(i + 1, min(i + 1 + look_ahead_k, N_chunk))]
                
                median_future_z = np.median(future_zs) if future_zs else curr_pt[2]
                margin = threshold_dz * 0.5 

                curr_global_i = int(curr_pt[5])
                prev_global_i = int(prev_pt[5])

                if curr_pt[2] > prev_pt[2]:
                    # 【当前点突高】
                    if median_future_z >= curr_pt[2] - margin:
                        valid_stack.append(curr_pt)
                        break
                    else:
                        if curr_global_i < immunity_count or curr_global_i >= global_N - immunity_count: 
                            valid_stack.append(curr_pt)
                        else:
                            anomaly_orig_indices.append(int(curr_pt[4]))
                        break
                else:
                    # 【当前点突低】
                    if median_future_z <= curr_pt[2] + margin:
                        if prev_global_i < immunity_count or prev_global_i >= global_N - immunity_count:
                            valid_stack.append(curr_pt)
                            break
                        else:
                            popped_pt = valid_stack.pop()
                            anomaly_orig_indices.append(int(popped_pt[4]))
                            continue
                    else:
                        if curr_global_i < immunity_count or curr_global_i >= global_N - immunity_count:
                            valid_stack.append(curr_pt)
                        else:
                            anomaly_orig_indices.append(int(curr_pt[4]))
                        break
            else:
                valid_stack.append(curr_pt)
                break

    return anomaly_orig_indices


def detect_anomaly_indices(points, log_callback=None):
    """外层包裹：负责合轴、全局扫描、以及【循环迭代式】孤岛重算"""
    N = len(points)
    if N <= 40:
        return np.array([], dtype=int)
    
    # 1. 自动获取合并轴方向 (PCA)
    xy_data = points[:, :2]
    xy_centered = xy_data - np.mean(xy_data, axis=0)
    cov_matrix = np.cov(xy_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    main_direction = eigenvectors[:, np.argmax(eigenvalues)]
    
    if log_callback:
        log_callback(f"    -> 主方向向量: ({main_direction[0]:.2f}, {main_direction[1]:.2f})")
    
    # 投影合轴
    t_values = np.dot(xy_data, main_direction)
    
    extended_points = np.zeros((N, 5))
    extended_points[:, :3] = points
    extended_points[:, 3] = t_values
    extended_points[:, 4] = np.arange(N)
    
    # 按合轴方向排序
    sorted_points = extended_points[extended_points[:, 3].argsort()]
    
    sorted_points_with_idx = np.zeros((N, 6))
    sorted_points_with_idx[:, :5] = sorted_points
    sorted_points_with_idx[:, 5] = np.arange(N)
    
    # ================= 阶段 1：全局初步扫描 =================
    global_anomalies = process_chunk(sorted_points_with_idx, global_N=N, immunity_count=20)
    
    # ================= 阶段 2：孤岛检测与拯救 (无限循环直至收敛) =================
    orig_idx_to_global_sorted_idx = {int(pt[4]): int(pt[5]) for pt in sorted_points_with_idx}
    final_anomalies = set(global_anomalies)
    
    iteration = 1
    total_rescued_all_iters = 0
    
    while True:
        # 1. 获取当前剩下的所有异常点在全局排序中的序号，并升序排列
        current_anomaly_sorted_indices = sorted([orig_idx_to_global_sorted_idx[idx] for idx in final_anomalies])
        
        islands = []
        current_island = []
        
        # 2. 聚类寻找连续断层的点云块
        for idx in current_anomaly_sorted_indices:
            if not current_island:
                current_island.append(idx)
            else:
                # 严格连续性检测
                if idx == current_island[-1] + 1:
                    current_island.append(idx)
                else:
                    # 断开了，结算上一个孤岛
                    if len(current_island) >= 3:
                        islands.append(current_island)
                    current_island = [idx]
                    
        # 结算最后一个孤岛
        if len(current_island) >= 3:
            islands.append(current_island)
            
        # 如果当前这一轮找不到任何的孤岛了，直接跳出循环！
        if not islands:
            break
            
        rescued_in_this_iteration = 0
        
        # 3. 对本轮找到的所有孤岛进行拯救
        for island_indices in islands:
            island_pts = [sorted_points_with_idx[idx] for idx in island_indices]
            
            # 重新验证 (关闭豁免区)
            island_true_anomalies = process_chunk(island_pts, global_N=N, immunity_count=0)
            
            # 计算被拯救的点
            island_orig_indices = {int(pt[4]) for pt in island_pts}
            rescued_points = island_orig_indices - set(island_true_anomalies)
            
            # 从全局异常库中剔除
            for rp in rescued_points:
                final_anomalies.remove(rp)
                
            rescued_in_this_iteration += len(rescued_points)
            
        # 如果这整整一轮都没有救回任何一个点（说明剩下的连续点是真的异常连片），跳出死循环
        if rescued_in_this_iteration == 0:
            break
            
        total_rescued_all_iters += rescued_in_this_iteration
        if log_callback:
            log_callback(f"    -> [循环第{iteration}轮]")
        
        iteration += 1

    if total_rescued_all_iters > 0 and log_callback:
        log_callback(f"    -> 累计经过 {iteration} 轮循环")

    return np.array(list(final_anomalies), dtype=int)


# ================= UI 界面与文件处理部分 =================
class PointCloudApp:
    def __init__(self, root):
        self.root = root
        self.root.title("LAS点云异常过滤")
        self.root.geometry("650x500")
        
        self.selected_files = []
        
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)
        
        self.btn_select = tk.Button(btn_frame, text="选择 LAS 文件 (可多选)", command=self.select_files, width=22)
        self.btn_select.grid(row=0, column=0, padx=10)
        
        self.btn_process = tk.Button(btn_frame, text="开始处理", command=self.process_files, width=15, state=tk.DISABLED)
        self.btn_process.grid(row=0, column=1, padx=10)
        
        tk.Label(root, text="已选择的文件:").pack(anchor="w", padx=20)
        
        self.listbox = tk.Listbox(root, selectmode=tk.EXTENDED, width=85, height=8)
        self.listbox.pack(pady=5, padx=20)
        
        tk.Label(root, text="处理日志:").pack(anchor="w", padx=20)
        
        self.log_text = tk.Text(root, width=85, height=13, state=tk.DISABLED)
        self.log_text.pack(pady=5, padx=20)
        
    def log(self, message):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        self.root.update()

    def select_files(self):
        files = filedialog.askopenfilenames(
            title="选择点云数据文件",
            filetypes=[("LAS/LAZ Files", "*.las *.laz"), ("All Files", "*.*")]
        )
        if files:
            self.selected_files = list(files)
            self.listbox.delete(0, tk.END)
            for f in self.selected_files:
                self.listbox.insert(tk.END, f)
            
            self.btn_process.config(state=tk.NORMAL)
            self.log(f"成功加载 {len(self.selected_files)} 个文件。")

    def process_files(self):
        if not self.selected_files:
            return
            
        self.btn_process.config(state=tk.DISABLED)
        self.btn_select.config(state=tk.DISABLED)
        
        for file_path in self.selected_files:
            filename = os.path.basename(file_path)
            self.log(f"\n正在读取: {filename} ...")
            try:
                las = laspy.read(file_path)
                points = np.vstack((las.x, las.y, las.z)).transpose()
                total_points = len(points)
                
                self.log(f"-> 包含 {total_points} 个点，启动多段式异常检测...")
                
                anomaly_indices = detect_anomaly_indices(points, log_callback=self.log)
                num_anomalies = len(anomaly_indices)
                
                if num_anomalies > 0:
                    las.classification[anomaly_indices] = 1
                
                # 获取原文件的目录
                dir_name = os.path.dirname(file_path)
                base_name, ext = os.path.splitext(filename)
                
                # ====== 新增修改：创建 process 文件夹 ======
                process_dir = os.path.join(dir_name, "process")
                os.makedirs(process_dir, exist_ok=True)  # 如果文件夹已存在则不报错，直接使用
                
                # 在 process 文件夹下构建输出路径
                out_path = os.path.join(process_dir, f"{base_name}_processed{ext}")
                # ============================================
                
                las.write(out_path)
                
                self.log(f"✓ 完成: 最终定位 {num_anomalies} 个异常点(修改为Class 1)。")
                self.log(f"-> 已保存至: process/{os.path.basename(out_path)}")
                
            except Exception as e:
                self.log(f"✗ 处理 {filename} 时出错: {str(e)}")
                
        self.log("-" * 45)
        self.log("所有文件处理完毕！")
        
        self.btn_process.config(state=tk.NORMAL)
        self.btn_select.config(state=tk.NORMAL)
        messagebox.showinfo("完成", "批量处理已完成，分类已修改！")

if __name__ == "__main__":
    root = tk.Tk()
    app = PointCloudApp(root)
    root.mainloop()