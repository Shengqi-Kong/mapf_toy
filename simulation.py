"""
MAPF 可视化仿真 GUI —— 基于 tkinter。

操作说明:
  1. 选择编辑模式（障碍物 / 起点 / 终点）
  2. 在网格上点击放置，右键取消
  3. 起点和终点按顺序配对（第 1 个起点对应第 1 个终点）
  4. 点击「运行」按钮查看路径规划动画
"""

import tkinter as tk
from tkinter import messagebox
from mapf_env import MAPFEnv
from cbs import cbs_search

# ========== 配置 ==========
DEFAULT_ROWS = 10
DEFAULT_COLS = 10
CELL_SIZE = 50
ANIM_DELAY_MS = 300

COLORS_AGENT = [
    "#e74c3c", "#3498db", "#2ecc71", "#f39c12",
    "#9b59b6", "#1abc9c", "#e67e22", "#34495e",
]


class Simulation:
    def __init__(self, master: tk.Tk):
        self.master = master
        master.title("MAPF 仿真器")
        master.resizable(False, False)

        self.rows = DEFAULT_ROWS
        self.cols = DEFAULT_COLS
        self.obstacles: set = set()
        self.starts: list = []
        self.goals: list = []
        self.mode = "obstacle"  # obstacle / start / goal
        self.paths = None
        self.anim_id = None
        self.timestep = 0
        self.max_timestep = 0

        self._build_ui()
        self._draw_grid()

    # ---------- UI 构建 ----------

    def _build_ui(self):
        # 顶部控制栏
        ctrl = tk.Frame(self.master)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=6, pady=4)

        # 网格尺寸
        tk.Label(ctrl, text="行:").pack(side=tk.LEFT)
        self.row_var = tk.IntVar(value=self.rows)
        tk.Spinbox(ctrl, from_=3, to=30, width=4, textvariable=self.row_var).pack(side=tk.LEFT)
        tk.Label(ctrl, text=" 列:").pack(side=tk.LEFT)
        self.col_var = tk.IntVar(value=self.cols)
        tk.Spinbox(ctrl, from_=3, to=30, width=4, textvariable=self.col_var).pack(side=tk.LEFT)
        tk.Button(ctrl, text="重置网格", command=self._reset_grid).pack(side=tk.LEFT, padx=6)

        # 分隔
        tk.Frame(ctrl, width=2, bd=1, relief=tk.SUNKEN).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        # 编辑模式
        self.mode_var = tk.StringVar(value="obstacle")
        for text, val in [("障碍物", "obstacle"), ("起点", "start"), ("终点", "goal")]:
            tk.Radiobutton(ctrl, text=text, variable=self.mode_var, value=val,
                           command=self._on_mode_change).pack(side=tk.LEFT, padx=2)

        # 分隔
        tk.Frame(ctrl, width=2, bd=1, relief=tk.SUNKEN).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        # 运行 / 停止 / 清除
        tk.Button(ctrl, text="▶ 运行", command=self._run, bg="#2ecc71", fg="white",
                  font=("", 10, "bold")).pack(side=tk.LEFT, padx=4)
        tk.Button(ctrl, text="■ 停止", command=self._stop).pack(side=tk.LEFT, padx=2)
        tk.Button(ctrl, text="清除路径", command=self._clear_paths).pack(side=tk.LEFT, padx=2)

        # 画布
        canvas_w = self.cols * CELL_SIZE
        canvas_h = self.rows * CELL_SIZE
        self.canvas = tk.Canvas(self.master, width=canvas_w, height=canvas_h, bg="white")
        self.canvas.pack(padx=6, pady=6)
        self.canvas.bind("<Button-1>", self._on_left_click)
        self.canvas.bind("<Button-3>", self._on_right_click)

        # 底部状态栏
        self.status_var = tk.StringVar(value="模式: 障碍物 | 左键放置, 右键取消")
        tk.Label(self.master, textvariable=self.status_var, anchor=tk.W,
                 relief=tk.SUNKEN, bd=1).pack(side=tk.BOTTOM, fill=tk.X)

    # ---------- 网格绘制 ----------

    def _draw_grid(self):
        self.canvas.delete("all")
        for r in range(self.rows):
            for c in range(self.cols):
                x1, y1 = c * CELL_SIZE, r * CELL_SIZE
                x2, y2 = x1 + CELL_SIZE, y1 + CELL_SIZE
                fill = "white"
                if (r, c) in self.obstacles:
                    fill = "#2c3e50"
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=fill,
                                             outline="#bdc3c7", width=1)

        # 起点
        for i, (r, c) in enumerate(self.starts):
            color = COLORS_AGENT[i % len(COLORS_AGENT)]
            cx, cy = c * CELL_SIZE + CELL_SIZE // 2, r * CELL_SIZE + CELL_SIZE // 2
            rad = CELL_SIZE // 3
            self.canvas.create_oval(cx - rad, cy - rad, cx + rad, cy + rad,
                                    fill=color, outline="white", width=2)
            self.canvas.create_text(cx, cy, text=f"S{i}", fill="white",
                                    font=("", 11, "bold"))

        # 终点
        for i, (r, c) in enumerate(self.goals):
            color = COLORS_AGENT[i % len(COLORS_AGENT)]
            cx, cy = c * CELL_SIZE + CELL_SIZE // 2, r * CELL_SIZE + CELL_SIZE // 2
            rad = CELL_SIZE // 3
            self.canvas.create_rectangle(cx - rad, cy - rad, cx + rad, cy + rad,
                                         fill="white", outline=color, width=3)
            self.canvas.create_text(cx, cy, text=f"G{i}", fill=color,
                                    font=("", 11, "bold"))

    def _draw_agents(self, timestep: int):
        """在当前网格上叠加绘制智能体位置。"""
        self._draw_grid()
        if not self.paths:
            return
        for i, path in enumerate(self.paths):
            t = min(timestep, len(path) - 1)
            r, c = path[t]
            color = COLORS_AGENT[i % len(COLORS_AGENT)]
            cx, cy = c * CELL_SIZE + CELL_SIZE // 2, r * CELL_SIZE + CELL_SIZE // 2
            rad = CELL_SIZE // 2.5
            self.canvas.create_oval(cx - rad, cy - rad, cx + rad, cy + rad,
                                    fill=color, outline="white", width=2)
            self.canvas.create_text(cx, cy, text=str(i), fill="white",
                                    font=("", 13, "bold"))

        self.status_var.set(f"动画播放中  t = {timestep} / {self.max_timestep}")

    # ---------- 交互事件 ----------

    def _cell_from_event(self, event):
        c = event.x // CELL_SIZE
        r = event.y // CELL_SIZE
        if 0 <= r < self.rows and 0 <= c < self.cols:
            return (r, c)
        return None

    def _on_mode_change(self):
        mode_text = {"obstacle": "障碍物", "start": "起点", "goal": "终点"}
        self.mode = self.mode_var.get()
        self.status_var.set(f"模式: {mode_text[self.mode]} | 左键放置, 右键取消")

    def _on_left_click(self, event):
        pos = self._cell_from_event(event)
        if pos is None:
            return
        self._stop()
        self.paths = None

        if self.mode == "obstacle":
            if pos not in self.starts and pos not in self.goals:
                self.obstacles.add(pos)
        elif self.mode == "start":
            if pos not in self.obstacles and pos not in self.starts:
                self.starts.append(pos)
        elif self.mode == "goal":
            if pos not in self.obstacles and pos not in self.goals:
                self.goals.append(pos)

        self._draw_grid()

    def _on_right_click(self, event):
        pos = self._cell_from_event(event)
        if pos is None:
            return
        self._stop()
        self.paths = None

        if self.mode == "obstacle":
            self.obstacles.discard(pos)
        elif self.mode == "start":
            if pos in self.starts:
                self.starts.remove(pos)
        elif self.mode == "goal":
            if pos in self.goals:
                self.goals.remove(pos)

        self._draw_grid()

    # ---------- 运行 / 动画 ----------

    def _run(self):
        self._stop()

        if len(self.starts) == 0:
            messagebox.showwarning("提示", "请至少放置一个起点和一个终点。")
            return
        if len(self.starts) != len(self.goals):
            messagebox.showwarning("提示",
                                   f"起点数 ({len(self.starts)}) 与终点数 ({len(self.goals)}) 不一致，"
                                   "请确保一一对应。")
            return

        env = MAPFEnv(
            width=self.cols,
            height=self.rows,
            obstacles=list(self.obstacles),
            starts=list(self.starts),
            goals=list(self.goals),
        )

        self.status_var.set("正在求解 CBS …")
        self.master.update_idletasks()

        solution = cbs_search(env)
        if solution is None:
            messagebox.showerror("无解", "CBS 未找到可行路径，请检查地图设置。")
            self.status_var.set("求解失败")
            return

        self.paths = solution
        self.max_timestep = max(len(p) - 1 for p in solution)
        self.timestep = 0
        self._animate()

    def _animate(self):
        if self.timestep > self.max_timestep:
            self.status_var.set(f"动画完成  总步数 = {self.max_timestep}")
            self.anim_id = None
            return
        self._draw_agents(self.timestep)
        self.timestep += 1
        self.anim_id = self.master.after(ANIM_DELAY_MS, self._animate)

    def _stop(self):
        if self.anim_id is not None:
            self.master.after_cancel(self.anim_id)
            self.anim_id = None

    def _clear_paths(self):
        self._stop()
        self.paths = None
        self._draw_grid()
        self.status_var.set("路径已清除")

    def _reset_grid(self):
        self._stop()
        self.rows = self.row_var.get()
        self.cols = self.col_var.get()
        self.obstacles.clear()
        self.starts.clear()
        self.goals.clear()
        self.paths = None

        canvas_w = self.cols * CELL_SIZE
        canvas_h = self.rows * CELL_SIZE
        self.canvas.config(width=canvas_w, height=canvas_h)
        self._draw_grid()
        self.status_var.set("网格已重置")


def main():
    root = tk.Tk()
    Simulation(root)
    root.mainloop()


if __name__ == "__main__":
    main()
