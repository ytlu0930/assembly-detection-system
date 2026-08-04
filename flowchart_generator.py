import os
import sys
import platform
try:
    import graphviz
except ImportError:  # Matplotlib remains the supported offline fallback.
    graphviz = None
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
except ImportError:  # Pillow fallback below keeps the module usable offline.
    plt = None
    patches = None
from pathlib import Path
from PIL import Image, ImageDraw

def get_system_chinese_font():
    """自動判斷作業系統並回傳對應的中文字型名稱"""
    sys_name = platform.system()
    if sys_name == "Windows":
        return "Microsoft JhengHei" # 微軟正黑體
    elif sys_name == "Darwin":
        return "PingFang TC"        # Mac 蘋方體
    else:
        return "WenQuanYi Zen Hei"  # Linux 文泉驛正黑

def setup_graphviz_environment():
    """自動搜尋常見的 Graphviz 安裝路徑並寫入環境變數"""
    if platform.system() == "Windows":
        possible_paths = [
            r"C:\Program Files\Graphviz\bin",
            r"C:\Program Files (x86)\Graphviz\bin",
            os.path.expanduser(r"~\AppData\Local\Programs\Graphviz\bin")
        ]
        for p in possible_paths:
            if os.path.exists(p) and p not in os.environ["PATH"]:
                os.environ["PATH"] += os.pathsep + p

def generate_flowchart_matplotlib(step_id: str, error_reports: list, target_path: str) -> bool:
    """備援機制：當 Graphviz 無法輸出圖片時，使用 Matplotlib 直接繪製流程圖"""
    try:
        font_name = get_system_chinese_font()
        plt.rcParams['font.sans-serif'] = [font_name, 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        fig, ax = plt.subplots(figsize=(7, 9))
        ax.axis('off')

        # 節點定義 (X, Y, 寬度, 高度, 文字, 顏色, 形狀)
        nodes = [
            (0.5, 0.88, 0.65, 0.1, "開始分析\n(Image Input)", "#E1F5FE", "ellipse"),
            (0.5, 0.70, 0.65, 0.1, "GPT-4o Vision 分析\n(Structured Output)", "#F3E5F5", "box"),
            (0.5, 0.52, 0.65, 0.1, "JSON Schema 驗證\n(Format Check)", "#E8EAF6", "box"),
            (0.5, 0.34, 0.65, 0.1, "邏輯比對與錯誤偵測\n(Error Detection)", "#FFF3E0", "box"),
        ]

        if not error_reports:
            status_text = "✅ 狀態：完全正確\n(All Correct)"
            status_color = "#E8F5E9"
            border_color = "#2E7D32"
        else:
            status_text = f"❌ 狀態：偵測到錯誤\n(發現 {len(error_reports)} 個問題)"
            status_color = "#FFEBEE"
            border_color = "#C62828"

        nodes.append((0.5, 0.16, 0.65, 0.1, status_text, status_color, "ellipse"))

        # 繪製節點
        for idx, (x, y, w, h, text, color, shape) in enumerate(nodes):
            edge_c = border_color if idx == len(nodes)-1 else "#424242"
            if shape == "ellipse":
                patch = patches.Ellipse((x, y), w, h, facecolor=color, edgecolor=edge_c, linewidth=1.8)
            else:
                patch = patches.FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=0.02", facecolor=color, edgecolor=edge_c, linewidth=1.5)
            ax.add_patch(patch)
            ax.text(x, y, text, ha='center', va='center', fontsize=11, fontweight='bold', color="#212121")

        # 繪製箭頭與標籤
        for i in range(len(nodes) - 1):
            y_start = nodes[i][1] - nodes[i][3]/2
            y_end = nodes[i+1][1] + nodes[i+1][3]/2
            
            line_color = border_color if i == len(nodes) - 2 else "#616161"
            ax.annotate('', xy=(0.5, y_end), xytext=(0.5, y_start),
                        arrowprops=dict(arrowstyle="->", color=line_color, lw=2))
            
            # 最後一條線的標籤（偏移避免重疊）
            if i == len(nodes) - 2:
                lbl = "無錯誤" if not error_reports else "發現錯誤"
                lbl_color = "#2E7D32" if not error_reports else "#C62828"
                ax.text(0.53, (y_start + y_end) / 2, lbl, color=lbl_color, fontweight='bold', fontsize=10)

        plt.tight_layout()
        plt.savefig(target_path, dpi=300, bbox_inches='tight')
        plt.close()
        return os.path.exists(target_path)
    except Exception as e:
        print(f"[錯誤] Matplotlib 繪圖亦失敗: {e}")
        return False

def generate_flowchart(step_id: str, error_reports: list, output_dir: str = "output/flowcharts") -> str:
    """
    生成積木組裝分析狀態的動態流程圖 (視覺優化雙引擎)
    """
    setup_graphviz_environment()

    # 確保輸出目錄為絕對路徑
    abs_output_dir = os.path.abspath(output_dir)
    os.makedirs(abs_output_dir, exist_ok=True)
    
    target_png_path = os.path.join(abs_output_dir, f"{step_id}_flowchart.png")
    font_name = get_system_chinese_font()

    # 嘗試引擎一：Graphviz Direct Pipe
    try:
        if plt is None or patches is None:
            return False
        if graphviz is None:
            raise RuntimeError("python-graphviz is not installed")
        dot = graphviz.Digraph(comment=f'Lego Assembly Pipeline - {step_id}')
        dot.attr(rankdir='TB', size='8,8', dpi='300')
        
        # 全域樣式設定
        dot.attr('node', fontname=font_name, fontsize='11', shape='box', style='filled,rounded', 
                 color='#424242', fillcolor='#F5F5F5', margin='0.2,0.1')
        dot.attr('edge', fontname=font_name, fontsize='10', color='#616161', penwidth='1.5')

        # 節點定義
        dot.node('A', '開始分析\n(Image Input)', shape='ellipse', fillcolor='#E1F5FE', color='#0288D1')
        dot.node('B', 'GPT-4o Vision 分析\n(Structured Output)', fillcolor='#F3E5F5', color='#7B1FA2')
        dot.node('C', 'JSON Schema 驗證\n(Format Check)', fillcolor='#E8EAF6', color='#3F51B5')
        dot.node('D', '邏輯比對與錯誤偵測\n(Error Detection)', fillcolor='#FFF3E0', color='#F57C00')

        if not error_reports:
            dot.node('E', '✅ 狀態：完全正確\n(All Correct)', shape='ellipse', fillcolor='#E8F5E9', color='#2E7D32', fontcolor='#1B5E20')
            edge_color = '#2E7D32'
            detect_label = '  無錯誤  '
        else:
            error_text = f'❌ 狀態：偵測到錯誤\n(發現 {len(error_reports)} 個問題)'
            dot.node('E', error_text, shape='ellipse', fillcolor='#FFEBEE', color='#C62828', fontcolor='#B71C1C')
            edge_color = '#C62828'
            detect_label = '  發現錯誤  '

        dot.edge('A', 'B')
        dot.edge('B', 'C')
        dot.edge('C', 'D')
        dot.edge('D', 'E', label=detect_label, color=edge_color, fontcolor=edge_color)

        png_bytes = dot.pipe(format='png')
        with open(target_png_path, 'wb') as f:
            f.write(png_bytes)

        if os.path.exists(target_png_path) and os.path.getsize(target_png_path) > 0:
            print(f"[Agent] ✅ Graphviz 繪圖成功產出 (美化版)：\n👉 {target_png_path}")
            return target_png_path

    except Exception as e:
        print(f"[訊息] Graphviz 引擎無法產出圖檔 ({e})，自動切換至 Matplotlib 繪圖引擎...")

    # 嘗試引擎二：Matplotlib 自動接手繪製
    success = generate_flowchart_matplotlib(step_id, error_reports, target_png_path)
    if success:
        print(f"[Agent] ✅ Matplotlib 備援引擎繪圖成功產出 (美化版)：\n👉 {target_png_path}")
        return target_png_path

    print(f"[錯誤] 無法生成流程圖：{target_png_path}")
    return ""


def generate_sop_flowchart(
    correction_sop: dict,
    output_dir: str = "output/flowcharts",
) -> str:
    """Generate the optional overview from structured SOP, not Vision JSON.

    This intentionally remains separate from per-step instruction images, which
    are the primary repair output.
    """
    if not isinstance(correction_sop, dict):
        raise TypeError("correction_sop must be a dictionary")
    steps = correction_sop.get("steps", [])
    if not isinstance(steps, list):
        raise ValueError("correction_sop.steps must be a list")

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    step_id = str(correction_sop.get("source_step_id", "step"))
    target = output / f"{step_id}_sop_flowchart.png"

    if plt is None or patches is None:
        width = 1200
        height = max(360, (len(steps) + 2) * 150)
        canvas = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(canvas)
        labels = ["START LOCAL REPAIR"]
        labels.extend(
            f"STEP {item.get('step_number', index)} - {item.get('action', 'repair')}"
            for index, item in enumerate(steps, 1) if isinstance(item, dict)
        )
        labels.append("VERIFY AGAINST REFERENCE" if steps else "ASSEMBLY CORRECT")
        for index, label in enumerate(labels):
            top = 40 + index * 130
            draw.rounded_rectangle((180, top, 1020, top + 75), radius=20, outline="#424242", width=4, fill="#FFF3E0")
            draw.text((230, top + 25), label, fill="black")
            if index < len(labels) - 1:
                draw.line((600, top + 75, 600, top + 125), fill="#616161", width=5)
                draw.polygon([(600, top + 125), (585, top + 105), (615, top + 105)], fill="#616161")
        canvas.save(target)
        return str(target)

    font_name = get_system_chinese_font()
    plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    node_count = max(2, len(steps) + 2)
    fig_height = max(4.5, node_count * 1.3)
    fig, ax = plt.subplots(figsize=(9, fig_height))
    ax.axis("off")
    labels = ["開始局部修正"]
    if steps:
        labels.extend(
            f"步驟 {item.get('step_number', index)}｜{item.get('action', 'repair')}\n"
            f"{item.get('instruction', '')}"
            for index, item in enumerate(steps, 1)
            if isinstance(item, dict)
        )
        labels.append("完成並對照正確參考圖")
    else:
        labels.append("組裝正確，無需修正")

    ys = list(reversed([(index + 1) / (len(labels) + 1) for index in range(len(labels))]))
    for index, (label, y_value) in enumerate(zip(labels, ys)):
        color = "#E8F5E9" if index in {0, len(labels) - 1} else "#FFF3E0"
        box = patches.FancyBboxPatch(
            (0.12, y_value - 0.035), 0.76, 0.07,
            boxstyle="round,pad=0.012", facecolor=color, edgecolor="#424242", linewidth=1.5,
        )
        ax.add_patch(box)
        ax.text(0.5, y_value, label, ha="center", va="center", fontsize=10)
        if index < len(labels) - 1:
            next_y = ys[index + 1]
            ax.annotate(
                "", xy=(0.5, next_y + 0.04), xytext=(0.5, y_value - 0.04),
                arrowprops=dict(arrowstyle="->", color="#616161", lw=1.8),
            )
    fig.savefig(target, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(target)

if __name__ == "__main__":
    print("=" * 60)
    print(f"📍 當前 Terminal 執行目錄 (CWD): {os.getcwd()}")
    print("=" * 60)
    
    path1 = generate_flowchart("step_01", [])
    path2 = generate_flowchart("step_02", [{"part_id": "P01", "error_type": "positionerror"}])
    
    target_dir = os.path.abspath("output/flowcharts")
    print("\n" + "=" * 60)
    print(f"📂 實體資料夾檢查: {target_dir}")
    if os.path.exists(target_dir):
        files = os.listdir(target_dir)
        print(f"📄 目前資料夾內的檔案數量: {len(files)}")
        for f in files:
            file_full = os.path.join(target_dir, f)
            size = os.path.getsize(file_full)
            print(f"   └── 🖼️ {f} (檔案大小: {size} bytes)")
    else:
        print("❌ 目錄尚未建立！")
    print("=" * 60)
