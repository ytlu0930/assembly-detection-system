from graphviz import Digraph

d = Digraph("Flowchart")

d.node("A", "開始")
d.node("B", "分析圖片")
d.node("C", "輸出結果")

d.edge("A", "B")
d.edge("B", "C")

d.render("graphviz_flowchart", format="png", cleanup=True)

print("完成")
