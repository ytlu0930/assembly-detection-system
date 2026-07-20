import networkx as nx
import matplotlib.pyplot as plt

G = nx.DiGraph()

G.add_edge("開始", "分析圖片")
G.add_edge("分析圖片", "輸出結果")

plt.figure(figsize=(6,4))

pos = nx.spring_layout(G, seed=1)

nx.draw(
    G,
    pos,
    with_labels=True,
    node_size=2500,
    font_size=12,
    arrows=True
)

plt.savefig("networkx_flowchart.png")

plt.show()
