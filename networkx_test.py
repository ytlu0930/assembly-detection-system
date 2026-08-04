"""Manual NetworkX rendering check; optional dependencies are loaded lazily."""


def main() -> int:
    import matplotlib.pyplot as plt
    import networkx as nx
    graph = nx.DiGraph()
    graph.add_edge("Start", "Analyze image")
    graph.add_edge("Analyze image", "Output result")
    nx.draw(graph, nx.spring_layout(graph, seed=1), with_labels=True, node_size=2500, arrows=True)
    plt.savefig("networkx_flowchart.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
