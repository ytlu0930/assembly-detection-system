"""Manual Graphviz installation check; optional dependency is loaded lazily."""


def main() -> int:
    from graphviz import Digraph
    diagram = Digraph("Flowchart")
    diagram.node("A", "Start")
    diagram.node("B", "Analyze image")
    diagram.node("C", "Output result")
    diagram.edges([("A", "B"), ("B", "C")])
    diagram.render("graphviz_flowchart", format="png", cleanup=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
