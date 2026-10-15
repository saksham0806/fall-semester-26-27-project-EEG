from pathlib import Path
import importlib.util


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "generate_figures", ROOT / "diagrams" / "generate_figures.py"
)
figures = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(figures)


def test_result_contracts():
    data = figures.load_results(ROOT)
    assert data["stress"]["Riemannian"] == (0.756037037037037, 0.05031459330094776)
    assert data["depression"]["SVM"] == 0.771551724137931
    assert data["density"]["riemannian"][16] == 0.6824712643678161
    assert data["all3"]["CogLoad"]["margin"] < 0


if __name__ == "__main__":
    test_result_contracts()
    print("figure data contracts: OK")
