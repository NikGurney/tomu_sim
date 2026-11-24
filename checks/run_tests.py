"""Standard-library runner for the pytest-compatible test functions."""

import importlib
import inspect


MODULES = (
    "tests.test_data_model",
    "tests.test_inference",
    "tests.test_motivated_episode",
    "tests.test_proliferation",
    "tests.test_residue",
    "tests.test_runs",
    "tests.test_scenarios",
    "tests.test_trace",
)


def main() -> None:
    """Execute every zero-argument test function in deterministic order."""

    count = 0
    for module_name in MODULES:
        module = importlib.import_module(module_name)
        tests = [
            (name, function)
            for name, function in inspect.getmembers(module, inspect.isfunction)
            if name.startswith("test_")
        ]
        for _, function in tests:
            function()
            count += 1
    print(f"{count} test functions passed")


if __name__ == "__main__":
    main()
