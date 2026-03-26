# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "pybind11-stubgen>=2.5.5",
# ]
# ///
import pybind11_stubgen


def main():
    pybind11_stubgen.main(["mujoco", "-o", "./.typing"])


if __name__ == "__main__":
    main()
