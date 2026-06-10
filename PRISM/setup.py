import setuptools
import os

script_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_folder)

setuptools.setup(
    name="prism",
    version="0.1.0",
    description="PRISM: Pareto-optimal Risk-constrained drIving Style Manifold",
    python_requires=">=3.9",
    packages=setuptools.find_packages(where="."),
    package_dir={"": "."},
    install_requires=[
        "stable-baselines3",
        "torch",
        "numpy",
        "scipy",
        "matplotlib",
        "pyyaml",
    ],
)
