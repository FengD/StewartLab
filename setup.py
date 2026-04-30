from setuptools import find_packages, setup


setup(
    name="stewart_test",
    version="0.1.0",
    description="Isaac Lab Stewart platform reinforcement learning extension.",
    package_dir={"": "source/stewart_test"},
    packages=find_packages(where="source/stewart_test", include=["stewart_test*"]),
    install_requires=["psutil"],
    python_requires=">=3.10",
    zip_safe=False,
)
