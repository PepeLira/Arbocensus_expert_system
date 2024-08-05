from setuptools import setup, find_namespace_packages

setup(
    name="tree_reviewer",
    version="0.1.0",
    packages=find_namespace_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
)