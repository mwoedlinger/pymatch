import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pymatch", # Replace with your own username
    version="0.1.1",
    author="Matthias Wödlinger",
    author_email="matthias.woedlinger@gmail.com",
    description="Neural networks from scratch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mwoedlinger/pymatch",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3',
)