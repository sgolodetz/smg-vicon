from setuptools import find_packages, setup

# with open("README.md", "r") as fh:
#     long_description = fh.read()

setup(
    name="smg-vicon",
    version="0.0.1",
    author="Stuart Golodetz",
    author_email="stuart.golodetz@cs.ox.ac.uk",
    description="Interface to a Vicon system",
    long_description="",  #long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sgolodetz/smg-vicon",
    packages=find_packages(include=["smg.vicon", "smg.vicon.*"]),
    include_package_data=True,
    install_requires=[
        "numpy",
        "smg-skeletons",
        "smg-utility"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
