from setuptools import setup, find_packages

setup(
    name="rankbench",  
    version="0.1.0", 
    description="Benchmarking representations for ranking",
    long_description=open("README.md").read(),  
    long_description_content_type="text/markdown",
    author="Ankit Sonthalia",
    author_email="ankitsonthalia24@gmail.com",
    url="https://github.com/aktsonthalia/rankanything",  
    package_dir={"": "src"},  # Root package directory is 'src'
    packages=find_packages(where="src"),  # Find packages inside 'src'
    install_requires=[
        "numpy>=1.18.0",
        "scipy>=1.4.0"
        # Add other dependencies here
    ],
    python_requires=">=3.7",  # Minimum Python version
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Replace with your chosen license
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": [
            "your_command=your_package.module:function"  # CLI command (optional)
        ]
    },
    include_package_data=True,  # Include data files specified in MANIFEST.in
    zip_safe=False,  # Set to False if using package data
)
