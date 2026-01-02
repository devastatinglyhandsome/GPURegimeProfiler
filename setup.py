from setuptools import setup, find_packages

setup(
    name="GPURegimeProfiler",
    version="0.1.0",
    description="GPU performance profiler with three-regime classification (overhead/memory/compute-bound)",
    author="GPU Performance Team",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "matplotlib>=3.3.0",
        "numpy>=1.19.0",
        "seaborn>=0.11.0",
        "pynvml>=11.0.0",
        "tqdm>=4.60.0",
    ],
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Hardware",
    ],
    entry_points={
        'console_scripts': [
            'gpu-profile=gpu_regime_profiler.cli:main',
        ],
    },
)
