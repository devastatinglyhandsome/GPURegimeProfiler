from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="gpu-regime-profiler",
    version="1.0.0",
    description="GPU performance profiler with three-regime classification (overhead/memory/compute-bound)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Prithiv",
    author_email="prithivnatarajan@gmail.com",
    url="https://github.com/devastatinglyhandsome/GPURegimeProfiler",
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
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Hardware",
    ],
    entry_points={
        'console_scripts': [
            'gpu-profile=gpu_regime_profiler.cli:main',
        ],
    },
    keywords="gpu profiling performance optimization pytorch cuda",
    extras_require={
        'dashboard': [
            'fastapi>=0.100.0',
            'uvicorn[standard]>=0.23.0',
            'websockets>=11.0',
            'requests>=2.28.0',
            'python-multipart>=0.0.5',
            'pyngrok>=5.0.0',
        ],
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/devastatinglyhandsome/GPURegimeProfiler/issues",
        "Source": "https://github.com/devastatinglyhandsome/GPURegimeProfiler",
        "Documentation": "https://github.com/devastatinglyhandsome/GPURegimeProfiler#readme",
    },
)
