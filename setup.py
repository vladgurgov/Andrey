"""
Setup script for Mobile Agent Library.
"""

from setuptools import setup, find_packages
import os

# Read README file
current_dir = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(current_dir, "README.md"), "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open(os.path.join(current_dir, "requirements.txt"), "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="mobile-agent-lib",
    version="1.0.0",
    author="Mobile Agent Team",
    author_email="support@mobileagent.dev",
    description="AI-powered Android automation tool using ADB and OpenAI Vision API",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/mobile-agent-lib",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Testing",
        "Topic :: System :: Systems Administration",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0", 
            "black>=22.0.0",
            "flake8>=5.0.0",
        ],
    },
    keywords="android automation adb openai vision ai mobile testing",
    project_urls={
        "Bug Reports": "https://github.com/your-username/mobile-agent-lib/issues",
        "Source": "https://github.com/your-username/mobile-agent-lib",
        "Documentation": "https://github.com/your-username/mobile-agent-lib/blob/main/README.md",
    },
    include_package_data=True,
    zip_safe=False,
) 