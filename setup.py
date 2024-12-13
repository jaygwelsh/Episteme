from setuptools import setup, find_packages

setup(
    name='episteme',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        # Allow newer numpy for Python 3.12 compatibility
        'numpy>=1.21.0,<3.0.0',
        # Allow newer hydra-core (1.3.2) for Python 3.12
        'hydra-core>=1.1.0,<2.0.0',  # or <2.0.0 to allow 1.3.2
        # If you must keep mlflow <2.0.0, choose a version <=1.29.0 for mlflow in requirements.txt
        # Otherwise, relax mlflow constraint:
        'mlflow>=1.28.0,<3.0.0',  # Allows mlflow 2.x
        # ... other dependencies as needed ...
    ],
    author='Your Name',
    author_email='your.email@example.com',
    description='Episteme: A lean, flexible neural network framework.',
    url='https://github.com/yourusername/Episteme',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
