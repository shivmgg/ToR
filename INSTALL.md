# Installation

## Requirements
```bash
 - python >= 3.8
 - pytorch >= 1.12.1  # For scatter_reduce
 - torchvision        # With matching version for your pytorch install
 - timm == 0.4.12     # Might work on other versions, but this is what we tested
 - jupyter            # For example notebooks
 - scipy              # For visualization and sometimes torchvision requires it
```

## Setup
First, clone the repository:
```bash
git clone https://github.com/shivmgg/ToR.git
cd tor
```
Either install the requirements listed above manually, or use our conda environment:
```bash
conda env create --file environment.yml
conda activate tome
```
Then set up the tome package with:
```bash
python setup.py build develop
```
or
```pip 
install .
```