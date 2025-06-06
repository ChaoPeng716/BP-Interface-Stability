# Supporting Materials for "AI-Driven Discovery and Intelligent Molecular Engineering Design for Enhancing Interface Stability of Black Phosphorus"

This repository contains the supporting code and materials for the article "AI-Driven Discovery and Intelligent Molecular Engineering Design for Enhancing Interface Stability of Black Phosphorus". The project implements two different graph neural network (GNN) approaches for enabling high-throughput predictions of molecule-black phosphorus interaction energy (Eint_BP) and molecule-water interaction energy (Eint_H2O).

## Repository Structure

### GNN_Eint_BP
**Graph Neural Network for Black Phosphorus-molecule Interaction Energy Prediction**

This directory contains a GNN implementation based on the gnn_eads framework, which is derived from GAME-Net (Graph-based Adsorption on Metal Energy-neural Network). The GAME-NET architecture was employed for Eint_BP prediction due to its demonstrated success in predicting molecular adsorption energies, incorporating surface-specific information intrinsically.

**Main Files:**
- `train_GNN.py`: Script for training GNN models with customizable hyperparameters
- `prediction_GNN.py`: Script for making predictions using trained models
- `GNN.yaml`: Environment configuration file
- `requirements.txt`: Python dependencies
- `src/gnn_eads/`: Core GNN implementation modules

### GNN_Eint_H2O
**Graph Neural Network for Water-molecule Interaction Energy Prediction**

This directory contains a GNN implementation based on the ChemProp framework. The directed message passing neural network (D-MPNN) architecture implemented in ChemProp was adopted for Eint_H2O prediction, considering its proven performance across a variety of molecular property prediction tasks, including analogous tasks like hydration free energy and water solubility prediction.

**Main Files:**
- `train_Eint_H2O.ipynb`: Training workflow notebook
- `predict_Eint_H2O.ipynb`: Prediction workflow notebook
- `mpnn_model.py`: Message Passing Neural Network implementation
- `preprocess.py`: Data preprocessing utilities
- `dataset_Eint_H2O.csv`: Training dataset
- `Eint_H2O_1203.ckpt`: Pre-trained model checkpoint

## Installation and Setup

### For GNN_Eint_BP

1. **Environment Setup:**
   ```bash
   cd GNN_Eint_BP
   conda env create -f GNN.yaml
   conda activate GNN
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### For GNN_Eint_H2O

1. **Install ChemProp:**
   ```bash
   pip install chemprop
   ```

## Usage

### GNN_Eint_BP

**Training:**
```bash
python train_GNN.py -i hyper_config.toml -o output_directory
```
Note: Update the data path in the configuration file.

**Prediction:**
```bash
python prediction_GNN.py
```
Note: Update the file paths in the script.

### GNN_Eint_H2O

**Training:**
Follow the workflow in `train_Eint_H2O.ipynb`.

**Prediction:**
Follow the workflow in `predict_Eint_H2O.ipynb`.

## References

- **GAME-Net**: Graph-based Adsorption on Metal Energy-neural Network for predicting adsorption energy of closed-shell molecules on metal surfaces. Original implementation available at: https://gitlab.com/iciq-tcc/nlopez-group/gnn_eads
- **ChemProp**: Message Passing Neural Networks for Molecule Property Prediction. GitHub repository: https://github.com/chemprop/chemprop
  - Yang, K., Swanson, K., Jin, W., Coley, C., Eiden, P., Gao, H., ... & Barzilay, R. (2019). Analyzing learned molecular representations for property prediction. Journal of chemical information and modeling, 59(8), 3370-3388.



