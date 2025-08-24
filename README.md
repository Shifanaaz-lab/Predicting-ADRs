# Predicting-ADRs
This project uses a deep learning system with GNNs, CNNs, and FFNNs to forecast ADRs by modeling protein-drug interactions. The goal is to apply AI to critical healthcare problems by predicting potential drug side effects with high accuracy.

Predicting Adverse Drug Reactions (ADRs) using Deep Learning
This project uses a deep learning system with GNNs, CNNs, and FFNNs to forecast ADRs by modeling protein-drug interactions. The goal is to apply AI to critical healthcare problems by predicting potential drug side effects with high accuracy.

Project Highlights
Multi-Model Architecture: The system employs a combination of GNNs, CNNs, and FFNNs to capture different aspects of protein-drug interaction data, from structural relationships to sequence-based patterns.

High Accuracy: The model achieved an impressive Area Under the Curve (ROC-AUC) of 0.9880, demonstrating its strong predictive capability.

Dataset Strategy: The model was trained on synthetic datasets and a focus on class balancing to ensure robust and reliable performance.

Computational Efficiency: The project utilizes GPU acceleration through PyTorch and CUDA to handle large-scale data and complex model training.

Technologies Used
Python: The core programming language for the project.

PyTorch & PyTorch Geometric: The primary framework for building and training the deep learning models, especially the Graph Neural Networks.

Numpy & Pandas: For efficient data manipulation and preprocessing.

Scikit-learn: Used for splitting the data and calculating performance metrics.

Google Colab: The environment used for development and execution, leveraging its GPU capabilities.

Other Libraries: The project also leverages torchvision, networkx, and other relevant packages as seen in the notebook.

Getting Started
Prerequisites
To run this notebook, you will need to have a Google account to use Google Colab. The notebook handles most of the library installations, but you can install them manually if running locally:

pip install torch torchvision torch-geometric pandas numpy scikit-learn networkx



Dataset
Due to the nature of the data (likely large or proprietary), the dataset is not included in this repository. The notebook is set up to mount a Google Drive, suggesting the data is stored there.

Usage
Open the final_gnn(afra).ipynb notebook in Google Colab.

Follow the instructions in the notebook to mount your Google Drive.

Ensure your dataset is correctly placed in your Google Drive and the paths in the notebook are updated accordingly.

Run the cells sequentially to install dependencies, load the data, train the model, and evaluate the results.

Contribution
This project is a strong starting point for further research. Feel free to fork the repository and experiment with different model architectures, datasets, or training strategies via a pull request.

License
This project is licensed under the MIT License - see the LICENSE file for details.
