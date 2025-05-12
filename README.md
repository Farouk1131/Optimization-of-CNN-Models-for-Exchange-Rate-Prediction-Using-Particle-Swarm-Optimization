# Brain Tumor Classification with CNN and PSO Optimization

This repository contains a Convolutional Neural Network (CNN) implementation for brain tumor classification from MRI images, along with Particle Swarm Optimization (PSO) for hyperparameter tuning. The project includes a Jupyter notebook for experimentation, Python scripts for training and optimization, and a GUI application for easy model inference.

## Project Overview

The system classifies brain MRI images into four categories:
- Glioma
- Meningioma
- No tumor
- Pituitary

The project consists of the following components:
1. CNN model implementation for brain tumor classification
2. PSO algorithm for hyperparameter optimization
3. Jupyter notebook for experimentation and visualization
4. GUI application for easy model inference

## Software Dependencies

### Required Packages

```
torch>=1.7.0
torchvision>=0.8.0
numpy>=1.19.0
matplotlib>=3.3.0
scikit-learn>=0.23.0
pandas>=1.1.0
seaborn>=0.11.0
pillow>=8.0.0
PyQt5>=5.15.0 (for GUI application)
```

### Installation

Install all required packages using pip:

```bash
pip install torch torchvision numpy matplotlib scikit-learn pandas seaborn pillow PyQt5
```

For GPU acceleration (recommended for training):

```bash
pip install torch==1.7.0+cu110 torchvision==0.8.0+cu110 -f https://download.pytorch.org/whl/torch_stable.html
```

## Dataset Structure

The dataset should be organized in the following structure:

```
Dataset/
├── Training/
│   ├── glioma/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── meningioma/
│   │   └── ...
│   ├── notumor/
│   │   └── ...
│   └── pituitary/
│       └── ...
└── Testing/
    ├── glioma/
    │   └── ...
    ├── meningioma/
    │   └── ...
    ├── notumor/
    │   └── ...
    └── pituitary/
        └── ...
```

## Running the Code

### Training the CNN Model

To train the basic CNN model without PSO optimization:

```bash
python brain_tumor/brain_tumor_cnn.py
```

This will:
1. Load and preprocess the dataset
2. Train the CNN model
3. Evaluate on the test set
4. Save the best model as `best_model.pth`
5. Generate a confusion matrix visualization

### Running PSO Optimization

To optimize the CNN hyperparameters using PSO:

```bash
python brain_tumor_cnn_pso.py
```

This will:
1. Run the PSO algorithm to find optimal hyperparameters
2. Train multiple CNN models with different hyperparameter configurations
3. Save the best hyperparameters as `best_pso_params.npy`
4. Print optimization progress and results

### Using the Jupyter Notebook

The Jupyter notebook `brain_tumor_cnn_pso_notebook.ipynb` provides an interactive environment for experimentation:

```bash
jupyter notebook brain_tumor_cnn_pso_notebook.ipynb
```

The notebook includes:
- Data exploration and visualization
- CNN model implementation
- PSO algorithm implementation
- Training and evaluation
- Results visualization

### Using the GUI Application

To use the GUI application for inference:

```bash
python brain_tumor_gui.py
```

The GUI allows you to:
1. Load a trained model
2. Upload an MRI image
3. Preprocess the image
4. Make a prediction
5. View the classification results with confidence scores

## File Descriptions

- `brain_tumor/brain_tumor_cnn.py`: Implementation of the CNN model for brain tumor classification
- `brain_tumor_cnn_pso.py`: Implementation of PSO for hyperparameter optimization
- `brain_tumor_cnn_pso_notebook.ipynb`: Jupyter notebook for experimentation and visualization
- `brain_tumor_gui.py`: GUI application for model inference
- `best_model.pth`: Saved weights of the best CNN model
- `best_pso_params.npy`: Saved optimal hyperparameters found by PSO
- `confusion_matrix.png`: Visualization of model performance on test data

## Understanding the PSO Algorithm

The PSO algorithm optimizes the following hyperparameters:
- Learning rate
- Weight decay (L2 regularization)
- Dropout rates for different layers
- Number of filters in convolutional layers
- Batch size

The optimization process works as follows:
1. Initialize a population of particles with random hyperparameter values
2. For each iteration:
   - Evaluate each particle by training a CNN with its hyperparameters
   - Update personal best and global best positions
   - Update particle velocities and positions
3. Return the best hyperparameters found

## Interpreting the Outputs

### Training Outputs

During training, the following metrics are displayed:
- Training loss and accuracy for each epoch
- Validation loss and accuracy for each epoch
- Learning rate adjustments
- Early stopping notifications

Example:
```
Epoch 1/50
Train Loss: 0.9876, Train Acc: 0.6543
Val Loss: 0.8765, Val Acc: 0.7654
```

### PSO Optimization Outputs

During PSO optimization, the following information is displayed:
- Current iteration number
- Global best fitness (validation accuracy)
- Average fitness of all particles
- Best hyperparameters found so far

Example:
```
Iteration 5/20
Global best fitness: 0.8765
Average fitness: 0.7654
Best hyperparameters:
Learning rate: 0.001234
Weight decay: 0.000123
...
```

### Evaluation Outputs

After training, the model is evaluated on the test set, producing:
- Test loss and accuracy
- Classification report with precision, recall, and F1-score
- Confusion matrix visualization

Example:
```
Test Loss: 0.3456, Test Acc: 0.8765

Classification Report:
              precision    recall  f1-score   support
     glioma      0.92      0.89      0.90        50
 meningioma      0.88      0.90      0.89        50
    notumor      0.95      0.96      0.95        50
  pituitary      0.91      0.92      0.91        50
```

The confusion matrix (`confusion_matrix.png`) shows the number of correct and incorrect predictions for each class, helping to identify which classes are most challenging for the model.

### GUI Outputs

The GUI application displays:
- Model information (architecture, device)
- Selected image
- Prediction results with class probabilities
- Status messages for each step of the process

## Troubleshooting

### Common Issues

1. **CUDA out of memory error**:
   - Reduce batch size
   - Use a smaller model
   - Free up GPU memory by closing other applications

2. **Model not loading**:
   - Ensure the model file path is correct
   - Check that the model architecture matches the saved weights
   - Use the "Load Model Manually" button in the GUI to select the model file

3. **Dataset not found**:
   - Verify the dataset directory structure
   - Check the paths in the code match your actual directory structure

4. **Slow PSO optimization**:
   - Reduce the number of particles or iterations
   - Use a smaller CNN model for initial optimization
   - Enable GPU acceleration

## Citation

If you use this code in your research, please cite:

```
@software{brain_tumor_cnn_pso,
  author = {Your Name},
  title = {Brain Tumor Classification with CNN and PSO Optimization},
  year = {2023},
  url = {https://github.com/yourusername/brain-tumor-classification}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset provided by [source of the dataset]
- PSO implementation inspired by [reference]
- CNN architecture based on [reference]
