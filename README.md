# CNN Univariate Model for Exchange Rate Prediction

## Model Overview
This project implements a Convolutional Neural Network (CNN) for univariate time series forecasting of the GBP/USD exchange rate. The model uses historical exchange rate data as the sole input feature and is optimized using Particle Swarm Optimization (PSO) to find optimal hyperparameters.

## Software Dependencies
- Python 3.6+
- TensorFlow 2.x
- Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- tqdm (for progress bars)

## Installation Instructions
1. Clone the repository:
   ```
   https://github.com/Farouk1131/Optimization-of-CNN-Models-for-Exchange-Rate-Prediction-Using-Particle-Swarm-Optimization

   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

## Running the CNN Univariate Model
1. **Data Preparation**:
   - Run `2. Data Preparation.ipynb` to create the preprocessed dataset

2. **Base CNN Model**:
   - Run `5. CNN Univariate.ipynb` to implement the base CNN model
   - This notebook creates a CNN model with manually selected hyperparameters

3. **PSO-Optimized CNN Model**:
   - Run `10. PSO CNN Univariate.ipynb` to implement the PSO-optimized version
   - This uses `pso_optimizer.py` to find optimal hyperparameters
   - Note: The optimization process may take several hours

## Model Architecture
The CNN Univariate model consists of:
- Input layer accepting time series data with a specified window size
- 1-3 convolutional layers (optimized by PSO)
- Max pooling layers
- Flatten layer
- 1-3 dense layers (optimized by PSO)
- Output layer with a single neuron for regression

## Interpreting Outputs

### Visualizations
- **Training Loss Curve**: Shows model learning progress over epochs
- **PSO Optimization Progress**: Displays RMSE improvement as PSO searches for optimal hyperparameters
- **Actual vs. Predicted Plot**: Compares model predictions against actual exchange rates
  - Located in `PLOTS/PSO_CNN_U.png` after running the PSO notebook

### Performance Metrics
- **RMSE (Root Mean Squared Error)**: Primary evaluation metric
  - Original CNN Univariate RMSE: 0.03028
  - PSO-optimized CNN RMSE: [Check your results]
- **Improvement Percentage**: ((0.03028 - PSO_RMSE) / 0.03028) * 100

## Hyperparameters Optimized by PSO
- Number of convolutional layers
- Number of filters per layer
- Kernel size
- Number of dense layers
- Neurons per dense layer
- Activation functions
- Learning rate
- Batch size
- Dropout rate

## Troubleshooting
- **Memory Issues**: Reduce batch size or model complexity
- **Slow Convergence**: Adjust PSO parameters in `pso_optimizer.py`
- **Overfitting**: Increase dropout rate or reduce model complexity

## Project Structure
- `DATA/`: Contains all raw and processed datasets
- `PLOTS/`: Contains visualization outputs
- `pso_optimizer.py`: Implementation of the PSO algorithm
- Jupyter Notebooks:
  - `1. Introduction.ipynb`: Project overview and goals
  - `2. Data Preparation.ipynb`: Data cleaning and preprocessing
  - `3. ARIMA.ipynb`: Implementation of the base econometric model
  - `4. MLP Univariate.ipynb`: Implementation of the MLP model with single feature
  - `5. CNN Univariate.ipynb`: Implementation of the CNN model with single feature
  - `6. MLP Multivariate.ipynb`: Implementation of the MLP model with multiple features
  - `7. CNN Multivariate.ipynb`: Implementation of the CNN model with multiple features
  - `8. Comparison.ipynb`: Comparison of all base models
  - `9. PSO MLP Univariate.ipynb`: PSO-optimized MLP model
  - `10. PSO CNN Univariate.ipynb`: PSO-optimized CNN model
  - `11. PSO MLP Multivariate.ipynb`: PSO-optimized MLP model with multiple features
  - `12. PSO CNN Multivariate.ipynb`: PSO-optimized CNN model with multiple features
  - `13. Final Comparison.ipynb`: Comprehensive comparison of all models

## Acknowledgements
- Original research paper methodology for comparison
- Jason Brownlee (www.machinelearningmastery.com) for time series forecasting techniques



