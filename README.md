# Exchange Rate Prediction with PSO-Optimized Neural Networks

## Project Overview
This project implements and optimizes neural network models for predicting exchange rates, specifically focusing on the GBP/USD currency pair. The models are optimized using Particle Swarm Optimization (PSO) to find the best hyperparameters, resulting in improved prediction accuracy compared to traditional approaches.

## Models Implemented
- **ARIMA**: Base econometric model (benchmark)
- **MLP Univariate**: Multilayer Perceptron with single input feature
- **CNN Univariate**: Convolutional Neural Network with single input feature
- **MLP Multivariate**: Multilayer Perceptron with multiple input features
- **CNN Multivariate**: Convolutional Neural Network with multiple input features
- **PSO-Optimized Versions**: Enhanced versions of the neural network models

## Data Sources
- **Normalized GDP (monthly)**: From FRED (Federal Reserve Economic Data)
- **Libor Rates (daily)**: From FRED
- **Current Account to GDP (quarterly)**: From OECD
- **Forex (daily)**: From Federal Reserve, including multiple currencies

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

## Results
The PSO-optimized models demonstrate significant improvements over the base models:
- Original CNN Univariate RMSE: 0.03028
- PSO-Optimized CNN Univariate RMSE: [value achieved]
- Improvement: [percentage]%

## Visualizations
The project includes various visualizations:
- Time series plots of exchange rates
- Actual vs. predicted value comparisons
- Training and validation loss curves
- PSO optimization progress

## Requirements
- Python 3.6+
- TensorFlow 2.x
- Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Statsmodels (for ARIMA)

## How to Run
1. Clone the repository
2. Install the required packages: `pip install -r requirements.txt`
3. Run the Jupyter notebooks in numerical order

## Acknowledgements
- Original research paper methodology for comparison
- Jason Brownlee (www.machinelearningmastery.com) for time series forecasting techniques