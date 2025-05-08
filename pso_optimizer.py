"""
Particle Swarm Optimization (PSO) implementation for optimizing neural network hyperparameters
in the Exchange Rate Prediction project.

This module provides functionality to optimize hyperparameters for both MLP and CNN models,
in both univariate and multivariate configurations.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import models, layers, backend, optimizers
from keras.callbacks import EarlyStopping
import time
import random
from typing import Dict, List, Callable, Tuple, Any, Union
import matplotlib.pyplot as plt

class Particle:
    """
    Represents a particle in the PSO algorithm, which corresponds to a specific
    hyperparameter configuration for a neural network.
    """
    def __init__(self, bounds: Dict[str, Tuple[float, float]], inertia_weight: float = 0.5,
                 cognitive_weight: float = 1.5, social_weight: float = 1.5):
        """
        Initialize a particle with random position and velocity within the given bounds.

        Args:
            bounds: Dictionary mapping parameter names to (min, max) tuples
            inertia_weight: Weight for the particle's previous velocity
            cognitive_weight: Weight for the particle's personal best
            social_weight: Weight for the swarm's global best
        """
        self.position = {}
        self.velocity = {}
        self.bounds = bounds
        self.inertia_weight = inertia_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight

        # Initialize random position and velocity within bounds
        for param, (min_val, max_val) in bounds.items():
            if isinstance(min_val, int) and isinstance(max_val, int):
                self.position[param] = random.randint(min_val, max_val)
                self.velocity[param] = random.uniform(-abs(max_val - min_val) * 0.1,
                                                     abs(max_val - min_val) * 0.1)
            else:
                self.position[param] = random.uniform(min_val, max_val)
                self.velocity[param] = random.uniform(-abs(max_val - min_val) * 0.1,
                                                     abs(max_val - min_val) * 0.1)

        self.best_position = self.position.copy()
        self.best_fitness = float('inf')  # For minimization problems like RMSE
        self.current_fitness = float('inf')

    def update_velocity(self, global_best_position: Dict[str, float]):
        """
        Update the particle's velocity based on its previous velocity,
        personal best, and global best.

        Args:
            global_best_position: The best position found by any particle in the swarm
        """
        for param in self.position:
            # Skip parameters that might be added by the fitness function but aren't part of the optimization
            if param not in global_best_position or param not in self.best_position:
                continue

            r1 = random.random()
            r2 = random.random()

            cognitive_component = self.cognitive_weight * r1 * (self.best_position[param] - self.position[param])
            social_component = self.social_weight * r2 * (global_best_position[param] - self.position[param])

            self.velocity[param] = (self.inertia_weight * self.velocity[param] +
                                   cognitive_component + social_component)

    def update_position(self):
        """Update the particle's position based on its velocity."""
        for param in self.position:
            # Skip parameters that aren't part of the optimization
            if param not in self.velocity or param not in self.bounds:
                continue

            self.position[param] += self.velocity[param]

            # Ensure the position stays within bounds
            min_val, max_val = self.bounds[param]
            if isinstance(min_val, int) and isinstance(max_val, int):
                self.position[param] = int(max(min_val, min(max_val, self.position[param])))
            else:
                self.position[param] = max(min_val, min(max_val, self.position[param]))

    def update_best(self, fitness: float):
        """
        Update the particle's personal best if the current fitness is better.

        Args:
            fitness: The fitness value of the current position
        """
        self.current_fitness = fitness
        if fitness < self.best_fitness:
            self.best_fitness = fitness
            # Only copy the parameters that are part of the optimization bounds
            self.best_position = {param: self.position[param] for param in self.bounds}


class PSOOptimizer:
    """
    Particle Swarm Optimization algorithm for hyperparameter optimization.
    """
    def __init__(self, bounds: Dict[str, Tuple[float, float]], num_particles: int = 10,
                 max_iterations: int = 20, inertia_weight: float = 0.5,
                 cognitive_weight: float = 1.5, social_weight: float = 1.5):
        """
        Initialize the PSO optimizer.

        Args:
            bounds: Dictionary mapping parameter names to (min, max) tuples
            num_particles: Number of particles in the swarm
            max_iterations: Maximum number of iterations to run
            inertia_weight: Weight for the particle's previous velocity
            cognitive_weight: Weight for the particle's personal best
            social_weight: Weight for the swarm's global best
        """
        self.bounds = bounds
        self.num_particles = num_particles
        self.max_iterations = max_iterations
        self.inertia_weight = inertia_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight

        # Initialize particles
        self.particles = [
            Particle(bounds, inertia_weight, cognitive_weight, social_weight)
            for _ in range(num_particles)
        ]

        self.global_best_position = None
        self.global_best_fitness = float('inf')

        # For tracking progress
        self.fitness_history = []

    def optimize(self, fitness_function: Callable[[Dict[str, float]], float]) -> Tuple[Dict[str, float], float, List[float]]:
        """
        Run the PSO algorithm to find the optimal hyperparameters.

        Args:
            fitness_function: Function that takes hyperparameters and returns a fitness value (lower is better)

        Returns:
            Tuple containing:
            - The best hyperparameters found
            - The fitness value of the best hyperparameters
            - List of best fitness values at each iteration
        """
        start_time = time.time()

        # Initialize global best
        for particle in self.particles:
            # Create a copy of the position to avoid modifying the original
            position_copy = particle.position.copy()

            # Evaluate fitness with the copied position
            fitness = fitness_function(position_copy)
            particle.current_fitness = fitness
            particle.best_fitness = fitness

            if fitness < self.global_best_fitness:
                self.global_best_fitness = fitness
                # Only copy the parameters that are part of the optimization bounds
                self.global_best_position = {param: particle.position[param] for param in self.bounds}

        # Main PSO loop
        for iteration in range(self.max_iterations):
            for particle in self.particles:
                # Update velocity and position
                particle.update_velocity(self.global_best_position)
                particle.update_position()

                # Create a copy of the position to avoid modifying the original
                position_copy = particle.position.copy()

                # Evaluate fitness
                fitness = fitness_function(position_copy)
                particle.update_best(fitness)

                # Update global best
                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    # Only copy the parameters that are part of the optimization bounds
                    self.global_best_position = {param: particle.position[param] for param in self.bounds}

            # Record best fitness for this iteration
            self.fitness_history.append(self.global_best_fitness)

            # Print progress
            elapsed_time = time.time() - start_time
            print(f"Iteration {iteration+1}/{self.max_iterations}, Best Fitness: {self.global_best_fitness:.6f}, Time: {elapsed_time:.2f}s")

        return self.global_best_position, self.global_best_fitness, self.fitness_history

    def plot_progress(self, title: str = "PSO Optimization Progress"):
        """
        Plot the optimization progress over iterations.

        Args:
            title: Title for the plot
        """
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.fitness_history) + 1), self.fitness_history, marker='o')
        plt.title(title)
        plt.xlabel("Iteration")
        plt.ylabel("Best Fitness (RMSE)")
        plt.grid(True)
        plt.show()


# Helper functions for creating and evaluating models with specific hyperparameters
    """
    Create an MLP model with the given hyperparameters.

    Args:
        hyperparams: Dictionary of hyperparameters
        input_shape: Shape of the input data

    Returns:
        Compiled Keras model
    """
    backend.clear_session()
    model = models.Sequential()

    # Get activation function based on activation_choice
    activation_functions = ['swish', 'relu', 'tanh']
    activation = activation_functions[int(hyperparams.get('activation_choice', 0))]

    # First layer with input shape
    model.add(layers.Dense(
        hyperparams['neurons_layer_1'],
        activation=activation,
        kernel_initializer='he_uniform',
        input_shape=input_shape
    ))

    # Add additional layers based on hyperparameters
    for i in range(2, hyperparams['num_layers'] + 1):
        layer_size_param = f'neurons_layer_{i}'
        if layer_size_param in hyperparams:
            model.add(layers.Dense(
                hyperparams[layer_size_param],
                activation=activation,
                kernel_initializer='he_uniform'
            ))

            # Add dropout if specified
            if hyperparams.get('use_dropout', False) and hyperparams.get('dropout_rate', 0) > 0:
                model.add(layers.Dropout(hyperparams['dropout_rate']))

    # Output layer
    model.add(layers.Dense(1, activation='linear'))

    # Compile model
    optimizer = optimizers.Adam(learning_rate=hyperparams['learning_rate'])
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    return model

def create_cnn_model(hyperparams: Dict[str, Any], input_shape: Tuple[int, ...]) -> keras.Model:
    """
    Create a CNN model with the given hyperparameters.

    Args:
        hyperparams: Dictionary of hyperparameters
        input_shape: Shape of the input data

    Returns:
        Compiled Keras model
    """
    backend.clear_session()
    model = models.Sequential()

    # Get activation function based on activation_choice
    activation_functions = ['swish', 'relu', 'tanh']
    activation = activation_functions[int(hyperparams.get('activation_choice', 0))]

    # Reshape input for CNN if needed (for univariate data)
    if len(input_shape) == 1:
        model.add(layers.Reshape((input_shape[0], 1), input_shape=input_shape))
        current_shape = (input_shape[0], 1)
    else:
        current_shape = input_shape

    # Add convolutional layers
    # Keep track of the current sequence length
    current_seq_length = current_shape[0]

    for i in range(1, hyperparams['num_conv_layers'] + 1):
        filters_param = f'filters_layer_{i}'
        kernel_size_param = f'kernel_size_layer_{i}'

        filters = hyperparams.get(filters_param, 32)  # Default to 32 if not specified
        kernel_size = hyperparams.get(kernel_size_param, 3)  # Default to 3 if not specified

        model.add(layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            activation=activation,
            padding='same',
            input_shape=current_shape if i == 1 else None
        ))

        # Add pooling if specified and if the sequence is long enough
        if hyperparams.get('use_pooling', True):
            # Get the requested pool size
            requested_pool_size = hyperparams.get('pool_size', 2)

            # Determine a safe pool size based on current sequence length
            # We need at least 1 element after pooling
            safe_pool_size = min(requested_pool_size, current_seq_length)

            # Only add pooling if we can safely pool
            if safe_pool_size > 1:
                model.add(layers.MaxPooling1D(pool_size=safe_pool_size))
                # Update the sequence length after pooling
                current_seq_length = current_seq_length // safe_pool_size

    # Flatten before dense layers
    model.add(layers.Flatten())

    # Add dense layers
    for i in range(1, hyperparams['num_dense_layers'] + 1):
        dense_size_param = f'dense_neurons_layer_{i}'
        dense_size = hyperparams.get(dense_size_param, 64)  # Default to 64 if not specified

        model.add(layers.Dense(
            dense_size,
            activation=activation,
            kernel_initializer='he_uniform'
        ))

        # Add dropout if specified
        if hyperparams.get('use_dropout', False) and hyperparams.get('dropout_rate', 0) > 0:
            model.add(layers.Dropout(hyperparams['dropout_rate']))

    # Output layer
    model.add(layers.Dense(1, activation='linear'))

    # Compile model
    optimizer = optimizers.Adam(learning_rate=hyperparams['learning_rate'])
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    return model
