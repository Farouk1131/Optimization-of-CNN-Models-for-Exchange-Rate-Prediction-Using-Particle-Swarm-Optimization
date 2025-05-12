#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GUI for Particle Swarm Optimization (PSO) of Brain Tumor CNN model.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QPushButton, QSpinBox, 
                            QDoubleSpinBox, QGroupBox, QGridLayout, QTabWidget,
                            QTextEdit, QProgressBar, QComboBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# Import PSO and CNN code
from brain_tumor_cnn_pso import (BrainTumorCNN, PSO, prepare_data, evaluate_model, 
                                set_seed, device, IMG_HEIGHT, IMG_WIDTH)
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

class PSOWorker(QThread):
    """Worker thread for running PSO optimization"""
    update_progress = pyqtSignal(int, dict)
    finished = pyqtSignal(tuple)
    
    def __init__(self, params):
        super().__init__()
        self.params = params
        
    def run(self):
        # Create PSO instance with parameters from GUI
        pso = PSO(
            fitness_function=self.fitness_function,
            bounds=self.params['bounds'],
            num_particles=self.params['num_particles'],
            max_iterations=self.params['max_iterations'],
            w=self.params['w'],
            c1=self.params['c1'],
            c2=self.params['c2']
        )
        
        # Override optimize method to report progress
        original_optimize = pso.optimize
        
        def optimize_with_progress():
            for iteration in range(pso.max_iterations):
                # Run one iteration
                for i, particle in enumerate(pso.particles):
                    particle.fitness = pso.fitness_function(particle.position)
                    
                    if particle.fitness > particle.best_fitness:
                        particle.best_fitness = particle.fitness
                        particle.best_position = particle.position.copy()
                    
                    if particle.fitness > pso.global_best_fitness:
                        pso.global_best_fitness = particle.fitness
                        pso.global_best_position = particle.position.copy()
                
                # Update velocities and positions
                for particle in pso.particles:
                    r1 = np.random.random(len(pso.bounds))
                    r2 = np.random.random(len(pso.bounds))
                    
                    cognitive_velocity = pso.c1 * r1 * (particle.best_position - particle.position)
                    social_velocity = pso.c2 * r2 * (pso.global_best_position - particle.position)
                    
                    particle.velocity = pso.w * particle.velocity + cognitive_velocity + social_velocity
                    particle.position = particle.position + particle.velocity
                    
                    # Ensure positions are within bounds
                    for j in range(len(pso.bounds)):
                        if particle.position[j] < pso.bounds[j][0]:
                            particle.position[j] = pso.bounds[j][0]
                        elif particle.position[j] > pso.bounds[j][1]:
                            particle.position[j] = pso.bounds[j][1]
                
                # Calculate average fitness for this iteration
                avg_fitness = np.mean([p.fitness for p in pso.particles])
                
                # Store history
                if not hasattr(pso, 'history'):
                    pso.history = {'global_best_fitness': [], 'avg_fitness': []}
                pso.history['global_best_fitness'].append(pso.global_best_fitness)
                pso.history['avg_fitness'].append(avg_fitness)
                
                # Emit progress signal (iteration number and current history)
                progress_pct = int((iteration + 1) / pso.max_iterations * 100)
                self.update_progress.emit(progress_pct, pso.history)
            
            return pso.global_best_position, pso.global_best_fitness, pso.history
        
        # Replace optimize method
        pso.optimize = optimize_with_progress
        
        # Run optimization
        result = pso.optimize()
        self.finished.emit(result)
    
    def fitness_function(self, params):
        """Fitness function for PSO. Returns validation accuracy."""
        # Extract parameters
        learning_rate = params[0]
        weight_decay = params[1]
        dropout1 = params[2]
        dropout2 = params[3]
        dropout3 = params[4]
        dropout_fc = params[5]
        filters1 = int(params[6])
        filters2 = int(params[7])
        filters3 = int(params[8])
        batch_size = int(params[9])
        
        # Prepare data
        train_loader, val_loader = prepare_data(batch_size)
        
        # Create model with the given parameters
        model = BrainTumorCNN(
            filters1=filters1,
            filters2=filters2,
            filters3=filters3,
            dropout1=dropout1,
            dropout2=dropout2,
            dropout3=dropout3,
            dropout_fc=dropout_fc
        ).to(device)
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
        
        # Evaluate model
        val_acc = evaluate_model(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            scheduler,
            num_epochs=self.params['train_epochs'],
            patience=self.params['patience']
        )
        
        return val_acc

class BrainTumorCNNPSOGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Brain Tumor CNN PSO Optimizer")
        self.setGeometry(100, 100, 1000, 800)
        
        # Main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Create tabs
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)
        
        # Create tab widgets
        self.setup_tab = QWidget()
        self.results_tab = QWidget()
        
        # Add tabs
        self.tabs.addTab(self.setup_tab, "Setup")
        self.tabs.addTab(self.results_tab, "Results")
        
        # Setup the tabs
        self.init_setup_tab()
        self.init_results_tab()
        
        # Initialize worker
        self.pso_worker = None
        
        # Show the GUI
        self.show()
    
    def init_setup_tab(self):
        """Initialize the setup tab with parameter controls"""
        layout = QVBoxLayout(self.setup_tab)
        
        # PSO Parameters Group
        pso_group = QGroupBox("PSO Parameters")
        pso_layout = QGridLayout()
        pso_group.setLayout(pso_layout)
        
        # Number of particles
        pso_layout.addWidget(QLabel("Number of Particles:"), 0, 0)
        self.particles_spin = QSpinBox()
        self.particles_spin.setRange(2, 50)
        self.particles_spin.setValue(5)
        pso_layout.addWidget(self.particles_spin, 0, 1)
        
        # Max iterations
        pso_layout.addWidget(QLabel("Max Iterations:"), 1, 0)
        self.iterations_spin = QSpinBox()
        self.iterations_spin.setRange(1, 100)
        self.iterations_spin.setValue(10)
        pso_layout.addWidget(self.iterations_spin, 1, 1)
        
        # Inertia weight (w)
        pso_layout.addWidget(QLabel("Inertia Weight (w):"), 2, 0)
        self.w_spin = QDoubleSpinBox()
        self.w_spin.setRange(0.1, 1.0)
        self.w_spin.setSingleStep(0.1)
        self.w_spin.setValue(0.7)
        pso_layout.addWidget(self.w_spin, 2, 1)
        
        # Cognitive coefficient (c1)
        pso_layout.addWidget(QLabel("Cognitive Coefficient (c1):"), 3, 0)
        self.c1_spin = QDoubleSpinBox()
        self.c1_spin.setRange(0.1, 3.0)
        self.c1_spin.setSingleStep(0.1)
        self.c1_spin.setValue(1.5)
        pso_layout.addWidget(self.c1_spin, 3, 1)
        
        # Social coefficient (c2)
        pso_layout.addWidget(QLabel("Social Coefficient (c2):"), 4, 0)
        self.c2_spin = QDoubleSpinBox()
        self.c2_spin.setRange(0.1, 3.0)
        self.c2_spin.setSingleStep(0.1)
        self.c2_spin.setValue(1.5)
        pso_layout.addWidget(self.c2_spin, 4, 1)
        
        # Training Parameters Group
        train_group = QGroupBox("Training Parameters")
        train_layout = QGridLayout()
        train_group.setLayout(train_layout)
        
        # Training epochs
        train_layout.addWidget(QLabel("Training Epochs:"), 0, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 50)
        self.epochs_spin.setValue(10)
        train_layout.addWidget(self.epochs_spin, 0, 1)
        
        # Early stopping patience
        train_layout.addWidget(QLabel("Early Stopping Patience:"), 1, 0)
        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(1, 20)
        self.patience_spin.setValue(5)
        train_layout.addWidget(self.patience_spin, 1, 1)
        
        # Parameter Bounds Group
        bounds_group = QGroupBox("Parameter Bounds")
        bounds_layout = QGridLayout()
        bounds_group.setLayout(bounds_layout)
        
        # Learning rate bounds
        bounds_layout.addWidget(QLabel("Learning Rate:"), 0, 0)
        self.lr_min_spin = QDoubleSpinBox()
        self.lr_min_spin.setRange(0.00001, 0.1)
        self.lr_min_spin.setDecimals(5)
        self.lr_min_spin.setSingleStep(0.00001)
        self.lr_min_spin.setValue(0.00001)
        bounds_layout.addWidget(self.lr_min_spin, 0, 1)
        
        self.lr_max_spin = QDoubleSpinBox()
        self.lr_max_spin.setRange(0.00001, 0.1)
        self.lr_max_spin.setDecimals(5)
        self.lr_max_spin.setSingleStep(0.00001)
        self.lr_max_spin.setValue(0.01)
        bounds_layout.addWidget(self.lr_max_spin, 0, 2)
        
        # Weight decay bounds
        bounds_layout.addWidget(QLabel("Weight Decay:"), 1, 0)
        self.wd_min_spin = QDoubleSpinBox()
        self.wd_min_spin.setRange(0.0, 0.01)
        self.wd_min_spin.setDecimals(5)
        self.wd_min_spin.setSingleStep(0.00001)
        self.wd_min_spin.setValue(0.0)
        bounds_layout.addWidget(self.wd_min_spin, 1, 1)
        
        self.wd_max_spin = QDoubleSpinBox()
        self.wd_max_spin.setRange(0.0, 0.01)
        self.wd_max_spin.setDecimals(5)
        self.wd_max_spin.setSingleStep(0.00001)
        self.wd_max_spin.setValue(0.001)
        bounds_layout.addWidget(self.wd_max_spin, 1, 2)
        
        # Batch size options
        bounds_layout.addWidget(QLabel("Batch Size Options:"), 2, 0)
        self.batch_size_combo = QComboBox()
        self.batch_size_combo.addItems(["8", "16", "32", "64"])
        bounds_layout.addWidget(self.batch_size_combo, 2, 1, 1, 2)
        
        # Add groups to layout
        layout.addWidget(pso_group)
        layout.addWidget(train_group)
        layout.addWidget(bounds_group)
        
        # Start button
        self.start_button = QPushButton("Start Optimization")
        self.start_button.clicked.connect(self.start_optimization)
        layout.addWidget(self.start_button)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
    
    def init_results_tab(self):
        """Initialize the results tab with plots and text output"""
        layout = QVBoxLayout(self.results_tab)
        
        # Plots area
        self.figure = plt.figure(figsize=(10, 4))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # Results text area
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        layout.addWidget(self.results_text)
        
        # Save results button
        self.save_button = QPushButton("Save Results")
        self.save_button.clicked.connect(self.save_results)
        self.save_button.setEnabled(False)
        layout.addWidget(self.save_button)
    
    def start_optimization(self):
        """Start the PSO optimization process"""
        # Disable start button
        self.start_button.setEnabled(False)
        
        # Get parameter bounds
        batch_size_option = int(self.batch_size_combo.currentText())
        
        bounds = [
            (self.lr_min_spin.value(), self.lr_max_spin.value()),      # learning_rate
            (self.wd_min_spin.value(), self.wd_max_spin.value()),      # weight_decay
            (0.1, 0.5),                                                # dropout1
            (0.1, 0.5),                                                # dropout2
            (0.1, 0.5),                                                # dropout3
            (0.3, 0.7),                                                # dropout_fc
            (16, 64),                                                  # filters1
            (32, 128),                                                 # filters2
            (64, 256),                                                 # filters3
            (batch_size_option, batch_size_option)                     # batch_size (fixed)
        ]
        
        # Prepare parameters for worker
        params = {
            'bounds': bounds,
            'num_particles': self.particles_spin.value(),
            'max_iterations': self.iterations_spin.value(),
            'w': self.w_spin.value(),
            'c1': self.c1_spin.value(),
            'c2': self.c2_spin.value(),
            'train_epochs': self.epochs_spin.value(),
            'patience': self.patience_spin.value()
        }
        
        # Create and start worker
        self.pso_worker = PSOWorker(params)
        self.pso_worker.update_progress.connect(self.update_progress)
        self.pso_worker.finished.connect(self.optimization_finished)
        self.pso_worker.start()
        
        # Clear results
        self.results_text.clear()
        self.results_text.append("Optimization started...\n")
        
        # Reset progress bar
        self.progress_bar.setValue(0)
    
    def update_progress(self, progress, history):
        """Update progress bar and plots"""
        self.progress_bar.setValue(progress)
        
        # Update plots
        self.figure.clear()
        
        # Plot global best fitness
        ax1 = self.figure.add_subplot(121)
        ax1.plot(history['global_best_fitness'], 'b-', label='Global Best')
        ax1.set_title('Global Best Fitness')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Fitness (Validation Accuracy)')
        ax1.grid(True)
        ax1.legend()
        
        # Plot average fitness
        ax2 = self.figure.add_subplot(122)
        ax2.plot(history['avg_fitness'], 'r-', label='Average')
        ax2.set_title('Average Fitness')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Fitness (Validation Accuracy)')
        ax2.grid(True)
        ax2.legend()
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def optimization_finished(self, result):
        """Handle optimization completion"""
        best_params, best_fitness, history = result
        
        # Enable start button
        self.start_button.setEnabled(True)
        
        # Enable save button
        self.save_button.setEnabled(True)
        
        # Display results
        self.results_text.append("\nPSO Optimization Results:")
        self.results_text.append(f"Best fitness (validation accuracy): {best_fitness:.4f}")
        self.results_text.append("\nBest parameters:")
        self.results_text.append(f"Learning rate: {best_params[0]:.6f}")
        self.results_text.append(f"Weight decay: {best_params[1]:.6f}")
        self.results_text.append(f"Dropout1: {best_params[2]:.2f}")
        self.results_text.append(f"Dropout2: {best_params[3]:.2f}")
        self.results_text.append(f"Dropout3: {best_params[4]:.2f}")
        self.results_text.append(f"Dropout FC: {best_params[5]:.2f}")
        self.results_text.append(f"Filters1: {int(best_params[6])}")
        self.results_text.append(f"Filters2: {int(best_params[7])}")
        self.results_text.append(f"Filters3: {int(best_params[8])}")
        self.results_text.append(f"Batch size: {int(best_params[9])}")
        
        # Store results for saving
        self.best_params = best_params
    
    def save_results(self):
        """Save optimization results"""
        # Save parameters to numpy file
        np.save('best_pso_params.npy', self.best_params)
        
        # Save plot
        self.figure.savefig('pso_optimization_progress.png')
        
        # Notify user
        self.results_text.append("\nResults saved to:")
        self.results_text.append("- best_pso_params.npy")
        self.results_text.append("- pso_optimization_progress.png")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BrainTumorCNNPSOGUI()
    sys.exit(app.exec_())
