"""
Weight Generator: Instance-specific fusion using context features and a small MLP
"""
import numpy as np

class WeightGenerator:
    """
    A small neural network (implemented in NumPy) to generate dynamic weights
    based on user-movie context features.
    """
    def __init__(self, input_size=4, hidden_size=8, output_size=3):
        # Initialize weights randomly with fixed seed for reproducibility
        np.random.seed(42)
        
        # Layer 1: Input to Hidden
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros(hidden_size)
        
        # Layer 2: Hidden to Output
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros(output_size)
        
    def _softmax(self, z):
        """Standard Softmax implementation for weight normalization"""
        exp_z = np.exp(z - np.max(z)) # Stability shift
        return exp_z / exp_z.sum()

    def _relu(self, x):
        """ReLU Activation"""
        return np.maximum(0, x)

    def extract_features(self, user_id, movie_id, dataset_stats):
        """
        Extract normalized context features x = [u_density, m_pop, u_avg, m_avg]
        
        Args:
            user_id, movie_id: IDs
            dataset_stats: dict containing counts and averages
        """
        cnt_u = dataset_stats['user_counts'].get(user_id, 0)
        cnt_i = dataset_stats['movie_counts'].get(movie_id, 0)
        avg_u = dataset_stats['user_avgs'].get(user_id, 3.5)
        avg_i = dataset_stats['movie_avgs'].get(movie_id, 3.5)
        
        # Normalize (assuming some realistic max values)
        max_u_count = dataset_stats.get('max_u_count', 500)
        max_i_count = dataset_stats.get('max_i_count', 1000)
        
        u_density = min(cnt_u / max_u_count, 1.0)
        m_pop = min(cnt_i / max_i_count, 1.0)
        u_avg_norm = avg_u / 5.0
        m_avg_norm = avg_i / 5.0
        
        return np.array([u_density, m_pop, u_avg_norm, m_avg_norm])

    def forward(self, x):
        """
        Forward pass through the MLP
        x -> Layer 1 -> ReLU -> Layer 2 -> Softmax -> Weights
        """
        # Save x for backward pass
        self.last_x = x
        
        # Hidden layer
        self.last_z1 = np.dot(x, self.W1) + self.b1
        self.last_h = self._relu(self.last_z1)
        
        # Output layer (Raw scores z)
        self.last_z2 = np.dot(self.last_h, self.W2) + self.b2
        
        # Weights (Softmax)
        self.last_w = self._softmax(self.last_z2)
        
        return self.last_w

    def backward(self, dL_dw, learning_rate=0.01):
        """
        Backpropagation through Softmax and MLP
        dL_dw: Gradient of loss wrt output weights (size 3)
        """
        # 1. Gradient of Softmax: dL/dz2 = w * (dL/dw - sum(dL/dw * w))
        # This is the simplified chain rule for softmax
        w = self.last_w
        grad_z2 = w * (dL_dw - np.dot(dL_dw, w))
        
        # 2. Gradient of Linear Layer 2: dL/dW2 = h.T * grad_z2
        dL_dW2 = np.outer(self.last_h, grad_z2)
        dL_db2 = grad_z2
        
        # 3. Gradient of ReLU: dL/dz1 = (W2 * grad_z2) * [z1 > 0]
        grad_h = np.dot(self.W2, grad_z2)
        grad_z1 = grad_h * (self.last_z1 > 0)
        
        # 4. Gradient of Linear Layer 1: dL/dW1 = x.T * grad_z1
        dL_dW1 = np.outer(self.last_x, grad_z1)
        dL_db1 = grad_z1
        
        # Update Weights (SGD)
        self.W1 -= learning_rate * dL_dW1
        self.b1 -= learning_rate * dL_db1
        self.W2 -= learning_rate * dL_dW2
        self.b2 -= learning_rate * dL_db2

    def generate_weights(self, user_id, movie_id, dataset_stats):
        """One-stop Shop for Weights"""
        x = self.extract_features(user_id, movie_id, dataset_stats)
        return self.forward(x)
