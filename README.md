# DDoS_Attack_Detection
🛡️ Advanced DDoS Attack Detection Framework | 8 ML Models Compared | 99.98% Accuracy | CICIDS2017 Dataset Analysis 🔍 | Supply Chain Security Integration 🔒 | Python Implementation

# Machine Learning-Based DDoS Attack Detection Framework

## Overview
This project implements and evaluates various machine learning algorithms for detecting Distributed Denial of Service (DDoS) attacks in network traffic. The framework analyzes network behavior patterns using the CICIDS2017 dataset and employs eight different classification algorithms to identify potential DDoS attacks with high accuracy.

## Key Features
- Implementation of 8 different machine learning models:
  - Random Forest
  - Decision Tree
  - Convolutional Neural Network (CNN)
  - Artificial Neural Network (ANN)
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
  - Logistic Regression
  - Naive Bayes

- Comprehensive performance evaluation using multiple metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - ROC Curves
  - Processing Time

- Novel time-weighted performance metric for real-world applicability assessment
- Feature engineering and selection of 24 critical network parameters
- Integration capabilities with existing IDS/IPS systems

## Results
| Model | Accuracy (%) | Precision (%) | Recall (%) | F1 (%) | Time (s) |
|-------|-------------|---------------|------------|---------|----------|
| LR    | 99.86       | 99.87         | 99.88      | 99.87   | 17.919   |
| NB    | 86.11       | 85.00         | 85.00      | 85.00   | 0.400    |
| ANN   | 98.35       | 98.00         | 98.00      | 98.00   | 100.03   |
| CNN   | 99.67       | 99.66         | 99.67      | 99.66   | 218.35   |
| DT    | 99.98       | 100.0         | 100.0      | 100.0   | 7.801    |
| RF    | 99.98       | 99.00         | 99.00      | 99.00   | 210.48   |
| SVM   | 99.10       | 100.0         | 100.0      | 100.0   | 98.670   |
| KNN   | 99.10       | 99.10         | 99.10      | 99.10   | 147.89   |

## Technical Architecture
- Data Preprocessing Pipeline
  - Feature selection and engineering
  - Data cleaning and normalization
  - Class balancing
- Model Training and Evaluation Framework
- Performance Metrics Calculation
- Visualization Components

## Dataset
The project utilizes the CICIDS2017 dataset, which contains:
- Benign network traffic
- Various types of DDoS attacks
- Complete network configuration
- Full packet captures
- Labeled data for supervised learning

## Implementation Details
- Python-based implementation
- Key libraries: TensorFlow, Keras, scikit-learn
- Feature engineering focusing on 24 critical network parameters
- Cross-validation for model evaluation
- Confusion matrix generation for detailed analysis

## Practical Applications
- Network security enhancement
- Integration with existing IDS/IPS systems
- Supply chain cybersecurity
- Real-time threat detection
- Network traffic analysis

## Future Work
- Model optimization and hyperparameter tuning
- Real-time implementation testing
- Feature importance analysis
- Ensemble method exploration
- Integration with SDN environments

## Requirements
- Python 3.8+
- TensorFlow 2.x
- scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn

## Setup and Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/ddos-detection-framework.git

# Install required packages
pip install -r requirements.txt

# Run the main script
python main.py
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contributors
- Laela Olsen
- Jin Heo
- Yesha Modi
- Nikhil Swami
- Advaith Venkatsubramanian
- Vignesh Mohana Velu
- Anchala Balaraj

## Citation
If you use this work in your research, please cite:
```
@article{olsen2024ddos,
  title={Applications of Machine Learning to DDoS Attack Detection and Prevention},
  author={Olsen, Laela and others},
  year={2024},
  institution={Arizona State University}
}
```
