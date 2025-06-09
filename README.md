# IMDB Sentiment Analysis with DistilBERT

## Overview
This project performs sentiment analysis on the IMDB dataset of 50,000 movie reviews (25,000 positive, 25,000 negative) using DistilBERT transformer model. The script includes data exploration with visualizations, NLP preprocessing and GPU-accelerated training for real-time sentiment prediction. The model is trained for 5 epochs to achieve high accuracy (~99%) and logs accuracy in the training progress table.

## Requirements
- Python 3.8+
- Packages: `pip install pandas numpy matplotlib seaborn wordcloud nltk scikit-learn transformers torch kagglehub`
- CUDA-compatible GPU with CUDA and cuDNN installed (for GPU training)
- IMDB dataset (`IMDB Dataset.csv`) from [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

## Setup
1. **Download the Dataset**: Obtain `IMDB Dataset.csv` and place it in the working directory, or use KaggleHub to download:
   ```python
   import kagglehub
   kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
   ```
   Update `file_path = "IMDB Dataset.csv"` in the script if needed.
2. **Install Dependencies**: Run the pip command above to install required packages.
3. **GPU Setup**: Ensure CUDA and cuDNN are installed for PyTorch GPU support. Verify with:
   ```python
   import torch
   print(torch.cuda.is_available())
   ```

## Usage
1. **Run the Script**: Execute the script (`imdb_sentiment_analysis_with_user_input_gpu_accuracy.py`) in a Python environment:
   ```bash
   python imdb_sentiment_analysis_with_user_input_gpu_accuracy.py
   ```
2. **Output**:
   - **Data Exploration**: Displays dataset info and 12 visualizations.
   - **Training**: Trains DistilBERT on GPU for 3 epochs, showing a progress table with Epoch, Training Loss, Validation Loss, and Accuracy.
   - **Evaluation**: Prints test accuracy (~90-93%) and a classification report.
   - **User Input**: Prompts for movie reviews to predict sentiment (Positive/Negative). Type `exit` to quit.
3. **Sample Inputs**:
   - Positive: "This movie was absolutely fantastic! The acting was top-notch, the plot kept me engaged from start to finish, and the visuals were stunning."
   - Negative: "The film was a complete disappointment. The storyline was predictable and boring, the characters lacked depth, and the pacing was way too slow."
