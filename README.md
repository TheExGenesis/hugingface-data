# 🤗 Hugging Face Models Explorer

A Streamlit app for exploring and visualizing Hugging Face models by task, downloads, and parameters.

## Features

- **Task Overview**: View maximum downloads and parameters for models across different tasks
- **Model Details**: Explore models within a specific task, sorted by downloads or parameters
- **Downloads vs Parameters**: Visualize the relationship between model downloads and parameters
- **Model Comparison**: Compare specific models side by side

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:

```bash
streamlit run explore-models.py
```

2. The app will load the model data from `models_by_task.csv` if available, or fetch it from the Hugging Face Hub API.

3. Use the sidebar to select tasks of interest.

4. Navigate between tabs to explore different visualizations:
   - **Task Overview**: High-level view of tasks by downloads and parameters
   - **Model Details**: Detailed view of models within a specific task
   - **Download vs Parameters**: Scatter plot showing relationship between downloads and parameters
   - **Compare Models**: Search for and compare specific models

## Data Collection

The app uses data from the Hugging Face Hub API. If the `models_by_task.csv` file doesn't exist, the app will:

1. Fetch the top 1000 models by downloads
2. Identify all available tasks
3. Fetch the top 100 models for each task
4. Combine and save the data to `models_by_task.csv` for future use

This process may take some time when running for the first time.

## Dependencies

- streamlit
- pandas
- plotly
- huggingface_hub
- tqdm
- toolz

## Useful dev things

- `source hf-explorer-env/bin/activate`
- `pip install -r requirements.txt`
- `streamlit run explore-models.py`
