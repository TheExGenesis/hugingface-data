# %%
from huggingface_hub import HfApi
import pandas as pd
from tqdm import tqdm
from toolz import pipe, curry, concat
from typing import List, Dict, Any, Optional, Tuple
from functools import reduce

api = HfApi()


# %%
# Functional approach to get top models by downloads
def fetch_models_info(
    limit: Optional[int] = None, task_filter: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Fetch models and their metadata, limited to specified count if provided

    Args:
        limit: Maximum number of models to fetch
        task_filter: Filter models by specific task (e.g., 'text-generation', 'sentence-similarity')
    """
    # Apply task filter if provided
    filter_params = task_filter
    models_list = api.list_models(
        limit=limit, sort="downloads", direction=-1, filter=filter_params
    )

    data = []
    try:
        for model in tqdm(models_list, desc=f"Task: {task_filter or 'all'}"):
            try:
                info = api.model_info(model.modelId)
                # Safely extract attributes with fallbacks to None for missing attributes
                model_data = {
                    "model_id": model.modelId,
                    "downloads": getattr(
                        info, "downloads", 0
                    ),  # Default to 0 for sorting
                    "tasks": getattr(info, "pipeline_tag", None),
                    "parameters": (
                        getattr(info.safetensors, "total", None)
                        if hasattr(info, "safetensors") and info.safetensors
                        else None
                    ),
                    "tags": getattr(info, "tags", None),
                    "created_at": getattr(info, "created_at", None),
                }

                # Extract card_data attributes safely
                if hasattr(info, "card_data") and info.card_data:
                    model_data.update(
                        {
                            "license": getattr(info.card_data, "license", None),
                            "datasets": getattr(info.card_data, "datasets", None),
                            "model_name": getattr(info.card_data, "model_name", None),
                            "metrics": getattr(info.card_data, "metrics", None),
                            "base_model": getattr(info.card_data, "base_model", None),
                            "languages": getattr(info.card_data, "language", None),
                        }
                    )
                else:
                    model_data.update(
                        {
                            "license": None,
                            "datasets": None,
                            "model_name": None,
                            "metrics": None,
                            "base_model": None,
                            "languages": None,
                        }
                    )

                data.append(model_data)
            except Exception as e:
                print(f"Error processing model {model.modelId}: {e}")
    except KeyboardInterrupt:
        print("Operation interrupted by user. Returning partial results.")

    return data


# %%
# Execute the pipeline to get top 1000 models by downloads
models_data = fetch_models_info(limit=1000)

# Create DataFrame
df = pd.DataFrame(models_data)
# %%
task_counts = df.value_counts("tasks")
print("\nModels per task category:")
print(task_counts)
available_tasks = task_counts.index.tolist()

# Get top 100 models for each task
task_dataframes = []

from concurrent.futures import ThreadPoolExecutor
from functools import partial


def fetch_task_models(task):
    print(f"\nFetching top 100 models for task: {task}")
    task_models = fetch_models_info(limit=100, task_filter=task)
    if task_models and len(task_models) > 0:  # Check if list exists and is not empty
        task_df = pd.DataFrame(task_models)
        print(f"Found {len(task_models)} models for task: {task}")
        return task_df
    return None


# Use ThreadPoolExecutor to run fetches in parallel with retry logic
from toolz import curry
from time import sleep


@curry
def fetch_with_retry(task, max_retries=3, backoff_factor=2):
    """Fetch task models with exponential backoff retry logic"""
    for attempt in range(max_retries):
        try:
            return fetch_task_models(task)
        except Exception as e:
            wait_time = backoff_factor**attempt
            print(
                f"Error fetching task {task} (attempt {attempt+1}/{max_retries}): {e}"
            )
            print(f"Retrying in {wait_time} seconds...")
            if attempt < max_retries - 1:  # Don't sleep after the last attempt
                sleep(wait_time)
            else:
                print(
                    f"Failed to fetch models for task {task} after {max_retries} attempts"
                )
    return None


with ThreadPoolExecutor(max_workers=min(5, len(available_tasks))) as executor:
    # Reduced max_workers to 5 to avoid overwhelming the API
    results = list(executor.map(fetch_with_retry, available_tasks))
    # Filter out None values explicitly
    task_dataframes.extend([df for df in results if df is not None])
# %%
# Combine all task-specific dataframes
df_by_task = pd.concat(task_dataframes, ignore_index=True)

# Remove potential duplicates (a model might be in multiple task categories)
df_by_task = df_by_task.drop_duplicates(subset=["model_id"])

df_by_task.to_csv("models_by_task.csv", index=False)

# %%
