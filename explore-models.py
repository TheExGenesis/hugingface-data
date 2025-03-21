import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional
from toolz import pipe, curry
import os
from functools import reduce
from datetime import datetime, timedelta

# Set page title and layout
st.set_page_config(
    page_title="Hugging Face Models Explorer",
    page_icon="🤗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# App title and description
st.title("🤗 Hugging Face Models Explorer")
st.markdown(
    """
    Explore models from the Hugging Face Hub based on task, downloads, and parameters.
    Select options from the sidebar to customize the visualizations.
    """
)

# Sidebar for controls
st.sidebar.header("Data Options")


@st.cache_data
def load_data():
    """Load the model data from CSV if available, otherwise fetch it"""
    if os.path.exists("models_by_task.csv"):
        df = pd.read_csv("models_by_task.csv")
        # Make sure parameters are numeric
        df["parameters"] = pd.to_numeric(df["parameters"], errors="coerce")

        # Convert created_at to datetime if it exists
        if "created_at" in df.columns:
            df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
        else:
            # If created_at is missing, add a placeholder column with current date
            # This is just for compatibility until the data is refreshed with created_at included
            df["created_at"] = datetime.now()

        return df
    else:
        st.warning(
            "models_by_task.csv not found. Please run the data collection script first."
        )
        from hf import fetch_models_info

        with st.spinner("Fetching models data from Hugging Face Hub..."):
            # Get top 1000 models by downloads
            models_data = fetch_models_info(limit=1000)
            df = pd.DataFrame(models_data)

            # Get available tasks
            task_counts = df.value_counts("tasks")
            available_tasks = task_counts.index.tolist()

            # Get top 100 models for each task
            task_dataframes = []

            for task in available_tasks:
                with st.spinner(f"Fetching top 100 models for task: {task}"):
                    task_models = fetch_models_info(limit=100, task_filter=task)
                    if task_models:
                        task_df = pd.DataFrame(task_models)
                        task_dataframes.append(task_df)

            # Combine all task-specific dataframes
            df_by_task = pd.concat(task_dataframes, ignore_index=True)

            # Remove potential duplicates
            df_by_task = df_by_task.drop_duplicates(subset=["model_id"])

            # Save to CSV for future use
            df_by_task.to_csv("models_by_task.csv", index=False)

            # Make sure parameters are numeric
            df_by_task["parameters"] = pd.to_numeric(
                df_by_task["parameters"], errors="coerce"
            )

            # Convert created_at to datetime
            if "created_at" in df_by_task.columns:
                df_by_task["created_at"] = pd.to_datetime(
                    df_by_task["created_at"], errors="coerce"
                )

            return df_by_task


# Load the data
with st.spinner("Loading data..."):
    df = load_data()

# Get all available tasks
available_tasks = sorted(df["tasks"].dropna().unique().tolist())

# Pre-selected representative tasks across modalities
default_tasks = [
    "text-generation",
    "feature-extraction",
    "automatic-speech-recognition",
    "image-classification",
    "image-text-to-text",
    "text-to-image",
    "translation",
]

# Filter default tasks to only include those available in the loaded data
default_tasks = [task for task in default_tasks if task in available_tasks]

# Sidebar: Task selection
selected_tasks = st.sidebar.multiselect(
    "Select tasks to explore",
    options=available_tasks,
    default=default_tasks,
)

# Date range filter for creation date
st.sidebar.markdown("### Filter by Creation Date")

# Calculate default date range (last 18 months)
current_date = datetime.now()
default_start_date = current_date - timedelta(days=18 * 30)  # Approximately 18 months

# Allow user to select date range
use_date_filter = st.sidebar.checkbox("Filter by creation date", value=True)
if use_date_filter:
    start_date = st.sidebar.date_input(
        "From", value=default_start_date, max_value=current_date
    )
    end_date = st.sidebar.date_input(
        "To", value=current_date, min_value=start_date, max_value=current_date
    )

    # Convert to datetime for filtering
    start_datetime = datetime.combine(start_date, datetime.min.time())
    end_datetime = datetime.combine(end_date, datetime.max.time())

# Filter data based on selected tasks and date range
if selected_tasks:
    filtered_df = df[df["tasks"].isin(selected_tasks)]
else:
    filtered_df = df

# Apply date filter if enabled
if use_date_filter and "created_at" in df.columns:
    date_mask = (filtered_df["created_at"] >= start_datetime) & (
        filtered_df["created_at"] <= end_datetime
    )
    filtered_df = filtered_df[date_mask]

# Display dataset summary
st.sidebar.markdown("### Dataset Summary")
st.sidebar.markdown(f"**Total models:** {len(df)}")
st.sidebar.markdown(f"**Tasks available:** {len(available_tasks)}")
st.sidebar.markdown(f"**Models in selected tasks:** {len(filtered_df)}")
if use_date_filter:
    st.sidebar.markdown(f"**Models in selected date range:** {len(filtered_df)}")

# Tabs for different visualization categories
tab1, tab2, tab3, tab4 = st.tabs(
    ["Task Overview", "Model Details", "Download vs Parameters", "Compare Models"]
)

# Tab 1: Task Overview
with tab1:
    st.header("Task Overview")

    # Plot maximum downloads by task
    def plot_max_downloads_by_task(data: pd.DataFrame) -> go.Figure:
        """Create an interactive bar plot showing the maximum downloads for each task"""
        if data.empty:
            return go.Figure().add_annotation(text="No data available", showarrow=False)

        # Find the maximum downloads per task
        max_downloads_by_task = data.groupby("tasks")["downloads"].max().reset_index()
        max_downloads_by_task = max_downloads_by_task.sort_values(
            "downloads", ascending=False
        )

        # Create interactive bar chart with plotly
        fig = px.bar(
            max_downloads_by_task,
            x="tasks",
            y="downloads",
            title="Maximum Downloads by Task",
            labels={"tasks": "Task", "downloads": "Number of Downloads"},
            color="downloads",
            color_continuous_scale="Viridis",
        )

        # Improve layout
        fig.update_layout(
            xaxis_tickangle=-45,
            height=500,
            hoverlabel=dict(bgcolor="white", font_size=12),
        )

        return fig

    # Plot parameters of top model by task
    def plot_parameters_top_model_by_task(data: pd.DataFrame) -> go.Figure:
        """Create an interactive bar plot showing parameter count of most downloaded model for each task"""
        if data.empty:
            return go.Figure().add_annotation(text="No data available", showarrow=False)

        # First get the model with max downloads for each task
        top_models = data.loc[data.groupby("tasks")["downloads"].idxmax()]
        # Sort by downloads
        top_models = top_models.sort_values("downloads", ascending=False)

        # Add model_id as hover info
        top_models["hover_text"] = top_models.apply(
            lambda x: f"Model: {x['model_id']}<br>Downloads: {x['downloads']:,}", axis=1
        )

        # Create plotly bar chart
        fig = px.bar(
            top_models,
            x="tasks",
            y="parameters",
            title="Parameters of Most Downloaded Model by Task",
            labels={"tasks": "Task", "parameters": "Number of Parameters"},
            color="parameters",
            color_continuous_scale="Viridis",
            hover_data=["model_id", "downloads"],
        )

        # Improve layout
        fig.update_layout(
            xaxis_tickangle=-45,
            height=500,
            hoverlabel=dict(bgcolor="white", font_size=12),
        )

        return fig

    # Plot treemap of downloads by task
    def plot_task_treemap(data: pd.DataFrame) -> go.Figure:
        """Create a treemap visualization of downloads by task"""
        if data.empty:
            return go.Figure().add_annotation(text="No data available", showarrow=False)

        # Aggregate downloads by task
        task_downloads = data.groupby("tasks")[["downloads"]].sum().reset_index()

        # Create treemap
        fig = px.treemap(
            task_downloads,
            path=["tasks"],
            values="downloads",
            title="Distribution of Downloads Across Tasks",
            color="downloads",
            color_continuous_scale="Viridis",
        )

        fig.update_layout(
            height=500,
        )

        return fig

    # Display plots
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(
            plot_max_downloads_by_task(filtered_df), use_container_width=True
        )

    with col2:
        st.plotly_chart(
            plot_parameters_top_model_by_task(filtered_df), use_container_width=True
        )

    st.plotly_chart(plot_task_treemap(filtered_df), use_container_width=True)

# Tab 2: Model Details
with tab2:
    st.header("Model Details by Task")

    if not selected_tasks:
        st.warning("Please select at least one task in the sidebar.")
    else:
        # Task selection for this tab
        selected_task = st.selectbox("Select a task to explore", options=selected_tasks)

        # Control number of models to display
        max_models = st.slider(
            "Number of models to display", min_value=5, max_value=100, value=20, step=5
        )

        # Sort options
        sort_by = st.radio(
            "Sort models by:", options=["Downloads", "Parameters"], horizontal=True
        )

        # Get data for selected task
        task_data = df[df["tasks"] == selected_task]

        # Apply date filter if enabled
        if use_date_filter and "created_at" in df.columns:
            date_mask = (task_data["created_at"] >= start_datetime) & (
                task_data["created_at"] <= end_datetime
            )
            task_data = task_data[date_mask]

        # Sort based on user selection
        if sort_by == "Downloads":
            task_data = task_data.sort_values("downloads", ascending=False)
            y_column = "downloads"
            title = f"Top {max_models} Models by Downloads for Task: {selected_task}"
            y_label = "Number of Downloads"
        else:  # Parameters
            task_data = task_data.dropna(subset=["parameters"])
            task_data = task_data.sort_values("parameters", ascending=False)
            y_column = "parameters"
            title = f"Top {max_models} Models by Parameters for Task: {selected_task}"
            y_label = "Number of Parameters"

        # Limit to max_models
        task_data = task_data.head(max_models)

        # Create bar plot
        fig = px.bar(
            task_data,
            x="model_id",
            y=y_column,
            title=title,
            labels={"model_id": "Model ID", y_column: y_label},
            log_y=True,  # Log scale for better visualization
            hover_data=["downloads", "parameters", "created_at"],
            color="downloads" if sort_by == "Parameters" else "parameters",
            color_continuous_scale="Viridis",
        )

        fig.update_layout(
            xaxis_tickangle=-45,
            height=600,
            xaxis={"categoryorder": "total descending"},
        )

        st.plotly_chart(fig, use_container_width=True)

        # Display data table with model details
        st.subheader(f"Model Data for Task: {selected_task}")
        display_columns = [
            "model_id",
            "downloads",
            "parameters",
            "created_at",
            "license",
            "base_model",
        ]
        # Only include columns that exist in the DataFrame
        display_columns = [col for col in display_columns if col in task_data.columns]

        # Format created_at dates to be more readable if column exists
        display_df = task_data[display_columns].reset_index(drop=True).copy()
        if "created_at" in display_df.columns:
            display_df["created_at"] = display_df["created_at"].dt.strftime("%Y-%m-%d")

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

# Tab 3: Download vs Parameters
with tab3:
    st.header("Downloads vs Parameters")

    # Create scatter plot of downloads vs parameters
    def plot_downloads_vs_parameters(data: pd.DataFrame) -> go.Figure:
        """Create an interactive scatter plot of downloads vs parameters for all models, colored by task"""
        # Filter to only models with parameter data
        valid_data = data.dropna(subset=["parameters"])

        if valid_data.empty:
            return go.Figure().add_annotation(text="No data available", showarrow=False)

        # Create interactive scatter plot
        fig = px.scatter(
            valid_data,
            x="parameters",
            y="downloads",
            color="tasks",
            log_x=True,
            log_y=True,
            hover_data=["model_id"],
            title="Downloads vs. Parameters for All Models",
            labels={
                "parameters": "Number of Parameters (log scale)",
                "downloads": "Number of Downloads (log scale)",
                "tasks": "Task",
            },
            height=700,
        )

        return fig

    st.plotly_chart(plot_downloads_vs_parameters(filtered_df), use_container_width=True)

# Tab 4: Compare Models
with tab4:
    st.header("Compare Specific Models")

    # Model selection
    model_search = st.text_input("Search for models by ID (e.g., 'gpt2', 'bert-base')")

    if model_search:
        matching_models = df[df["model_id"].str.contains(model_search, case=False)]
        if matching_models.empty:
            st.warning(f"No models found matching '{model_search}'")
        else:
            st.success(f"Found {len(matching_models)} matching models")

            selected_models = st.multiselect(
                "Select models to compare", options=matching_models["model_id"].tolist()
            )

            if selected_models:
                # Get data for selected models
                models_data = df[df["model_id"].isin(selected_models)]

                # Create comparison plot
                fig = go.Figure()

                # Add bar for downloads
                fig.add_trace(
                    go.Bar(
                        x=models_data["model_id"],
                        y=models_data["downloads"],
                        name="Downloads",
                        marker_color="royalblue",
                    )
                )

                # Update layout
                fig.update_layout(
                    title="Model Comparison: Downloads",
                    xaxis_tickangle=-45,
                    height=500,
                    yaxis=dict(title="Number of Downloads", type="log"),
                    xaxis=dict(title="Model ID"),
                )

                st.plotly_chart(fig, use_container_width=True)

                # Only show parameters plot if we have parameter data
                if not models_data["parameters"].isna().all():
                    # Create parameters plot
                    fig2 = go.Figure()

                    # Add bar for parameters
                    fig2.add_trace(
                        go.Bar(
                            x=models_data["model_id"],
                            y=models_data["parameters"],
                            name="Parameters",
                            marker_color="green",
                        )
                    )

                    # Update layout
                    fig2.update_layout(
                        title="Model Comparison: Parameters",
                        xaxis_tickangle=-45,
                        height=500,
                        yaxis=dict(title="Number of Parameters", type="log"),
                        xaxis=dict(title="Model ID"),
                    )

                    st.plotly_chart(fig2, use_container_width=True)

                # Show detailed data table
                st.subheader("Detailed Model Comparison")
                display_columns = [
                    "model_id",
                    "downloads",
                    "parameters",
                    "created_at",
                    "tasks",
                    "license",
                    "base_model",
                ]

                # Only include columns that exist in the DataFrame
                display_columns = [
                    col for col in display_columns if col in models_data.columns
                ]

                # Format created_at dates to be more readable if column exists
                display_df = models_data[display_columns].reset_index(drop=True).copy()
                if "created_at" in display_df.columns:
                    display_df["created_at"] = display_df["created_at"].dt.strftime(
                        "%Y-%m-%d"
                    )

                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True,
                )

# Footer
st.markdown("---")
st.markdown("**Data Source:** Hugging Face Hub API")
st.markdown("**Note:** Parameter information may be missing for some models.")
