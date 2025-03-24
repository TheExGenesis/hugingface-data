import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional
from toolz import pipe, curry
import os
from functools import reduce
from datetime import datetime, timedelta
import pytz
import numpy as np

# Set page title and layout
st.set_page_config(
    page_title="Hugging Face Models Explorer",
    page_icon="🤗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# App title and description
st.title("🤗 Hugging Face Models Explorer")
st.markdown(
    """
    Explore models from the Hugging Face Hub based on task, downloads, and parameters. The tasks are not exhaustive, I just took the tasks present in top 1000 downloaded models.
    Select options from the sidebar to customize the visualizations.
    """
)


# Helper function to format large numbers
def format_big_number(num: float) -> str:
    """Format large numbers in a human-readable way (K for thousands, M for millions, B for billions)"""
    if num is None or pd.isna(num):
        return "N/A"

    abs_num = abs(num)
    sign = "-" if num < 0 else ""

    if abs_num >= 1_000_000_000:
        return f"{sign}{abs_num / 1_000_000_000:.1f}B"
    elif abs_num >= 1_000_000:
        return f"{sign}{abs_num / 1_000_000:.1f}M"
    elif abs_num >= 1_000:
        return f"{sign}{abs_num / 1_000:.1f}K"
    else:
        return f"{sign}{abs_num:.0f}"


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
    "text-to-image",
    "text-to-speech",
    "text-to-audio",
    "text-to-video",
    "text-to-3d",
    "image-text-to-text",
    "image-to-image",
    "image-to-3d",
    "image-to-video",
    "automatic-speech-recognition",
    "image-segmentation",
    "depth-estimation",
    # "sentence-similarity",
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
    # Convert timezone-naive datetime to UTC timezone to match DataFrame
    start_datetime = pytz.UTC.localize(start_datetime)
    end_datetime = pytz.UTC.localize(end_datetime)

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
tab1, tab2, tab3 = st.tabs(
    ["Overview of Tasks", "Models per Task", "Download vs Parameters"]
)

# Tab 1: Task Overview
with tab1:
    st.header("Overview of Tasks")

    # Plot maximum downloads by task
    def plot_max_downloads_by_task(data: pd.DataFrame) -> go.Figure:
        """Create an interactive bar plot showing the maximum downloads for each task"""
        if data.empty:
            return go.Figure().add_annotation(text="No data available", showarrow=False)

        # Find the maximum downloads per task along with the corresponding model data
        task_groups = data.groupby("tasks")
        max_downloads_models = []

        for task, group in task_groups:
            # Get the model with max downloads
            if not group.empty:
                max_model = group.loc[group["downloads"].idxmax()]
                max_downloads_models.append(max_model)

        # Convert to DataFrame
        max_downloads_by_task = pd.DataFrame(max_downloads_models)
        max_downloads_by_task = max_downloads_by_task.sort_values(
            "downloads", ascending=False
        )

        # Add days_since_creation for coloring if created_at exists
        if "created_at" in max_downloads_by_task.columns:
            max_downloads_by_task["days_since_creation"] = (
                datetime.now(pytz.UTC) - max_downloads_by_task["created_at"]
            ).dt.days
        else:
            max_downloads_by_task["days_since_creation"] = 0

        # Create interactive bar chart with plotly
        fig = px.bar(
            max_downloads_by_task,
            x="tasks",
            y="downloads",
            title="Maximum Downloads by Task",
            labels={"tasks": "Task", "downloads": "Downloads (log)"},
            color="days_since_creation",
            color_continuous_scale="Viridis_r",  # Reversed so newer models are brighter
            color_continuous_midpoint=365,  # Midpoint at 1 year
            log_y=True,  # Use log scale for downloads
            hover_data=["model_id"],
        )

        # Improve layout
        fig.update_layout(
            xaxis_tickangle=-45,
            height=500,
            hoverlabel=dict(bgcolor="white", font_size=12),
            coloraxis_colorbar=dict(
                title="Model Age",
                tickvals=[0, 182, 365, 730, 1095],
                ticktext=["New", "6 mo", "1 year", "2 years", "3 years"],
            ),
        )

        # Format y-axis tick labels with B instead of G for billions
        # Use tickmode manual and define explicit ticks with formatted values
        max_val = max_downloads_by_task["downloads"].max()
        magnitude = 0
        while max_val >= 10:
            max_val /= 10
            magnitude += 1

        # Create tick points based on the magnitude of the data
        base = 10 ** (
            magnitude // 3 * 3
        )  # Find nearest thousand, million, billion, etc.
        tick_vals = []
        tick_texts = []

        for i in range(0, magnitude + 4, 3):
            val = 10**i
            if val <= 10 ** (magnitude + 3):
                tick_vals.append(val)
                if i == 0:
                    tick_texts.append("1")
                elif i == 3:
                    tick_texts.append("1K")
                elif i == 6:
                    tick_texts.append("1M")
                elif i == 9:
                    tick_texts.append("1B")
                elif i == 12:
                    tick_texts.append("1T")

        fig.update_yaxes(tickmode="array", tickvals=tick_vals, ticktext=tick_texts)

        # Update hover template to include formatted numbers and creation date
        hover_template = "<b>%{x}</b><br>"
        hover_template += "Downloads: %{y:,.0f}<br>"
        hover_template += "Model: %{customdata[0]}"
        if "created_at" in max_downloads_by_task.columns:
            hover_template += "<br>Age: %{marker.color:.0f} days"
        hover_template += "<extra></extra>"

        fig.update_traces(hovertemplate=hover_template)

        return fig

    # Plot parameters of top model by task
    def plot_parameters_top_model_by_task(data: pd.DataFrame) -> go.Figure:
        """Create an interactive bar plot showing parameter count of most downloaded model for each task"""
        if data.empty:
            return go.Figure().add_annotation(text="No data available", showarrow=False)

        # Group models by task, then for each task find the most downloaded model that has parameters data
        task_groups = data.groupby("tasks")
        top_models_list = []

        for task, group in task_groups:
            # Filter to models with parameter data
            models_with_params = group.dropna(subset=["parameters"])
            if not models_with_params.empty:
                # Get the most downloaded model with parameters
                top_model = models_with_params.loc[
                    models_with_params["downloads"].idxmax()
                ]
                top_models_list.append(top_model)

        # Convert list of Series to DataFrame
        if top_models_list:
            top_models = pd.DataFrame(top_models_list)
            # Sort by parameters (high to low)
            top_models = top_models.sort_values("parameters", ascending=False)

            # Add days_since_creation for coloring if created_at exists
            if "created_at" in top_models.columns:
                top_models["days_since_creation"] = (
                    datetime.now(pytz.UTC) - top_models["created_at"]
                ).dt.days
            else:
                top_models["days_since_creation"] = 0

            # Create plotly bar chart
            fig = px.bar(
                top_models,
                x="tasks",
                y="parameters",
                title="Parameters of Most Downloaded Model by Task (with parameter data)",
                labels={
                    "tasks": "Task",
                    "parameters": "Parameters (log)",
                },
                color="days_since_creation",
                color_continuous_scale="Viridis_r",  # Reversed so newer models are brighter
                color_continuous_midpoint=365,  # Midpoint at 1 year
                hover_data=["model_id", "downloads"],
                log_y=True,  # Use log scale for parameters
            )

            # Improve layout
            fig.update_layout(
                xaxis_tickangle=-45,
                height=500,
                hoverlabel=dict(bgcolor="white", font_size=12),
                coloraxis_colorbar=dict(
                    title="Model Age",
                    tickvals=[0, 182, 365, 730, 1095],
                    ticktext=["New", "6 mo", "1 year", "2 years", "3 years"],
                ),
            )

            # Format y-axis tick labels with B instead of G for billions
            # Use tickmode manual and define explicit ticks with formatted values
            max_val = top_models["parameters"].max()
            magnitude = 0
            while max_val >= 10:
                max_val /= 10
                magnitude += 1

            # Create tick points based on the magnitude of the data
            tick_vals = []
            tick_texts = []

            for i in range(0, magnitude + 4, 3):
                val = 10**i
                if val <= 10 ** (magnitude + 3):
                    tick_vals.append(val)
                    if i == 0:
                        tick_texts.append("1")
                    elif i == 3:
                        tick_texts.append("1K")
                    elif i == 6:
                        tick_texts.append("1M")
                    elif i == 9:
                        tick_texts.append("1B")
                    elif i == 12:
                        tick_texts.append("1T")

            fig.update_yaxes(tickmode="array", tickvals=tick_vals, ticktext=tick_texts)

            # Update hover template to show formatted numbers
            hover_template = "<b>%{x}</b><br>"
            hover_template += "Parameters: %{y:,.0f}<br>"
            hover_template += "Model: %{customdata[0]}<br>"
            hover_template += "Downloads: %{customdata[1]:,.0f}"
            if "created_at" in top_models.columns:
                hover_template += "<br>Age: %{marker.color:.0f} days"
            hover_template += "<extra></extra>"

            fig.update_traces(hovertemplate=hover_template)

            return fig
        else:
            return go.Figure().add_annotation(
                text="No models with parameter data available", showarrow=False
            )

    # Plot treemap of downloads by task
    def plot_task_treemap(data: pd.DataFrame) -> go.Figure:
        """Create a treemap visualization of downloads by task"""
        if data.empty:
            return go.Figure().add_annotation(text="No data available", showarrow=False)

        # Create a copy to avoid modifying the original dataframe
        data_copy = data.copy()

        # Calculate days_since_creation if created_at exists
        if "created_at" in data_copy.columns:
            data_copy["days_since_creation"] = (
                datetime.now(pytz.UTC) - data_copy["created_at"]
            ).dt.days
        else:
            data_copy["days_since_creation"] = 0

        # Aggregate downloads by task
        task_downloads = data_copy.groupby("tasks")[["downloads"]].sum().reset_index()

        # Calculate average age of models by task
        task_avg_age = (
            data_copy.groupby("tasks")[["days_since_creation"]].mean().reset_index()
        )

        # Merge the downloads and age data
        task_data = pd.merge(task_downloads, task_avg_age, on="tasks")

        # Create treemap
        fig = px.treemap(
            task_data,
            path=["tasks"],
            values="downloads",
            title="Distribution of Downloads Across Tasks (Color: Avg Model Age)",
            color="days_since_creation",
            color_continuous_scale="Viridis_r",  # Reversed so newer models are brighter
            color_continuous_midpoint=365,  # Midpoint at 1 year
        )

        fig.update_layout(
            height=500,
            coloraxis_colorbar=dict(
                title="Avg Model Age",
                tickvals=[0, 182, 365, 730, 1095],
                ticktext=["New", "6 mo", "1 year", "2 years", "3 years"],
            ),
        )

        # Update hover template to show formatted numbers with custom format
        hover_template = "<b>%{label}</b><br>"
        hover_template += "Downloads: %{value:,.0f}<br>"
        hover_template += "Avg Age: %{color:.0f} days"
        hover_template += "<extra></extra>"

        fig.update_traces(hovertemplate=hover_template)

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

    # Display table of all eligible models
    st.subheader("All Models in Selected Tasks")

    # Define columns to display
    display_columns = [
        "model_id",
        "tasks",
        "downloads",
        "parameters",
        "created_at",
        "license",
        "base_model",
    ]
    # Only include columns that exist in the DataFrame
    display_columns = [col for col in display_columns if col in filtered_df.columns]

    # Format numbers in the dataframe
    display_df = filtered_df[display_columns].reset_index(drop=True).copy()
    if "created_at" in display_df.columns:
        display_df["created_at"] = display_df["created_at"].dt.strftime("%Y-%m-%d")
    if "downloads" in display_df.columns:
        display_df["downloads_formatted"] = display_df["downloads"].apply(
            format_big_number
        )
        # Reorder columns to put downloads_formatted right after downloads
        cols = display_df.columns.tolist()
        download_idx = cols.index("downloads")
        cols.insert(download_idx + 1, cols.pop(cols.index("downloads_formatted")))
        display_df = display_df[cols]
    if "parameters" in display_df.columns:
        display_df["parameters_formatted"] = display_df["parameters"].apply(
            format_big_number
        )
        # Reorder columns to put parameters_formatted right after parameters
        cols = display_df.columns.tolist()
        params_idx = cols.index("parameters")
        cols.insert(params_idx + 1, cols.pop(cols.index("parameters_formatted")))
        display_df = display_df[cols]

    # Sort by downloads (descending)
    display_df = display_df.sort_values("downloads", ascending=False)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

# Tab 2: Model Details
with tab2:
    st.header("Model Details by Task")
    st.write(
        "This tab allows you to explore model details for each task. You can view the top models by downloads or parameters, and see their details in a table."
    )
    st.warning(
        "Note: Some models don't have parameter data available. These models are shown as N/A in the parameters column."
    )

    if not selected_tasks:
        st.warning("Please select at least one task in the sidebar.")
    else:
        # Control number of models to display
        max_models = st.slider(
            "Number of models to display per task",
            min_value=5,
            max_value=50,
            value=10,
            step=5,
        )

        # Calculate global max values for consistent scaling
        global_max_downloads = filtered_df["downloads"].max()
        global_min_downloads = filtered_df["downloads"].min()
        global_max_params = filtered_df.dropna(subset=["parameters"])[
            "parameters"
        ].max()
        global_min_params = filtered_df.dropna(subset=["parameters"])[
            "parameters"
        ].min()

        # Helper function to create plot for a single task
        def create_task_plot(
            task_name: str,
            task_data: pd.DataFrame,
            sort_column: str,
            max_models_to_show: int,
            global_min: float,
            global_max: float,
        ) -> go.Figure:
            """Create a bar chart for a specific task"""
            # Apply date filter if enabled
            if use_date_filter and "created_at" in task_data.columns:
                date_mask = (task_data["created_at"] >= start_datetime) & (
                    task_data["created_at"] <= end_datetime
                )
                task_data = task_data[date_mask]

            # Handle empty data case
            if task_data.empty:
                fig = go.Figure()
                fig.add_annotation(
                    text=f"No data available for task: {task_name}", showarrow=False
                )
                return fig

            # Sort based on user selection
            if sort_column == "downloads":
                task_data = task_data.sort_values("downloads", ascending=False).head(
                    max_models_to_show
                )
                # Limit to max_models
                y_column = "downloads"
                title = f"Top {max_models_to_show} Models by Downloads for Task: {task_name}"
                y_label = "Number of Downloads (log scale)"

            else:  # Parameters
                # task_data = task_data.dropna(subset=["parameters"])
                if task_data.empty:
                    fig = go.Figure()
                    fig.add_annotation(
                        text=f"No parameter data available for task: {task_name}",
                        showarrow=False,
                    )
                    return fig

                task_data = (
                    task_data.sort_values("downloads", ascending=False)
                    .head(max_models_to_show)
                    .sort_values("parameters", ascending=False)
                )
                y_column = "parameters"
                title = f"Top {max_models_to_show} Models by Parameters for Task: {task_name}"
                y_label = "Number of Parameters (log scale)"

            # Add days_since_creation for coloring if created_at exists
            if "created_at" in task_data.columns:
                task_data["days_since_creation"] = (
                    datetime.now(pytz.UTC) - task_data["created_at"]
                ).dt.days
            else:
                task_data["days_since_creation"] = 0

            # Create custom data for hover - add formatted versions of values
            task_data_copy = task_data.copy()
            hover_data = []

            for _, row in task_data.iterrows():
                downloads = (
                    row["downloads"]
                    if "downloads" in row and not pd.isna(row["downloads"])
                    else 0
                )
                parameters = (
                    row["parameters"]
                    if "parameters" in row and not pd.isna(row["parameters"])
                    else 0
                )
                created_at = (
                    row["created_at"].strftime("%Y-%m-%d")
                    if "created_at" in row and not pd.isna(row["created_at"])
                    else "N/A"
                )
                days_age = (
                    row["days_since_creation"]
                    if "days_since_creation" in row
                    and not pd.isna(row["days_since_creation"])
                    else 0
                )

                # Format values for display
                downloads_formatted = format_big_number(downloads)
                parameters_formatted = format_big_number(parameters)

                hover_data.append(
                    [
                        downloads,  # Original downloads for sorting
                        parameters,  # Original parameters for sorting
                        created_at,  # Formatted date
                        downloads_formatted,  # Pre-formatted downloads
                        parameters_formatted,  # Pre-formatted parameters
                        int(days_age) if not pd.isna(days_age) else 0,  # Integer days
                    ]
                )

            # Create bar plot with all the desired options
            fig = px.bar(
                task_data,
                x="model_id",
                y=y_column,
                title=title,
                labels={
                    "model_id": "Model ID",
                    y_column: y_label,
                    "days_since_creation": "Model Age (days)",
                },
                log_y=True,  # Log scale for better visualization
                # Color by creation date if available
                color=(
                    "days_since_creation"
                    if "days_since_creation" in task_data.columns
                    else None
                ),
                color_continuous_scale="Viridis_r",  # Reversed Viridis (newer models are brighter)
                color_continuous_midpoint=365,  # Midpoint at 1 year
            )

            # Update y-axis range to ensure consistent scaling across all plots
            fig.update_yaxes(
                range=[np.log10(max(global_min, 1)), np.log10(global_max * 1.05)]
            )

            # Update color axis title
            if "days_since_creation" in task_data.columns:
                fig.update_layout(
                    coloraxis_colorbar=dict(
                        title="Model Age",
                        tickvals=[0, 182, 365, 730, 1095],
                        ticktext=["New", "6 mo", "1 year", "2 years", "3 years"],
                    )
                )

            # Set layout options
            fig.update_layout(
                xaxis_tickangle=-45,
                height=500,
                xaxis={"categoryorder": "total descending"},
            )

            # Set the custom hover data
            fig.update_traces(customdata=hover_data)

            # Create a simple hover template with pre-formatted values
            hover_template = "<b>%{x}</b><br>"

            if y_column == "downloads":
                hover_template += f"<b>Downloads:</b> %{{customdata[3]}}<br>"
                hover_template += f"<b>Parameters:</b> %{{customdata[4]}}<br>"
            else:
                hover_template += f"<b>Parameters:</b> %{{customdata[4]}}<br>"
                hover_template += f"<b>Downloads:</b> %{{customdata[3]}}<br>"

            hover_template += "<b>Created:</b> %{customdata[2]}"

            # Add model age if available
            if "days_since_creation" in task_data.columns:
                hover_template += "<br><b>Model Age:</b> %{customdata[5]} days"

            hover_template += "<extra></extra>"
            fig.update_traces(hovertemplate=hover_template)

            return fig

        # Display plots for each task side by side
        for task in selected_tasks:
            # Add a task header
            st.subheader(f"Task: {task}")

            # Get data for the task
            task_data = df[df["tasks"] == task]

            # Create columns for downloads and parameters
            col1, col2 = st.columns(2)

            # Show downloads chart in the left column
            with col1:
                downloads_fig = create_task_plot(
                    task,
                    task_data,
                    "downloads",
                    max_models,
                    global_min_downloads,
                    global_max_downloads,
                )
                st.plotly_chart(downloads_fig, use_container_width=True)

            # Show parameters chart in the right column
            with col2:
                parameters_fig = create_task_plot(
                    task,
                    task_data,
                    "parameters",
                    max_models,
                    global_min_params,
                    global_max_params,
                )
                st.plotly_chart(parameters_fig, use_container_width=True)

            # Add a separator between tasks
            if task != selected_tasks[-1]:
                st.markdown("---")

# Tab 3: Download vs Parameters
with tab3:
    st.header("Downloads vs Parameters")

    # Create scatter plot of downloads vs parameters
    def plot_downloads_vs_parameters(data: pd.DataFrame) -> go.Figure:
        """Create an interactive scatter plot of downloads vs parameters for all models, colored by task"""
        # Filter to only models with parameter data and ensure numeric values
        valid_data = data.dropna(subset=["parameters", "downloads"]).copy()

        # Ensure parameters and downloads are numeric
        valid_data["parameters"] = pd.to_numeric(
            valid_data["parameters"], errors="coerce"
        )
        valid_data["downloads"] = pd.to_numeric(
            valid_data["downloads"], errors="coerce"
        )

        # Drop any rows with non-numeric values after conversion
        valid_data = valid_data.dropna(subset=["parameters", "downloads"])

        if valid_data.empty:
            return go.Figure().add_annotation(text="No data available", showarrow=False)

        # Add formatted columns directly to the dataframe
        valid_data["parameters_formatted"] = valid_data["parameters"].apply(
            format_big_number
        )
        valid_data["downloads_formatted"] = valid_data["downloads"].apply(
            format_big_number
        )

        # Create interactive scatter plot
        fig = px.scatter(
            valid_data,
            x="parameters",
            y="downloads",
            color="tasks",
            log_x=True,
            log_y=True,
            title="Downloads vs. Parameters for All Models",
            labels={
                "parameters": "Parameters (log)",
                "downloads": "Downloads (log)",
                "tasks": "Task",
            },
            height=700,
            hover_data=[
                "model_id",
                "tasks",
                "parameters_formatted",
                "downloads_formatted",
            ],
        )

        # Update hover template to show formatted values
        hover_template = "<b>%{customdata[0]}</b><br>"
        hover_template += "Parameters: %{customdata[2]}<br>"
        hover_template += "Downloads: %{customdata[3]}<br>"
        hover_template += "Task: %{customdata[1]}"
        hover_template += "<extra></extra>"
        fig.update_traces(hovertemplate=hover_template)

        return fig

    st.plotly_chart(plot_downloads_vs_parameters(filtered_df), use_container_width=True)
    st.dataframe(filtered_df)

# Footer
st.markdown("---")
st.markdown("**Data Source:** Hugging Face Hub API")
st.markdown("**Note:** Parameter information may be missing for some models.")
