import streamlit as st
import os
import time
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import sys
from PIL import Image
from pathlib import Path

st.set_page_config(
    page_title="Federated Learning with Differential Privacy",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .section-header {
        font-size: 1.8rem;
        color: #1976D2;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .subsection-header {
        font-size: 1.4rem;
        color: #1565C0;
        margin-top: 0.8rem;
        margin-bottom: 0.8rem;
    }
    .success-message {
        color: #4CAF50;
        font-weight: bold;
    }
    .results-box {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.markdown("<div class='main-header'>Feature-Specific Differential Privacy in Federated Learning</div>",
            unsafe_allow_html=True)
st.markdown("---")

# Sidebar with description
with st.sidebar:
    st.title("About")
    st.markdown("""
    This application provides an interface to run federated learning experiments with feature-specific differential privacy.

    The system uses attention mechanisms (CBAM) to apply varying levels of privacy protection to different features based on their importance.

    You can adjust various parameters related to:
    - Federated learning configuration
    - Differential privacy settings
    - Model architecture
    - Data distribution

    After running an experiment, you'll be able to view the results and visualizations.
    """)

    st.markdown("---")
    st.markdown("### Key Features")
    st.markdown("""
    - Feature-specific differential privacy
    - Privacy-utility tradeoff analysis
    - Multiple federated learning strategies
    - Support for IID and non-IID data distributions
    - Comprehensive privacy metrics and visualizations
    """)

# Main content divided into tabs
tab1, tab2, tab3 = st.tabs(["Configure Experiment", "Run Experiment", "Results"])

# Tab 1: Configure Experiment
with tab1:
    st.markdown("<div class='section-header'>Experiment Configuration</div>", unsafe_allow_html=True)

    # Create columns for better layout
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='subsection-header'>Federated Learning Parameters</div>", unsafe_allow_html=True)

        st.session_state.num_clients = st.slider(
            "Number of Clients",
            min_value=2,
            max_value=10,
            value=3,
            help="Number of clients to simulate in the federated learning setup"
        )

        st.session_state.num_rounds = st.slider(
            "Number of Rounds",
            min_value=1,
            max_value=20,
            value=5,
            help="Number of federated learning rounds to perform"
        )

        st.session_state.epochs = st.slider(
            "Epochs per Round",
            min_value=1,
            max_value=10,
            value=1,
            help="Number of training epochs per federated round"
        )

        st.session_state.strategy = st.radio(
            "Federated Learning Strategy",
            options=["fedavg", "fedprox"],
            index=0,
            help="FedAvg is the standard federated averaging algorithm. FedProx adds a proximal term to improve convergence for heterogeneous clients."
        )

        if st.session_state.strategy == "fedprox":
            st.session_state.proximal_mu = st.slider(
                "Proximal Term Weight (μ)",
                min_value=0.001,
                max_value=1.0,
                value=0.01,
                format="%.3f",
                help="Weight of the proximal term in FedProx (higher values enforce closer similarity to global model)"
            )
        else:
            st.session_state.proximal_mu = 0.01

        st.markdown("<div class='subsection-header'>Data Distribution Parameters</div>", unsafe_allow_html=True)

        st.session_state.distribution = st.radio(
            "Data Distribution Type",
            options=["iid", "non_iid"],
            index=0,
            help="IID (Independent and Identically Distributed) or non-IID data distribution across clients"
        )

        if st.session_state.distribution == "non_iid":
            st.session_state.alpha = st.slider(
                "Dirichlet Alpha Parameter",
                min_value=0.1,
                max_value=10.0,
                value=0.5,
                help="Controls the degree of non-IID-ness (lower values = more heterogeneous data distribution)"
            )
        else:
            st.session_state.alpha = 0.5

    with col2:
        st.markdown("<div class='subsection-header'>Differential Privacy Parameters</div>", unsafe_allow_html=True)

        st.session_state.noise_multiplier = st.slider(
            "Noise Multiplier (σ)",
            min_value=0.1,
            max_value=2.0,
            value=0.8,
            help="Controls the amount of noise added for privacy (higher = more privacy, less utility)"
        )

        st.session_state.max_grad_norm = st.slider(
            "Max Gradient Norm (C)",
            min_value=0.1,
            max_value=5.0,
            value=1.0,
            help="Maximum allowed gradient norm for clipping (lower = more privacy, potentially less utility)"
        )

        st.session_state.delta = st.select_slider(
            "Delta Parameter (δ)",
            options=[1e-6, 1e-5, 1e-4, 1e-3],
            value=1e-5,
            format_func=lambda x: f"{x:.0e}",
            help="Probability of privacy guarantee breaking (smaller is better)"
        )

        st.session_state.feature_specific = st.toggle(
            "Enable Feature-Specific Privacy",
            value=True,
            help="Use attention mechanisms to apply different amounts of noise to different features based on importance"
        )

        st.markdown("<div class='subsection-header'>Model Parameters</div>", unsafe_allow_html=True)

        st.session_state.model_type = st.radio(
            "Model Architecture Type",
            options=["resnet", "densenet"],
            index=0,
            help="Base model architecture to use"
        )

        if st.session_state.model_type == "resnet":
            st.session_state.model_name = st.selectbox(
                "Model Variant",
                options=["resnet18", "resnet34", "resnet50"],
                index=0,
                help="Specific model variant within the selected architecture"
            )
        else:
            st.session_state.model_name = st.selectbox(
                "Model Variant",
                options=["densenet121", "densenet169", "densenet201"],
                index=0,
                help="Specific model variant within the selected architecture"
            )

    # System paths and data paths
    st.markdown("<div class='subsection-header'>System & Data Paths</div>", unsafe_allow_html=True)

    # Add code directory setting
    col1, col2 = st.columns(2)
    with col1:
        # Default to current directory, but allow user to specify the correct location
        default_code_dir = "C:\\Users\\HP\\Documents\\GitHub\\IRP_Project\\FS-DP_Model"
        st.session_state.code_directory = st.text_input(
            "Code Directory",
            value=default_code_dir,
            help="Full path to the directory containing main.py, client.py, etc."
        )

    with col2:
        # Check if the path is valid and show status
        code_dir_valid = os.path.exists(os.path.join(st.session_state.code_directory, "main.py"))

        if code_dir_valid:
            st.success("✅ main.py found in specified directory")
        else:
            st.error("❌ main.py not found in specified directory")

    # Data paths
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.img_dir = st.text_input(
            "Image Directory",
            value="E:/IRP_dataset_new/IRP_Final_Images",
            help="Directory containing the retinopathy images"
        )

    with col2:
        st.session_state.labels_path = st.text_input(
            "Labels Path",
            value="E:/IRP_dataset_new/IRP_Final_Labels.csv",
            help="Path to CSV file with image labels"
        )

# Tab 2: Run Experiment
with tab2:
    st.markdown("<div class='section-header'>Run Experiment</div>", unsafe_allow_html=True)

    # Display current configuration summary
    with st.expander("Current Configuration", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Federated Learning:**")
            st.write(f"- Clients: {st.session_state.get('num_clients', 3)}")
            st.write(f"- Rounds: {st.session_state.get('num_rounds', 5)}")
            st.write(f"- Epochs: {st.session_state.get('epochs', 1)}")
            st.write(f"- Strategy: {st.session_state.get('strategy', 'fedavg')}")
            if st.session_state.get('strategy', 'fedavg') == "fedprox":
                st.write(f"- Proximal μ: {st.session_state.get('proximal_mu', 0.01)}")

        with col2:
            st.markdown("**Differential Privacy:**")
            st.write(f"- Noise Multiplier: {st.session_state.get('noise_multiplier', 0.8)}")
            st.write(f"- Max Gradient Norm: {st.session_state.get('max_grad_norm', 1.0)}")
            st.write(f"- Delta: {st.session_state.get('delta', 1e-5)}")
            st.write(f"- Feature-Specific: {st.session_state.get('feature_specific', True)}")

        with col3:
            st.markdown("**Model & Data:**")
            st.write(
                f"- Model: {st.session_state.get('model_type', 'resnet')} - {st.session_state.get('model_name', 'resnet18')}")
            st.write(f"- Distribution: {st.session_state.get('distribution', 'iid')}")
            if st.session_state.get('distribution', 'iid') == "non_iid":
                st.write(f"- Alpha: {st.session_state.get('alpha', 0.5)}")

    # Run experiment button
    run_experiment = st.button("Start Experiment", type="primary", use_container_width=True)

    if run_experiment:
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        log_output = st.empty()

        # Check if code directory exists and contains main.py
        code_dir = Path(st.session_state.code_directory)
        main_py_path = code_dir / "main.py"

        if not main_py_path.exists():
            status_text.error(f"Error: main.py not found at {main_py_path}")
            st.stop()

        # Prepare command with parameters
        cmd = [
            sys.executable,  # Use the current Python interpreter
            str(main_py_path),  # Use the full path to main.py
            "--num_clients", str(st.session_state.num_clients),
            "--num_rounds", str(st.session_state.num_rounds),
            "--epochs", str(st.session_state.epochs),
            "--noise_multiplier", str(st.session_state.noise_multiplier),
            "--max_grad_norm", str(st.session_state.max_grad_norm),
            "--delta", str(st.session_state.delta),
            "--distribution", st.session_state.distribution,
            "--alpha", str(st.session_state.alpha),
            "--strategy", st.session_state.strategy,
            "--proximal_mu", str(st.session_state.proximal_mu),
            "--feature_specific", str(st.session_state.feature_specific),
            "--model_type", st.session_state.model_type,
            "--model_name", st.session_state.model_name,
            "--img_dir", st.session_state.img_dir,
            "--labels_path", st.session_state.labels_path
        ]

        # Log the command
        status_text.text(f"Starting experiment from directory: {code_dir}")

        status_text.text("Starting experiment...")

        try:
            # Run experiment process from the correct directory
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                cwd=str(code_dir)  # Set the correct working directory
            )

            # Initialize logs
            logs = []

            # Get estimated total rounds (for progress tracking)
            total_steps = st.session_state.num_rounds * (st.session_state.num_clients + 2)  # Rough estimate
            current_step = 0

            # Process logs in real-time
            for line in process.stdout:
                logs.append(line.strip())

                # Display last 10 log lines
                log_output.text("\n".join(logs[-15:]))

                # Update progress based on log content
                if "Starting client" in line:
                    current_step += 1
                elif "Starting round" in line or "Round completed" in line:
                    current_step += 1

                # Update progress bar
                progress = min(current_step / total_steps, 1.0)
                progress_bar.progress(progress)

                # Update status text based on log content
                if "Server Round" in line and "Aggregating" in line:
                    round_num = line.split("Server Round")[1].split("-")[0].strip()
                    status_text.text(f"Processing round {round_num}...")
                elif "Creating visualization" in line:
                    status_text.text("Generating visualizations...")
                elif "Error" in line or "error" in line:
                    status_text.text(f"Error detected: {line}")

            # Finalize with return code
            return_code = process.wait()

            if return_code == 0:
                progress_bar.progress(1.0)
                status_text.markdown("<div class='success-message'>Experiment completed successfully!</div>",
                                     unsafe_allow_html=True)
                st.session_state.experiment_completed = True
                st.balloons()

                # Guide to view results
                st.info("You can now go to the Results tab to view the experiment outcomes.")
            else:
                status_text.error(f"Experiment failed with return code {return_code}")

        except Exception as e:
            status_text.error(f"Error running experiment: {str(e)}")

# Tab 3: Results
with tab3:
    st.markdown("<div class='section-header'>Experiment Results</div>", unsafe_allow_html=True)

    # Check if we need to refresh the results
    refresh_results = st.button("Refresh Results")


    # Function to load and display results
    def display_results():
        results_found = False

        # Use the correct code directory for finding results
        code_dir = Path(st.session_state.get('code_directory', '.'))
        results_path = code_dir / "aggregated_metrics" / "rounds_metadata.csv"

        # Check if metadata file exists
        if results_path.exists():
            results_found = True

            # Load rounds metadata
            rounds_df = pd.read_csv(results_path)

            # Display summary statistics
            st.markdown("<div class='subsection-header'>Performance Summary</div>", unsafe_allow_html=True)

            # Create metrics
            col1, col2, col3, col4 = st.columns(4)

            # Get latest round metrics
            last_round = rounds_df.iloc[-1]

            with col1:
                st.metric(
                    label="Final Accuracy",
                    value=f"{last_round['average_accuracy']:.4f}"
                )

            with col2:
                st.metric(
                    label="Final F1 Score",
                    value=f"{last_round['average_f1']:.4f}"
                )

            with col3:
                # Get the epsilon column (cumulative if available)
                if 'average_cumulative_epsilon' in last_round:
                    epsilon = last_round['average_cumulative_epsilon']
                    epsilon_label = "Privacy Budget (ε)"
                else:
                    epsilon = last_round['average_epsilon']
                    epsilon_label = "Privacy Budget (ε)"

                st.metric(
                    label=epsilon_label,
                    value=f"{epsilon:.4f}"
                )

            with col4:
                st.metric(
                    label="Feature-Specific Privacy",
                    value="Enabled" if last_round.get('feature_specific_effective', False) else "Disabled"
                )

            # Display visualization grid
            st.markdown("<div class='subsection-header'>Visualizations</div>", unsafe_allow_html=True)

            # Create collapsible sections for different visualization categories
            with st.expander("Performance Visualizations", expanded=True):
                # Try to load key performance visualizations
                perf_visuals = []

                # Check for common performance visualizations
                perf_paths = [
                    str(code_dir / "visualizations/final_performance_summary.png"),
                    str(code_dir / "visualizations/average_improvement_rate.png"),
                    str(code_dir / "visualizations/global_vs_client_metrics_round_*.png"),
                    str(code_dir / "visualizations/all_clients_accuracy_per_round.png")
                ]

                for path_pattern in perf_paths:
                    matching_files = glob.glob(path_pattern)
                    perf_visuals.extend(matching_files)

                # Display performance visualizations in grid
                if perf_visuals:
                    cols = st.columns(2)
                    for i, img_path in enumerate(perf_visuals):
                        try:
                            img = Image.open(img_path)
                            cols[i % 2].image(img, caption=os.path.basename(img_path), use_column_width=True)
                        except Exception as e:
                            cols[i % 2].error(f"Could not load image {img_path}: {str(e)}")
                else:
                    st.info("No performance visualizations found")

            with st.expander("Privacy Analysis Visualizations", expanded=True):
                # Try to load key privacy visualizations
                privacy_visuals = []

                # Check for common privacy visualizations
                privacy_paths = [
                    str(code_dir / "visualizations/final_privacy_utility_tradeoff.png"),
                    str(code_dir / "visualizations/final_privacy_budget_progression.png"),
                    str(code_dir / "visualizations/privacy_analysis/membership_inference_risk.png"),
                    str(code_dir / "visualizations/privacy_analysis/privacy_utility_tradeoff.png"),
                    str(code_dir / "visualizations/privacy_analysis/privacy_leakage_reduction.png"),
                    str(code_dir / "visualizations/privacy_analysis/epsilon_noise_tradeoff.png"),
                    str(code_dir / "visualizations/privacy_analysis/privacy_reconstruction_test.png")
                ]

                for path_pattern in privacy_paths:
                    matching_files = glob.glob(path_pattern)
                    privacy_visuals.extend(matching_files)

                # Display privacy visualizations in grid
                if privacy_visuals:
                    cols = st.columns(2)
                    for i, img_path in enumerate(privacy_visuals[:8]):  # Limit to 8 images
                        try:
                            img = Image.open(img_path)
                            cols[i % 2].image(img, caption=os.path.basename(img_path), use_column_width=True)
                        except Exception as e:
                            cols[i % 2].error(f"Could not load image {img_path}: {str(e)}")
                else:
                    st.info("No privacy visualizations found")

            with st.expander("Client-Specific Visualizations"):
                # Check if there are client-specific visualizations
                client_visuals = glob.glob(str(code_dir / "visualizations/client_*"))

                if client_visuals:
                    # Create selectbox for client selection
                    clients = sorted(list(set([os.path.basename(path).split("_")[1].split("client_")[0]
                                               for path in client_visuals if "client_" in path])))

                    selected_client = st.selectbox(
                        "Select Client",
                        options=clients if clients else ["No clients found"],
                        index=0
                    )

                    if clients:
                        # Get visualizations for selected client
                        client_paths = glob.glob(str(code_dir / f"visualizations/*client_{selected_client}*"))

                        if client_paths:
                            cols = st.columns(2)
                            for i, img_path in enumerate(client_paths[:8]):  # Limit to 8 images
                                try:
                                    img = Image.open(img_path)
                                    cols[i % 2].image(img, caption=os.path.basename(img_path), use_column_width=True)
                                except:
                                    pass  # Skip images that can't be loaded
                        else:
                            st.info(f"No visualizations found for client {selected_client}")
                else:
                    st.info("No client-specific visualizations found")

            # Display final summary report if available
            report_path = code_dir / "aggregated_metrics/final_summary_report.txt"
            if report_path.exists():
                with st.expander("Experiment Summary Report", expanded=True):
                    with open(report_path, "r") as f:
                        report_content = f.read()

                    st.text(report_content)

            # Display results file paths
            with st.expander("Results File Locations"):
                code_dir_str = str(code_dir)
                st.markdown(f"""
                You can find the experiment results in the following locations:

                - **Visualizations**: `{code_dir_str}/visualizations/`
                - **Metrics Data**: `{code_dir_str}/aggregated_metrics/`
                - **Saved Models**: `{code_dir_str}/saved_models/`
                - **Logs**: `{code_dir_str}/logs/`
                """)

        if not results_found:
            st.warning("No experiment results found. Please run an experiment first.")


    # Display results (initial or refreshed)
    if st.session_state.get('experiment_completed', False) or refresh_results:
        display_results()
    else:
        st.info("Run an experiment to view results here.")

# Add footer
st.markdown("---")
st.markdown("Feature-Specific Differential Privacy in Federated Learning | UI created with Streamlit")