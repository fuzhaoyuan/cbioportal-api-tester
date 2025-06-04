# app.py
import streamlit as st
import json
import requests
import pandas as pd
import time
import plotly.express as px
from urllib.parse import urlparse, urlunparse, urljoin

# --- Configuration ---
REQUEST_TIMEOUT = 30
SUCCESS_STATUS_CODES = range(200, 300)

st.set_page_config(layout="wide")
st.title("API Batch Tester & Visualizer")

# --- Utility Functions ---
def create_error_result(filename, error_type, message, url='N/A', target='N/A'):
    """Create standardized error result dictionary"""
    return {
        'sourceFile': filename, 'url': url, 'requestTarget': target,
        'status': f'Skipped - {error_type}', 'responseTimeMs': 0, 'error': message
    }

def extract_endpoint_path(url_string):
    """Extract the path from a URL string"""
    if pd.isna(url_string) or url_string == 'N/A':
        return 'N/A'
    try:
        parsed_url = urlparse(url_string)
        return parsed_url.path if parsed_url.path else '/'
    except Exception:
        return 'Invalid URL Path'

def normalize_url(url):
    """Normalize URL by adding scheme if missing"""
    if not url or not url.strip():
        return None
    url = url.strip()
    if url.startswith('//'):
        return 'https:' + url
    parsed_url = urlparse(url)
    if not parsed_url.scheme and parsed_url.netloc:
        return 'https:' + url
    return url

def build_request_targets(original_url, base_url_a, base_url_b):
    """Build list of target URLs based on base URL overrides"""
    if not original_url:
        return []
    
    parsed_original = urlparse(original_url)
    path_and_onwards = urlunparse(('', '', parsed_original.path, parsed_original.params, 
                                  parsed_original.query, parsed_original.fragment))
    
    targets = []
    if base_url_a and base_url_b:
        targets.extend([
            {'label': 'User URL A', 'url_val': urljoin(base_url_a, path_and_onwards)},
            {'label': 'User URL B', 'url_val': urljoin(base_url_b, path_and_onwards)}
        ])
    elif base_url_a:
        targets.append({'label': 'User URL A', 'url_val': urljoin(base_url_a, path_and_onwards)})
    elif base_url_b:
        targets.append({'label': 'User URL B', 'url_val': urljoin(base_url_b, path_and_onwards)})
    else:
        targets.append({'label': 'File URL', 'url_val': original_url})
    
    return targets

def send_post_request(target_url, payload_str, source_filename, request_target_label):
    """Send a POST request and return result dictionary"""
    final_target_url = normalize_url(target_url)
    if not final_target_url:
        return create_error_result(source_filename, 'Invalid Target URL', 
                                 "Target URL was empty or invalid", target=request_target_label)

    headers = {}
    body_to_send = None
    if payload_str and payload_str.strip() and payload_str.strip() != '{}':
        headers['Content-Type'] = 'application/json'
        body_to_send = payload_str.encode('utf-8')

    start_time = time.perf_counter()
    try:
        response = requests.post(final_target_url, headers=headers, data=body_to_send, timeout=REQUEST_TIMEOUT)
        end_time = time.perf_counter()
        response_time_ms = (end_time - start_time) * 1000
        
        return {
            'sourceFile': source_filename, 'url': final_target_url, 'requestTarget': request_target_label,
            'status': response.status_code, 'statusText': response.reason,
            'responseTimeMs': response_time_ms, 'error': None
        }
    except requests.exceptions.Timeout:
        end_time = time.perf_counter()
        response_time_ms = (end_time - start_time) * 1000
        return {
            'sourceFile': source_filename, 'url': final_target_url, 'requestTarget': request_target_label,
            'status': 'Error - Timeout', 'responseTimeMs': response_time_ms, 'error': 'Request timed out'
        }
    except requests.exceptions.RequestException as e:
        end_time = time.perf_counter()
        response_time_ms = (end_time - start_time) * 1000
        return {
            'sourceFile': source_filename, 'url': final_target_url, 'requestTarget': request_target_label,
            'status': 'Error - Request Failed', 'responseTimeMs': response_time_ms, 'error': str(e)
        }

def parse_json_file(uploaded_file):
    """Parse uploaded JSON file and extract request data"""
    try:
        file_content = uploaded_file.getvalue().decode("utf-8")
        json_data = json.loads(file_content)
        
        if not isinstance(json_data, dict) or 'body' not in json_data:
            return None, "'body' key not found"
        
        request_body = json_data['body']
        if not isinstance(request_body, dict):
            return None, "'body' not an object"
        
        return {'payload': request_body.get('data'), 'url': request_body.get('url')}, None
        
    except json.JSONDecodeError:
        return None, "Invalid JSON in file"
    except Exception as e:
        return None, str(e)

# --- UI Rendering Functions ---
def display_sidebar():
    """Handle all sidebar UI elements and return user inputs"""
    st.sidebar.header("Upload Files")
    uploaded_files = st.sidebar.file_uploader(
        "Select your JSON log files:",
        type=['json'],
        accept_multiple_files=True,
        help="Upload JSON files containing the request body structure."
    )

    st.sidebar.header("Base URL Overrides (Optional)")
    st.sidebar.caption("These will replace the domain part of URLs from files. Path & query from files will be retained.")
    user_base_url_a = st.sidebar.text_input(
        "Base URL A (e.g., http://localhost:8080)",
        key="user_base_url_a",
        help="If set, replaces the domain from file URLs. Include scheme (http/https)."
    ).strip()
    user_base_url_b = st.sidebar.text_input(
        "Base URL B (e.g., https://test-server.com)",
        key="user_base_url_b",
        help="If set with Base URL A, also used for comparison. Include scheme."
    ).strip()

    run_button = st.sidebar.button("Run Tests & Visualize", type="primary")

    # Provide feedback on URL usage
    if user_base_url_a and user_base_url_b:
        st.sidebar.info("Payloads will be sent to constructed URLs using Base URL A and Base URL B for comparison.")
    elif user_base_url_a:
        st.sidebar.info("Payloads will be sent to constructed URLs using Base URL A.")
    elif user_base_url_b:
        st.sidebar.info("Payloads will be sent to constructed URLs using Base URL B.")
    else:
        st.sidebar.info("No Base URL overrides. Full URLs from JSON files will be used.")
    
    st.sidebar.markdown("---")
    return uploaded_files, user_base_url_a, user_base_url_b, run_button

def display_results_and_charts(df_results, num_uploaded_files, user_base_url_a_active, user_base_url_b_active):
    """Display results DataFrame, summary statistics, and charts.
       Table will not show 'sourceFile' or full 'url'.
       If comparing URL A and B, their speeds will be on the same row.
    """
    st.header("Test Results")

    if df_results.empty:
        st.info("No results to display.")
        return

    # --- Prepare DataFrame for Display ---
    # Ensure 'endpointPath' exists (it should be added in the main flow after run_batch_processing)
    if 'endpointPath' not in df_results.columns:
        st.error("Internal error: 'endpointPath' column is missing in results.")
        return

    display_df = None

    if user_base_url_a_active and user_base_url_b_active:
        # Prepare data for pivoting - only results from User URL A and User URL B
        df_to_pivot = df_results[df_results['requestTarget'].isin(['User URL A', 'User URL B'])].copy()
        
        # We need at least one row for each target type (A and B) for a given sourceFile/endpointPath to make a meaningful pivot row.
        # Check if there are pairs to pivot.
        if not df_to_pivot.empty and len(df_to_pivot['requestTarget'].unique()) == 2:
            try:
                # Pivot to bring User URL A and User URL B metrics onto the same row.
                # sourceFile is used in index to ensure uniqueness of rows before aggregation,
                # then it's dropped for display.
                df_pivot = df_to_pivot.pivot_table(
                    index=['sourceFile', 'endpointPath'], 
                    columns='requestTarget',
                    values=['responseTimeMs', 'status', 'statusText', 'error'],
                    aggfunc='first' # Should be one entry per (file, endpoint, target)
                )
                
                # Flatten MultiIndex columns: (value, target_label) -> value_target_label_short
                df_pivot.columns = [
                    f"{val}_{tgt.replace('User URL ', '')}" for val, tgt in df_pivot.columns
                ]
                df_pivot.reset_index(inplace=True) # Makes 'sourceFile', 'endpointPath' regular columns

                # Select and order columns for display, excluding 'sourceFile'
                # Display 'endpointPath', then A's metrics, then B's metrics
                pivot_display_columns = ['endpointPath']
                metrics_to_include = ['responseTimeMs', 'status', 'statusText', 'error']
                targets_short = ['A', 'B']

                for metric_base in metrics_to_include:
                    for target_suffix in targets_short:
                        col_name = f"{metric_base}_{target_suffix}"
                        if col_name in df_pivot.columns:
                            pivot_display_columns.append(col_name)
                
                # Ensure all selected columns actually exist and are unique
                final_pivot_cols_for_display = []
                seen_cols = set()
                for col in pivot_display_columns:
                    if col in df_pivot.columns and col not in seen_cols:
                        final_pivot_cols_for_display.append(col)
                        seen_cols.add(col)
                
                if 'endpointPath' in final_pivot_cols_for_display: # Ensure endpointPath is present
                    display_df = df_pivot[final_pivot_cols_for_display]
                else: # Should not happen if logic is correct
                    display_df = df_pivot 
                    st.warning("Pivoted table created, but 'endpointPath' column was unexpectedly missing from display list. Showing all pivoted columns.")


            except Exception as e:
                st.error(f"Error pivoting data for comparison view: {e}. Displaying results in standard format.")
                # Fallback to standard display if pivoting fails
                standard_display_cols = ['endpointPath', 'requestTarget', 'status', 'statusText', 'responseTimeMs', 'error']
                display_df = df_results[[col for col in standard_display_cols if col in df_results.columns]]
        else:
            # Not enough distinct targets (A and B) for pivot, use standard display
            st.info("Not enough data from both User URL A and User URL B to create a comparative table row for each source file/endpoint. Displaying results individually.")
            standard_display_cols = ['endpointPath', 'requestTarget', 'status', 'statusText', 'responseTimeMs', 'error']
            display_df = df_results[[col for col in standard_display_cols if col in df_results.columns]]
    
    if display_df is None: # If not comparison mode, or if pivot conditions not met
        standard_display_cols = ['endpointPath', 'requestTarget', 'status', 'statusText', 'responseTimeMs', 'error']
        display_df = df_results[[col for col in standard_display_cols if col in df_results.columns]]

    st.dataframe(display_df, height=300, use_container_width=True)

    # --- Summary Statistics (remains largely the same, operates on original df_results) ---
    successful_requests_df = df_results[
        df_results['status'].apply(lambda x: isinstance(x, int) and x in SUCCESS_STATUS_CODES) & 
        df_results['responseTimeMs'].notna()
    ].copy()

    st.header("Summary Statistics")
    # ... (rest of your summary statistics code from your provided script - no change needed here) ...
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Files Processed", num_uploaded_files)
    attempted_requests = len(df_results[~df_results['status'].astype(str).str.startswith('Skipped')])
    col2.metric("Requests Attempted", attempted_requests)
    col3.metric("Successful Requests (2xx)", len(successful_requests_df))

    if not successful_requests_df.empty:
        avg_time = successful_requests_df['responseTimeMs'].mean()
        min_time = successful_requests_df['responseTimeMs'].min()
        max_time = successful_requests_df['responseTimeMs'].max()

        s_col1, s_col2, s_col3 = st.columns(3)
        s_col1.metric("Avg. Response Time (Overall 2xx)", f"{avg_time:.2f} ms")
        s_col2.metric("Min. Response Time (Overall 2xx)", f"{min_time:.2f} ms")
        s_col3.metric("Max. Response Time (Overall 2xx)", f"{max_time:.2f} ms")

        # --- Charting (remains largely the same, operates on successful_requests_df which has sourceFile) ---
        if user_base_url_a_active and user_base_url_b_active: 
            st.header("Comparative Response Time Chart (User Base URLs)")
            df_for_comp_chart = successful_requests_df[
                successful_requests_df['requestTarget'].isin(['User URL A', 'User URL B'])
            ].copy()

            # Ensure endpointPath column is available for charting
            if 'endpointPath' not in df_for_comp_chart.columns and 'url' in df_for_comp_chart.columns:
                 df_for_comp_chart['endpointPath'] = df_for_comp_chart['url'].apply(extract_endpoint_path)

            if not df_for_comp_chart.empty and len(df_for_comp_chart['requestTarget'].unique()) > 1:
                fig = px.bar(df_for_comp_chart, 
                             x='endpointPath', y='responseTimeMs', color='requestTarget', 
                             barmode='group',
                             labels={'responseTimeMs': 'Response Time (ms)', 'endpointPath': 'Endpoint Path', 'requestTarget': 'Target'},
                             title="Response Times: Base URL A vs Base URL B per Endpoint Path",
                             hover_data=['sourceFile', 'url']) # sourceFile is still useful in hover
                fig.update_layout(xaxis_title="Endpoint Path (from file URL)", yaxis_title="Response Time (ms)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough successful requests for both User URL A and User URL B for a comparative chart.")
                # Fallback chart if comparative not possible but there's data
                if not successful_requests_df.empty:
                    st.header("Response Time Chart (All Successful Requests)") # Changed header
                    chart_data = successful_requests_df.groupby(['endpointPath', 'requestTarget'])['responseTimeMs'].mean().reset_index()
                    if not chart_data.empty:
                        fig_fallback = px.bar(chart_data, x='endpointPath', y='responseTimeMs', color='requestTarget',
                                           labels={'responseTimeMs': 'Avg Response Time (ms)', 'endpointPath': 'Endpoint Path', 'requestTarget': 'Target'},
                                           title="Average Response Times per Endpoint Path and Target")
                        st.plotly_chart(fig_fallback, use_container_width=True)
                        
        elif not successful_requests_df.empty: # Only one or no user base URLs active
            st.header("Response Time Chart (All Successful Requests)")
             # Ensure endpointPath column is available for charting
            if 'endpointPath' not in successful_requests_df.columns and 'url' in successful_requests_df.columns:
                 successful_requests_df['endpointPath'] = successful_requests_df['url'].apply(extract_endpoint_path)

            chart_data = successful_requests_df.groupby(['endpointPath', 'requestTarget'])['responseTimeMs'].mean().reset_index()
            if not chart_data.empty:
                fig = px.bar(chart_data, x='endpointPath', y='responseTimeMs', color='requestTarget',
                            labels={'responseTimeMs': 'Avg Response Time (ms)', 'endpointPath': 'Endpoint Path', 'requestTarget': 'Target'},
                            title="Average Response Times per Endpoint Path")
                fig.update_layout(xaxis_title="Endpoint Path (from file URL)", yaxis_title="Avg Response Time (ms)")
                st.plotly_chart(fig, use_container_width=True)
    else: # No successful requests at all
        st.info("No successful requests (2xx) to display detailed statistics or chart.")

# --- Core Logic Function ---
def run_batch_processing(uploaded_files_list, base_url_a_str, base_url_b_str, progress_bar_obj):
    """Process uploaded files, send requests, and return list of results"""
    all_results = []
    num_total_files = len(uploaded_files_list)

    for i, uploaded_file in enumerate(uploaded_files_list, 1):
        filename = uploaded_file.name
        
        # Parse file
        file_data, error = parse_json_file(uploaded_file)
        if error:
            all_results.append(create_error_result(filename, 'File Parse Error', error))
            if progress_bar_obj: progress_bar_obj.progress(i / num_total_files)
            continue
        
        # Build target URLs
        targets = build_request_targets(file_data['url'], base_url_a_str, base_url_b_str)
        
        if not targets:
            if not file_data['url'] and (base_url_a_str or base_url_b_str):
                all_results.append(create_error_result(filename, 'File URL Missing for Base Override', 
                                                     "'body.url' missing in file"))
            else:
                all_results.append(create_error_result(filename, 'No URL Info', 
                                                     "No URL in file or Base URL override"))
            if progress_bar_obj: progress_bar_obj.progress(i / num_total_files)
            continue
        
        # Execute requests for all targets
        for target in targets:
            result = send_post_request(target['url_val'], file_data['payload'], filename, target['label'])
            all_results.append(result)
        
        if progress_bar_obj:
            progress_bar_obj.progress(i / num_total_files)
    
    return all_results

# --- Main Application Flow ---
if __name__ == "__main__":
    # Get user inputs from the sidebar
    uploaded_files, user_base_url_a, user_base_url_b, run_button = display_sidebar()

    if run_button and uploaded_files:
        # Initialize progress bar in the sidebar
        progress_bar = st.sidebar.progress(0)
        st.sidebar.info(f"Processing {len(uploaded_files)} files...")

        all_results_list = run_batch_processing(uploaded_files, user_base_url_a, user_base_url_b, progress_bar)

        st.sidebar.success("Processing complete!")

        if all_results_list:
            df_results_final = pd.DataFrame(all_results_list)
            df_results_final['endpointPath'] = df_results_final['url'].apply(extract_endpoint_path)

            display_results_and_charts(df_results_final, len(uploaded_files), bool(user_base_url_a), bool(user_base_url_b))
        else:
            st.warning("No results were generated from the processed files.")

    elif run_button:
        st.sidebar.warning("Please select JSON files first.")