import streamlit as st
import json
import requests
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set page config
st.set_page_config(
    page_title="API Performance Comparison",
    page_icon="⚡",
    layout="wide"
)

def extract_api_path(original_url):
    """Extract API path after /api/ from the original URL"""
    # Remove the leading // from original URL
    clean_url = original_url.lstrip('/')
    if clean_url.startswith('www.cbioportal.org'):
        clean_url = clean_url.replace('www.cbioportal.org', '', 1)
    
    # Find /api/ and return everything after it
    api_index = clean_url.find('/api/')
    if api_index != -1:
        return clean_url[api_index + 5:]  # +5 to skip '/api/'
    
    # If no /api/ found, return the path after domain
    return clean_url.lstrip('/')
def extract_request_data(json_file_content):
    """Extract request data from uploaded JSON file"""
    try:
        data = json.loads(json_file_content)
        body = data.get('body', {})
        
        # Extract data payload
        request_data = body.get('data', '{}')
        if isinstance(request_data, str):
            request_data = json.loads(request_data)
        
        # Extract method and URL
        method = body.get('method', 'POST')
        original_url = body.get('url', '')
        
        return {
            'data': request_data,
            'method': method,
            'original_url': original_url,
            'api_path': extract_api_path(original_url),
            'filename': getattr(json_file_content, 'name', 'unknown')
        }
    except Exception as e:
        st.error(f"Error parsing JSON: {str(e)}")
        return None

def make_request(url, data, method='POST', timeout=30):
    """Make HTTP request and measure response time"""
    start_time = time.time()
    
    try:
        headers = {
            'Content-Type': 'application/json',
            'Accept': '*/*'
        }
        
        if method.upper() == 'POST':
            response = requests.post(url, json=data, headers=headers, timeout=timeout)
        elif method.upper() == 'GET':
            response = requests.get(url, params=data, headers=headers, timeout=timeout)
        else:
            response = requests.request(method, url, json=data, headers=headers, timeout=timeout)
        
        end_time = time.time()
        response_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
        return {
            'success': True,
            'response_time': response_time,
            'status_code': response.status_code,
            'response_size': len(response.content),
            'timestamp': datetime.now()
        }
    except requests.exceptions.Timeout:
        return {
            'success': False,
            'error': 'Request timeout',
            'response_time': timeout * 1000,
            'timestamp': datetime.now()
        }
    except Exception as e:
        end_time = time.time()
        return {
            'success': False,
            'error': str(e),
            'response_time': (end_time - start_time) * 1000,
            'timestamp': datetime.now()
        }

def build_full_url(base_url, original_url):
    """Build full URL by replacing the domain part"""
    # Remove the leading // from original URL
    path_part = original_url.lstrip('/')
    if path_part.startswith('www.cbioportal.org'):
        path_part = path_part.replace('www.cbioportal.org', '', 1)
    
    # Ensure base URL ends with /
    if not base_url.endswith('/'):
        base_url += '/'
    
    # Ensure path starts without /
    path_part = path_part.lstrip('/')
    
    return base_url + path_part

def main():
    st.title("⚡ API Performance Comparison Tool")
    st.markdown("Compare response times between two API endpoints using your JSON request files.")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # File upload - moved to first position
        st.subheader("📁 Upload JSON Files")
        uploaded_files = st.file_uploader(
            "Choose JSON files",
            type=['json'],
            accept_multiple_files=True,
            help="Upload JSON files containing request data (each file should contain one request)"
        )
        
        # URL inputs
        st.subheader("🌐 API Endpoints")
        url1 = st.text_input(
            "URL 1 (Base URL)",
            value="https://public-data.cbioportal.org/",
            help="Enter the first API base URL"
        )
        
        url2 = st.text_input(
            "URL 2 (Base URL)",
            value="https://clickhouse-only-db.cbioportal.org/",
            help="Enter the second API base URL"
        )
        
        # Request settings
        st.subheader("⚙️ Request Settings")
        timeout = st.slider("Request Timeout (seconds)", 5, 60, 30)
        num_requests = st.slider("Number of requests per file", 1, 10, 1)
    
    if uploaded_files and url1 and url2:
        st.header("📊 Results")
        
        # Initialize results storage
        if 'results' not in st.session_state:
            st.session_state.results = []
        
        # Process files button
        if st.button("🚀 Run Performance Tests", type="primary"):
            st.session_state.results = []
            
            # Progress tracking
            total_tests = len(uploaded_files) * num_requests * 2  # 2 URLs
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            test_count = 0
            
            for file in uploaded_files:
                # Read and parse file
                file_content = file.read().decode('utf-8')
                request_info = extract_request_data(file_content)
                
                if request_info:
                    # Build full URLs
                    full_url1 = build_full_url(url1, request_info['original_url'])
                    full_url2 = build_full_url(url2, request_info['original_url'])
                    
                    st.subheader(f"Testing: {request_info['api_path']}")
                    
                    # Display request info
                    with st.expander("Request Details"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Method:**", request_info['method'])
                            st.write("**JSON File:**", file.name)
                        with col2:
                            st.write("**Full URL 1:**", full_url1)
                            st.write("**Full URL 2:**", full_url2)
                        st.json(request_info['data'])
                    
                    # Show individual file results
                    file_results = []
                    
                    # Run multiple requests for each URL with parallel execution
                    for i in range(num_requests):
                        status_text.text(f"Testing {file.name} - Request {i+1}/{num_requests}")
                        
                        # Test both URLs in parallel using ThreadPoolExecutor
                        with ThreadPoolExecutor(max_workers=2) as executor:
                            # Submit both requests simultaneously
                            future1 = executor.submit(
                                make_request,
                                full_url1, 
                                request_info['data'], 
                                request_info['method'], 
                                timeout
                            )
                            future2 = executor.submit(
                                make_request,
                                full_url2, 
                                request_info['data'], 
                                request_info['method'], 
                                timeout
                            )
                            
                            # Get results as they complete
                            result1 = future1.result()
                            result2 = future2.result()
                        
                        test_count += 2
                        progress_bar.progress(test_count / total_tests)
                        
                        # Store results
                        st.session_state.results.extend([
                            {
                                'api_path': request_info['api_path'],
                                'url': 'URL 1',
                                'full_url': full_url1,
                                'request_num': i + 1,
                                'file': file.name,
                                **result1
                            },
                            {
                                'api_path': request_info['api_path'],
                                'url': 'URL 2',
                                'full_url': full_url2,
                                'request_num': i + 1,
                                'file': file.name,
                                **result2
                            }
                        ])
                        
                        # Store file-specific results for display
                        file_results.extend([result1, result2])
                    
                    # Display results for this file
                    if file_results:
                        url1_times = [r['response_time'] for i, r in enumerate(file_results) if i % 2 == 0]
                        url2_times = [r['response_time'] for i, r in enumerate(file_results) if i % 2 == 1]
                        
                        url1_avg = sum(url1_times) / len(url1_times) if url1_times else 0
                        url2_avg = sum(url2_times) / len(url2_times) if url2_times else 0
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("URL 1 Average", f"{url1_avg:.2f} ms")
                        with col2:
                            st.metric("URL 2 Average", f"{url2_avg:.2f} ms")
            
            status_text.text("✅ All tests completed!")
            progress_bar.progress(1.0)
        
        # Display results if available
        if st.session_state.results:
            # Convert to DataFrame
            df = pd.DataFrame(st.session_state.results)
            
            # Summary statistics
            st.subheader("📈 Summary Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_time_url1 = df[df['url'] == 'URL 1']['response_time'].mean()
                st.metric("URL 1 Avg Response Time", f"{avg_time_url1:.2f} ms")
            
            with col2:
                avg_time_url2 = df[df['url'] == 'URL 2']['response_time'].mean()
                st.metric("URL 2 Avg Response Time", f"{avg_time_url2:.2f} ms")
            
            with col3:
                success_rate_url1 = (df[df['url'] == 'URL 1']['success'].sum() / len(df[df['url'] == 'URL 1'])) * 100
                st.metric("URL 1 Success Rate", f"{success_rate_url1:.1f}%")
            
            with col4:
                success_rate_url2 = (df[df['url'] == 'URL 2']['success'].sum() / len(df[df['url'] == 'URL 2'])) * 100
                st.metric("URL 2 Success Rate", f"{success_rate_url2:.1f}%")
            
            # Visualization - Average response time bar chart
            st.subheader("📊 Average Response Time Comparison")
            
            # Calculate average response time per API path and URL
            avg_df = df.groupby(['api_path', 'url'])['response_time'].mean().reset_index()
            
            # Create bar chart
            fig_bar = px.bar(
                avg_df,
                x='api_path',
                y='response_time',
                color='url',
                title='Average Response Time per API Endpoint',
                labels={'response_time': 'Average Response Time (ms)', 'api_path': 'API Endpoint'},
                barmode='group'
            )
            fig_bar.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_bar, use_container_width=True)
            
            # Detailed results table
            st.subheader("📋 Detailed Results")
            
            # Pivot the data to show URL1 and URL2 in the same row
            pivot_data = []
            
            # Group by api_path and file (remove request_num grouping)
            grouped = df.groupby(['api_path', 'file'])
            
            for (api_path, file_name), group in grouped:
                # Calculate average response times for each URL
                url1_data = group[group['url'] == 'URL 1']
                url2_data = group[group['url'] == 'URL 2']
                
                if not url1_data.empty and not url2_data.empty:
                    url1_avg_time = url1_data['response_time'].mean()
                    url2_avg_time = url2_data['response_time'].mean()
                    time_diff = abs(url2_avg_time - url1_avg_time)
                    
                    # Get status (assuming all requests for same file have same status)
                    url1_status = f"✅ {url1_data.iloc[0]['status_code']}" if url1_data.iloc[0]['success'] else f"❌ {url1_data.iloc[0].get('error', 'Failed')}"
                    url2_status = f"✅ {url2_data.iloc[0]['status_code']}" if url2_data.iloc[0]['success'] else f"❌ {url2_data.iloc[0].get('error', 'Failed')}"
                    
                    row_data = {
                        'api_path': api_path,
                        'url_1_response_time': round(url1_avg_time, 2),
                        'url_1_status': url1_status,
                        'url_2_response_time': round(url2_avg_time, 2),
                        'url_2_status': url2_status,
                        'time_difference': round(time_diff, 2),
                        'file': file_name
                    }
                    
                    pivot_data.append(row_data)
            
            # Create pivot dataframe
            pivot_df = pd.DataFrame(pivot_data)
            
            # Select and rename columns for display
            display_columns = ['api_path', 'url_1_response_time', 'url_1_status', 'url_2_response_time', 'url_2_status', 'time_difference', 'file']
            column_config = {
                'api_path': 'API Endpoint',
                'url_1_response_time': st.column_config.NumberColumn(
                    'URL 1 Response Time (ms)',
                    format="%.2f ms"
                ),
                'url_1_status': 'URL 1 Status',
                'url_2_response_time': st.column_config.NumberColumn(
                    'URL 2 Response Time (ms)',
                    format="%.2f ms"
                ),
                'url_2_status': 'URL 2 Status',
                'time_difference': st.column_config.NumberColumn(
                    'Time Difference (ms)',
                    format="%.2f ms",
                    help="Absolute difference between URL 1 and URL 2 response times."
                ),
                'file': 'File'
            }
            
            st.dataframe(
                pivot_df[display_columns],
                column_config=column_config,
                hide_index=True,
                use_container_width=True
            )

    
    elif uploaded_files:
        st.info("✅ Files uploaded successfully! Please enter both URL endpoints to start testing.")
    else:
        st.info("Please upload JSON files and enter API endpoints to begin.")


if __name__ == "__main__":
    main()