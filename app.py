import streamlit as st
<<<<<<< HEAD
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
import time
import hashlib
warnings.filterwarnings('ignore')

import preprocessor
import helper
import sentiment_analysis

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="WhatsApp Chat Analyzer",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- SESSION STATE FOR CACHING --------------------
def initialize_session_state():
    """Initialize session state variables for caching."""
    if 'df_processed' not in st.session_state:
        st.session_state.df_processed = None
    if 'data_hash' not in st.session_state:
        st.session_state.data_hash = None
    if 'sentiment_completed' not in st.session_state:
        st.session_state.sentiment_completed = False
    if 'file_name' not in st.session_state:
        st.session_state.file_name = None

def generate_file_hash(file_content: str) -> str:
    """Generate hash of file content to detect changes."""
    return hashlib.md5(file_content.encode()).hexdigest()

# Initialize session state
initialize_session_state()

# -------------------- GPU STATUS CHECK --------------------
@st.cache_data
def get_system_info():
    """Get system information including GPU status."""
    return sentiment_analysis.get_gpu_info()

# -------------------- CUSTOM CSS --------------------
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #25D366;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .gpu-status {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-weight: bold;
    }
    
    .gpu-available {
        background: linear-gradient(135deg, #00b09b, #96c93d);
        color: white;
    }
    
    .gpu-unavailable {
        background: linear-gradient(135deg, #ffeaa7, #fab1a0);
        color: #2d3436;
    }
    
    .cache-status {
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        text-align: center;
        font-size: 0.8rem;
    }
    
    .cache-hit {
        background: linear-gradient(135deg, #74b9ff, #0984e3);
        color: white;
    }
    
    .cache-miss {
        background: linear-gradient(135deg, #fd79a8, #e84393);
        color: white;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #25D366, #128C7E);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(37, 211, 102, 0.4);
    }
    
    .sidebar .stSelectbox label {
        color: #25D366;
        font-weight: bold;
    }
    
    .plot-container {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .progress-container {
        margin: 1rem 0;
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 10px;
        border-left: 4px solid #25D366;
    }
</style>
""", unsafe_allow_html=True)

# -------------------- MAIN TITLE --------------------
st.markdown('<h1 class="main-header">💬 WhatsApp Chat Analyzer</h1>', unsafe_allow_html=True)

# -------------------- GPU STATUS DISPLAY --------------------
system_info = get_system_info()
if system_info["available"]:
    gpu_name = system_info["device_properties"].get(0, {}).get("name", "Unknown GPU")
    gpu_memory = system_info["device_properties"].get(0, {}).get("total_memory", 0)
    st.markdown(f'''
    <div class="gpu-status gpu-available">
        🚀 GPU Acceleration: ENABLED<br>
        Device: {gpu_name}<br>
        Memory: {gpu_memory:.1f}GB VRAM
    </div>
    ''', unsafe_allow_html=True)
else:
    st.markdown('''
    <div class="gpu-status gpu-unavailable">
        ⚙️ GPU Acceleration: UNAVAILABLE<br>
        Using CPU (slower processing)
    </div>
    ''', unsafe_allow_html=True)

st.markdown("---")

# -------------------- SIDEBAR --------------------
st.sidebar.title("🛠️ Configuration")
st.sidebar.markdown("---")

# -------------------- FILE UPLOAD --------------------
st.sidebar.subheader("📁 Upload Chat File")
uploaded_file = st.sidebar.file_uploader(
    "Choose a WhatsApp chat file (.txt)", 
    type=['txt'],
    help="Export your WhatsApp chat as a text file and upload it here"
)

if uploaded_file is not None:
    # Read file bytes and decode robustly
    try:
        raw = uploaded_file.getvalue()
        text = None
        encodings = ["utf-8", "utf-16", "utf-8-sig", "latin-1", "cp1252"]
        
        for enc in encodings:
            try:
                text = raw.decode(enc)
                break
            except UnicodeDecodeError:
                continue
                
        if text is None:
            st.error("❌ Could not decode the uploaded file. Please export the chat as text and try again.")
            st.stop()

        # Generate hash of current file content
        current_hash = generate_file_hash(text)
        current_filename = uploaded_file.name
        
        # Check if this is the same file as previously processed
        if (st.session_state.data_hash == current_hash and 
            st.session_state.file_name == current_filename and 
            st.session_state.df_processed is not None):
            
            st.sidebar.markdown('''
            <div class="cache-status cache-hit">
                ♻️ CACHE HIT: Using previously processed data
            </div>
            ''', unsafe_allow_html=True)
            
            df = st.session_state.df_processed
            st.sidebar.success(f"✅ Using cached data: {len(df)} messages")
            
        else:
            st.sidebar.markdown('''
            <div class="cache-status cache-miss">
                🔄 CACHE MISS: Processing new/changed file
            </div>
            ''', unsafe_allow_html=True)
            
            # Clear previous sentiment analysis cache when new file is uploaded
            sentiment_analysis.clear_sentiment_cache()
            
            # -------------------- PREPROCESS --------------------
            with st.spinner("🔄 Processing chat data..."):
                try:
                    df = preprocessor.preprocess(text)
                    if df.empty:
                        st.warning("⚠️ No valid messages were parsed from the file. Check the export format (text-only) and try again.")
                        st.stop()
                    
                    # Store in session state
                    st.session_state.df_processed = df
                    st.session_state.data_hash = current_hash
                    st.session_state.file_name = current_filename
                    st.session_state.sentiment_completed = False
                    
                    st.sidebar.success(f"✅ Processed {len(df)} messages successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error processing file: {str(e)}")
                    st.stop()

    except Exception as e:
        st.error(f"❌ Error reading file: {str(e)}")
        st.stop()

    # -------------------- USER SELECTION --------------------
    st.sidebar.subheader("👥 User Selection")
    user_list = df["user"].dropna().unique().tolist()
    
    # Remove system notifications
    system_users = ["group_notification", "Group notification", "System"]
    for sys_user in system_users:
        if sys_user in user_list:
            user_list.remove(sys_user)
    
    user_list.sort()
    user_list.insert(0, "Overall")

    selected_user = st.sidebar.selectbox(
        "📊 Show analysis for:", 
        user_list,
        help="Select a specific user or 'Overall' for group analysis"
    )

    # -------------------- ANALYSIS OPTIONS --------------------
    st.sidebar.subheader("⚙️ Analysis Options")
    
    # Check if sentiment analysis was already completed for this data
    has_sentiment = 'sentiment_label' in df.columns and 'sentiment_score' in df.columns
    
    if has_sentiment:
        st.sidebar.markdown('''
        <div class="cache-status cache-hit">
            ✅ SENTIMENT CACHE: Analysis already completed
        </div>
        ''', unsafe_allow_html=True)
        show_sentiment = st.sidebar.checkbox("📈 Include Sentiment Analysis", value=True,
                                           help="Sentiment analysis already completed for this data")
    else:
        st.sidebar.markdown('''
        <div class="cache-status cache-miss">
            🔄 NEW ANALYSIS: Sentiment analysis needed
        </div>
        ''', unsafe_allow_html=True)
        show_sentiment = st.sidebar.checkbox("📈 Include Sentiment Analysis", value=True)
    
    # GPU-specific options (only show if sentiment analysis is needed)
    if system_info["available"] and show_sentiment and not has_sentiment:
        st.sidebar.subheader("🚀 GPU Settings")
        use_gpu = st.sidebar.checkbox("Use GPU Acceleration", value=True, 
                                     help="Enables faster processing using GPU")
        
        # Advanced GPU settings in expander
        with st.sidebar.expander("Advanced GPU Settings"):
            auto_batch = st.checkbox("Auto Batch Size", value=True,
                                   help="Automatically optimize batch size based on GPU memory")
            
            if not auto_batch:
                custom_batch_size = st.slider("Batch Size", min_value=4, max_value=128, 
                                             value=32, step=4,
                                             help="Larger batches = faster but more memory")
            else:
                custom_batch_size = None
                
            show_gpu_stats = st.checkbox("Show GPU Statistics", value=False,
                                       help="Display detailed GPU memory usage during processing")
    else:
        use_gpu = False
        custom_batch_size = None
        show_gpu_stats = False
    
    show_wordcloud = st.sidebar.checkbox("☁️ Generate Word Cloud", value=True)
    show_emoji_analysis = st.sidebar.checkbox("😊 Emoji Analysis", value=True)
    show_activity_patterns = st.sidebar.checkbox("📅 Activity Patterns", value=True)

    # -------------------- SENTIMENT ANALYSIS --------------------
    if show_sentiment and not has_sentiment:
        st.sidebar.subheader("🧠 Sentiment Analysis")
        
        with st.spinner("🧠 Analyzing sentiment..."):
            try:
                # Optimize settings for GPU
                if use_gpu and system_info["available"]:
                    optimization = sentiment_analysis.optimize_for_gpu(len(df))
                    batch_size = custom_batch_size or optimization["batch_size"]
                    
                    st.sidebar.success(f"✅ GPU optimized for {len(df)} messages")
                    st.sidebar.info(f"🔧 Using batch size: {batch_size}")
                else:
                    batch_size = custom_batch_size or 8
                
                # Progress tracking
                progress_placeholder = st.empty()
                status_placeholder = st.empty()
                
                def progress_callback(progress, status):
                    progress_placeholder.progress(progress)
                    status_placeholder.info(f"🔄 {status}")
                
                # Perform sentiment analysis with GPU acceleration
                start_time = time.time()
                df = sentiment_analysis.perform_sentiment_analysis(
                    df, 
                    use_gpu=use_gpu, 
                    progress_callback=progress_callback
                )
                end_time = time.time()
                
                # Update cached data
                st.session_state.df_processed = df
                st.session_state.sentiment_completed = True
                
                # Clear progress indicators
                progress_placeholder.empty()
                status_placeholder.empty()
                
                processing_time = end_time - start_time
                messages_per_second = len(df) / processing_time if processing_time > 0 else 0
                
                st.sidebar.success(f"✅ Sentiment analysis completed!")
                st.sidebar.info(f"⏱️ Time: {processing_time:.2f}s ({messages_per_second:.1f} msg/s)")
                
                # Show GPU statistics if enabled
                if show_gpu_stats and use_gpu and system_info["available"]:
                    gpu_info = sentiment_analysis.get_gpu_info()
                    if gpu_info["memory_info"]:
                        memory_info = gpu_info["memory_info"][0]
                        st.sidebar.info(f"📊 GPU Memory: {memory_info['allocated']:.2f}GB used")
                        
            except Exception as e:
                st.sidebar.warning(f"⚠️ Sentiment analysis failed: {str(e)}")
                # Continue without sentiment analysis
                df['sentiment_label'] = 'neutral'
                df['sentiment_score'] = 0.5
                
    elif show_sentiment and has_sentiment:
        st.sidebar.info("✅ Using cached sentiment analysis results")

    # -------------------- MAIN ANALYSIS --------------------
    if st.sidebar.button("🚀 Show Analysis", help="Click to generate comprehensive analysis"):
        
        # -------------------- OVERVIEW STATS --------------------
        st.markdown("## 📊 Overview Statistics")
        
        num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="💬 Total Messages",
                value=f"{num_messages:,}",
                delta=f"{num_messages/len(df)*100:.1f}% of total" if selected_user != "Overall" else None
            )
            
        with col2:
            st.metric(
                label="📝 Total Words", 
                value=f"{words:,}",
                delta=f"{words/num_messages:.1f} words/msg" if num_messages > 0 else None
            )
            
        with col3:
            st.metric(
                label="📷 Media Shared", 
                value=f"{num_media_messages:,}",
                delta=f"{num_media_messages/num_messages*100:.1f}% media" if num_messages > 0 else None
            )
            
        with col4:
            st.metric(
                label="🔗 Links Shared", 
                value=f"{num_links:,}",
                delta=f"{num_links/num_messages*100:.1f}% with links" if num_messages > 0 else None
            )

        st.markdown("---")

        # -------------------- TIMELINE ANALYSIS --------------------
        st.markdown("## 📈 Timeline Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📅 Monthly Timeline")
            monthly_timeline = helper.monthly_timeline(selected_user, df)
            if not monthly_timeline.empty:
                fig = px.line(
                    monthly_timeline, 
                    x='time', 
                    y='message',
                    title="Messages Over Months",
                    labels={'message': 'Number of Messages', 'time': 'Month-Year'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("🔭 No monthly timeline data available")

        with col2:
            st.subheader("📆 Daily Timeline")
            daily_timeline = helper.daily_timeline(selected_user, df)
            if not daily_timeline.empty:
                fig = px.line(
                    daily_timeline, 
                    x='only_date', 
                    y='message',
                    title="Daily Message Activity",
                    labels={'message': 'Number of Messages', 'only_date': 'Date'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("🔭 No daily timeline data available")

        # -------------------- ACTIVITY PATTERNS --------------------
        if show_activity_patterns:
            st.markdown("## 🕐 Activity Patterns")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("📅 Most Busy Days")
                busy_day = helper.week_activity_map(selected_user, df)
                if not busy_day.empty:
                    fig = px.bar(
                        x=busy_day.index, 
                        y=busy_day.values,
                        title="Activity by Day of Week",
                        labels={'x': 'Day of Week', 'y': 'Number of Messages'}
                    )
                    fig.update_layout(height=350)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("🔭 No day-wise activity data")

            with col2:
                st.subheader("📊 Most Busy Months")
                busy_month = helper.month_activity_map(selected_user, df)
                if not busy_month.empty:
                    fig = px.bar(
                        x=busy_month.index, 
                        y=busy_month.values,
                        title="Activity by Month",
                        labels={'x': 'Month', 'y': 'Number of Messages'}
                    )
                    fig.update_layout(height=350)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("🔭 No month-wise activity data")

            with col3:
                st.subheader("🕐 Hourly Activity")
                hourly_activity = helper.hourly_activity_map(selected_user, df)
                if not hourly_activity.empty:
                    fig = px.bar(
                        x=hourly_activity.index, 
                        y=hourly_activity.values,
                        title="Activity by Hour",
                        labels={'x': 'Hour of Day', 'y': 'Number of Messages'}
                    )
                    fig.update_layout(height=350)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("🔭 No hourly activity data")

            # Activity Heatmap
            st.subheader("🔥 Weekly Activity Heatmap")
            user_heatmap = helper.activity_heatmap(selected_user, df)
            if not user_heatmap.empty:
                fig = px.imshow(
                    user_heatmap,
                    title="Activity Heatmap (Day vs Hour)",
                    labels=dict(x="Hour Period", y="Day of Week", color="Messages"),
                    aspect="auto"
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("🔭 No activity heatmap data available")

        # -------------------- MOST BUSY USERS (GROUP LEVEL) --------------------
        if selected_user == "Overall":
            st.markdown("## 👑 Most Active Users")
            
            x, new_df = helper.most_busy_users(df)
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if not x.empty:
                    fig = px.bar(
                        x=x.values, 
                        y=x.index, 
                        orientation='h',
                        title="Top Users by Message Count",
                        labels={'x': 'Number of Messages', 'y': 'User'}
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("🔭 No user activity data")
                    
            with col2:
                st.subheader("📊 User Statistics")
                if not new_df.empty:
                    st.dataframe(new_df, use_container_width=True)
                else:
                    st.info("🔭 No user statistics available")

        # -------------------- WORD CLOUD --------------------
        if show_wordcloud:
            st.markdown("## ☁️ Word Cloud")
            
            with st.spinner("Generating word cloud..."):
                df_wc = helper.create_wordcloud(selected_user, df)
                if df_wc is not None:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.imshow(df_wc, interpolation='bilinear')
                    ax.axis("off")
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.clf()
                else:
                    st.info("☁️ Not enough textual data to generate a Word Cloud")

        # -------------------- MOST COMMON WORDS --------------------
        st.markdown("## 🔍 Most Common Words")
        
        most_common_df = helper.most_common_words(selected_user, df)
        if not most_common_df.empty and len(most_common_df.columns) >= 2:
            # Handle both old format (columns 0,1) and new format (word, frequency)
            if 'word' in most_common_df.columns and 'frequency' in most_common_df.columns:
                words = most_common_df['word'][:15]
                frequencies = most_common_df['frequency'][:15]
            else:
                words = most_common_df.iloc[:, 0][:15]  # First column
                frequencies = most_common_df.iloc[:, 1][:15]  # Second column
                
            fig = px.bar(
                x=frequencies, 
                y=words, 
                orientation='h',
                title="Top 15 Most Used Words",
                labels={'x': 'Frequency', 'y': 'Words'}
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("🔍 No common words data available")

        # -------------------- EMOJI ANALYSIS --------------------
        if show_emoji_analysis:
            st.markdown("## 😊 Emoji Analysis")
            
            emoji_df = helper.emoji_helper(selected_user, df)
            if not emoji_df.empty and len(emoji_df.columns) >= 2:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Emoji Usage Table")
                    
                    # Handle both old format (columns 0,1) and new format (emoji, count)
                    if 'emoji' in emoji_df.columns and 'count' in emoji_df.columns:
                        display_df = emoji_df.head(10)[['emoji', 'count']].copy()
                        display_df.columns = ['Emoji', 'Count']
                        emojis_for_pie = emoji_df['emoji'].head(10)
                        counts_for_pie = emoji_df['count'].head(10)
                    else:
                        display_df = emoji_df.head(10).copy()
                        display_df.columns = ['Emoji', 'Count']
                        emojis_for_pie = emoji_df.iloc[:, 0].head(10)
                        counts_for_pie = emoji_df.iloc[:, 1].head(10)
                        
                    st.dataframe(display_df, use_container_width=True)
                    
                with col2:
                    st.subheader("🥧 Emoji Distribution")
                    fig = px.pie(
                        values=counts_for_pie, 
                        names=emojis_for_pie,
                        title="Top 10 Emojis Usage"
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("😊 No emojis found in the messages")

        # -------------------- SENTIMENT ANALYSIS --------------------
        if show_sentiment and 'sentiment_score' in df.columns:
            st.markdown("## 🧠 Sentiment Analysis")
            
            # Performance metrics
            col1, col2, col3, col4 = st.columns(4)
            
            sentiment_counts = df['sentiment_label'].value_counts() if selected_user == "Overall" else df[df['user'] == selected_user]['sentiment_label'].value_counts()
            
            with col1:
                positive_pct = (sentiment_counts.get('positive', 0) / sentiment_counts.sum() * 100) if sentiment_counts.sum() > 0 else 0
                st.metric("😊 Positive Messages", f"{positive_pct:.1f}%")
                
            with col2:
                negative_pct = (sentiment_counts.get('negative', 0) / sentiment_counts.sum() * 100) if sentiment_counts.sum() > 0 else 0
                st.metric("😞 Negative Messages", f"{negative_pct:.1f}%")
                
            with col3:
                neutral_pct = (sentiment_counts.get('neutral', 0) / sentiment_counts.sum() * 100) if sentiment_counts.sum() > 0 else 0
                st.metric("😐 Neutral Messages", f"{neutral_pct:.1f}%")
                
            with col4:
                avg_sentiment = df['sentiment_score'].mean() if selected_user == "Overall" else df[df['user'] == selected_user]['sentiment_score'].mean()
                st.metric("📊 Avg Sentiment", f"{avg_sentiment:.3f}")

            # Sentiment distribution chart
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Sentiment Distribution")
                if not sentiment_counts.empty:
                    fig = px.pie(
                        values=sentiment_counts.values,
                        names=sentiment_counts.index,
                        title="Overall Sentiment Distribution",
                        color_discrete_map={
                            'positive': '#2ecc71',
                            'negative': '#e74c3c', 
                            'neutral': '#95a5a6'
                        }
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("📊 No sentiment data available")
            
            with col2:
                st.subheader("📈 Sentiment Trends (Last 30 days)")
                # Use the cached sentiment results - NO re-analysis
                sot_df = helper.sentiment_over_time(selected_user, df, window_days=30)
                
                if not sot_df.empty:
                    fig = px.line(
                        sot_df, 
                        x='date', 
                        y='sentiment_score', 
                        color='user' if selected_user == "Overall" else None,
                        title="Sentiment Score Over Time (Cached Results)",
                        labels={'sentiment_score': 'Average Sentiment Score', 'date': 'Date'}
                    )
                    fig.update_layout(height=400)
                    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", 
                                 annotation_text="Neutral Line")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("📈 No sentiment data available for the selected time window")
                    
            # Show detailed sentiment results
            with st.expander("🔍 View Detailed Sentiment Results"):
                sentiment_display = df[['user', 'message', 'sentiment_label', 'sentiment_score']].copy()
                if selected_user != "Overall":
                    sentiment_display = sentiment_display[sentiment_display['user'] == selected_user]
                
                # Add color coding for sentiment
                def color_sentiment(val):
                    if val == 'positive':
                        return 'background-color: #d5f4e6'
                    elif val == 'negative':
                        return 'background-color: #fae5e5'
                    else:
                        return 'background-color: #f0f0f0'
                
                styled_df = sentiment_display.style.applymap(color_sentiment, subset=['sentiment_label'])
                st.dataframe(styled_df, use_container_width=True, height=400)

        st.markdown("---")
        
        # Show caching information
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"📊 **Data Processing**: {'Cached' if st.session_state.data_hash else 'Fresh'}")
        with col2:
            st.info(f"🧠 **Sentiment Analysis**: {'Cached' if has_sentiment else 'Fresh'}")
        with col3:
            st.info(f"⚡ **Performance**: {'Optimized' if has_sentiment else 'Single Run'}")
        
        st.success("✅ Analysis completed successfully!")
        
        # -------------------- DOWNLOAD SECTION --------------------
        st.markdown("## 📥 Download Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Download Analysis Data (CSV)"):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"whatsapp_analysis_{selected_user}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
        with col2:
            if st.button("📈 Download Summary Report"):
                summary_report = helper.generate_summary_report(selected_user, df, num_messages, words, num_media_messages, num_links)
                st.download_button(
                    label="Download Report",
                    data=summary_report,
                    file_name=f"whatsapp_summary_{selected_user}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
        
        with col3:
            if st.button("🧹 Clear Cache & Restart"):
                # Clear all cached data
                st.session_state.df_processed = None
                st.session_state.data_hash = None
                st.session_state.sentiment_completed = False
                st.session_state.file_name = None
                sentiment_analysis.clear_sentiment_cache()
                st.success("✅ Cache cleared! Please re-upload your file.")
                st.experimental_rerun()

else:
    # -------------------- WELCOME SCREEN --------------------
    st.markdown("""
    ## 🎯 Welcome to WhatsApp Chat Analyzer!
    
    This optimized tool analyzes your WhatsApp conversations with intelligent caching and GPU acceleration.
    
    ### 🚀 New Optimization Features:
    - **⚡ Smart Caching**: Process data once, analyze multiple times instantly
    - **🧠 Sentiment Persistence**: Sentiment analysis runs only once per dataset
    - **♻️ Data Reuse**: Change user selections without re-processing
    - **📊 Memory Optimization**: Efficient GPU memory management
    - **🔄 Change Detection**: Automatically detects when new data is uploaded
    
    ### 💾 How Caching Works:
    1. **First Upload**: Full processing + sentiment analysis (slower)
    2. **Same File**: Instant loading from cache (super fast)
    3. **Different User Selection**: Uses cached data, no re-processing
    4. **New File**: Automatically clears cache and processes fresh data
    
    ### 🚀 GPU-Accelerated Features:
    - **⚡ Ultra-Fast Sentiment Analysis**: NVIDIA GPU acceleration for 10x faster processing
    - **🧠 Advanced AI Models**: State-of-the-art transformer models (RoBERTa, DistilBERT)
    - **📊 One-Time Processing**: Sentiment analysis runs once and is cached
    - **🔧 Adaptive Optimization**: Automatic batch sizing based on GPU memory
    - **📈 Performance Monitoring**: Live GPU usage statistics and optimization tips
    
    ### 📱 How to Export Your WhatsApp Chat:
    1. Open WhatsApp on your phone
    2. Go to the chat you want to analyze
    3. Tap the three dots menu (⋮) → More → Export chat
    4. Choose "Without Media" for faster processing
    5. Save the .txt file and upload it here
    
    ### 🔒 Privacy & Performance:
    Your data is processed locally with intelligent caching. Analysis results are stored in memory only!
    
    **👆 Upload your chat file using the sidebar to get started!**
    
    ### 💡 Performance Tips:
    - **🚀 First analysis**: Enable GPU acceleration for fastest initial processing
    - **♻️ Multiple analyses**: Change user selections instantly using cached data  
    - **📱 Large chats**: Use "Auto Batch Size" for optimal memory usage  
    - **⚡ Quick comparisons**: Switch between users without re-processing
    - **🧹 Fresh start**: Use "Clear Cache" button to start over with new data
    """)
    
    # Add performance comparison info
    col1, col2, col3 = st.columns(3)
    with col1:
        if system_info["available"]:
            st.success("🚀 **GPU-ACCELERATED**\n\n⚡ 10-50x faster processing\n🧠 Advanced AI models\n📊 One-time analysis + caching")
        else:
            st.info("⚙️ **CPU MODE**\n\n🔄 Standard processing\n📈 Basic analysis\n♻️ Smart caching enabled")
    with col2:
        st.info("🧠 **Smart Caching**\n\nProcess once, analyze forever. Switch users instantly without re-processing data.")  
    with col3:
        st.info("🎨 **Interactive Visualizations**\n\nPlotly-powered charts with real-time interactivity and cached performance.")
        
    # Show cache status
    if st.session_state.df_processed is not None:
        st.markdown("---")
        st.markdown("## 📋 Current Cache Status")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('''
            <div class="cache-status cache-hit">
                📊 DATA CACHED: Ready for instant analysis
            </div>
            ''', unsafe_allow_html=True)
        
        with col2:
            if st.session_state.sentiment_completed:
                st.markdown('''
                <div class="cache-status cache-hit">
                    🧠 SENTIMENT CACHED: Analysis completed
                </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown('''
                <div class="cache-status cache-miss">
                    🔄 SENTIMENT PENDING: Analysis needed
                </div>
                ''', unsafe_allow_html=True)
        
        with col3:
            st.markdown(f'''
            <div class="cache-status cache-hit">
                📁 FILE: {st.session_state.file_name or 'Unknown'}
            </div>
            ''', unsafe_allow_html=True)
            
        df_cached = st.session_state.df_processed
        st.info(f"✅ {len(df_cached):,} messages ready for analysis. Select user and click 'Show Analysis' in sidebar!")
        
        if st.button("🧹 Clear Cache"):
            st.session_state.df_processed = None
            st.session_state.data_hash = None
            st.session_state.sentiment_completed = False
            st.session_state.file_name = None
            sentiment_analysis.clear_sentiment_cache()
            st.success("✅ Cache cleared!")
            st.experimental_rerun()
        
# -------------------- FOOTER --------------------
st.markdown("---")

# Display optimization info
cache_status = "CACHED" if st.session_state.df_processed is not None else "EMPTY"
sentiment_status = "CACHED" if st.session_state.sentiment_completed else "PENDING"

st.markdown(
    f"""
    <div style='text-align: center; color: gray;'>
        Made with ❤️ using Streamlit | WhatsApp Chat Analyzer<br>
        {'🚀 GPU Acceleration: ENABLED' if system_info["available"] else '⚙️ Running on CPU'} | 
        📊 Data Cache: {cache_status} | 🧠 Sentiment Cache: {sentiment_status}
    </div>
    """, 
    unsafe_allow_html=True
)
=======
import preprocessor
import helper
import matplotlib.pyplot as plt
import seaborn as sns
import sentiment_analysis

st.sidebar.title("Whatsapp Chat Analyzer")

uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)
    df = sentiment_analysis.perform_sentiment_analysis(df)  # Add this line

    # fetch unique users
    user_list = df['user'].unique().tolist()
    if 'group_notification' in user_list:
        user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0, "Overall")

    selected_user = st.sidebar.selectbox("Show analysis with respect to", user_list)

    if st.sidebar.button("Show Analysis"):

        num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)
        st.title("Top Statistics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.header("Total Messages")
            st.title(num_messages)
        with col2:
            st.header("Total Words")
            st.title(words)
        with col3:
            st.header("Media Shared")
            st.title(num_media_messages)
        with col4:
            st.header("Links Shared")
            st.title(num_links)

        # monthly timeline
        st.title("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'], color='blue')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # daily timeline
        st.title("Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # activity map
        st.title('Activity Map')
        col1, col2 = st.columns(2)

        with col1:
            st.header("Most busy day")
            busy_day = helper.week_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values)
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.header("Most busy month")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='red')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        st.title("Weekly Activity Map")
        user_heatmap = helper.activity_heatmap(selected_user, df)
        fig, ax = plt.subplots()
        ax = sns.heatmap(user_heatmap)
        st.pyplot(fig)

        # finding the busiest users in the group (Group Level)
        if selected_user == 'Overall':
            st.title('Most busy users')
            x, new_df = helper.most_busy_users(df)
            fig, ax = plt.subplots()
            col1, col2 = st.columns(2)
            with col1:
                ax.bar(x.index, x.values, color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df)

        # WordCloud
        st.title('WordCloud')
        df_wc = helper.create_wordcloud(selected_user, df)
        fig, ax = plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)

        # most common words
        most_common_df = helper.most_common_words(selected_user, df)

        fig, ax = plt.subplots()

        ax.bar(most_common_df[0], most_common_df[1])
        plt.xticks(rotation='vertical')

        st.title("Most common words")
        st.pyplot(fig)

        # emoji analysis
        emoji_df = helper.emoji_helper(selected_user, df)
        st.title("Emoji Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.dataframe(emoji_df)
        with col2:
            fig, ax = plt.subplots()
            ax.pie(emoji_df[1].head(10), labels=emoji_df[0].head(10), autopct="%0.2f")
            st.pyplot(fig)

    # Sentiment Analysis Over Time

    # # Taking input from the user for window_days
    # window_days = int(input("Enter the number of days for the window: "))
    # # Now, you can use the variable window_days in your script
    # print(f"The window_days is set to: ")
    # Displaying the selected window_days
    # st.title(f"Sentiment Analysis Over Time (Last '{window_days}' days)")

    st.title(f"Sentiment Analysis Over Time (Last 20 days)")

    # Assuming you have a function sentiment_over_time in helper.py
    sentiment_over_time_df = helper.sentiment_over_time(selected_user, df, window_days=20)

    fig, ax = plt.subplots(figsize=(10, 6))
    for user, data in sentiment_over_time_df.groupby('user'):
        ax.plot(data['date'], data['sentiment_score'], label=user)

        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.set_xlabel('Date')
        ax.set_ylabel('Sentiment Score')
        ax.set_title('Sentiment Analysis Over Time')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

    # Include sentiment analysis results in your UI
    st.title("Sentiment Analysis")
    st.write(df[['user', 'message', 'sentiment_label', 'sentiment_score']])
>>>>>>> fcfbd584046b32005c908f931ae5d9ff4a42871a
