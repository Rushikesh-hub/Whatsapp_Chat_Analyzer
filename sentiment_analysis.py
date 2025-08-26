<<<<<<< HEAD
import pandas as pd
import numpy as np
from functools import lru_cache
from typing import List, Dict, Any, Tuple, Optional
import re
import hashlib
import warnings
warnings.filterwarnings('ignore')

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
    
    # Check GPU availability
    if torch.cuda.is_available():
        GPU_AVAILABLE = True
        DEVICE = 0  # Use first GPU
        GPU_COUNT = torch.cuda.device_count()
        GPU_NAME = torch.cuda.get_device_name(0)
        print(f"🚀 GPU acceleration available: {GPU_NAME}")
        print(f"📊 Total GPUs: {GPU_COUNT}")
    else:
        GPU_AVAILABLE = False
        DEVICE = -1  # Use CPU
        print("⚠️ GPU not available, using CPU")
        
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    GPU_AVAILABLE = False
    DEVICE = -1
    print("Warning: transformers library not available. Using fallback sentiment analysis.")

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

# Model configurations
MODEL_CONFIGS = {
    "primary": "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "backup": "distilbert-base-uncased-finetuned-sst-2-english",
    "multilingual": "nlptown/bert-base-multilingual-uncased-sentiment"
}

class SentimentAnalyzer:
    """Enhanced sentiment analyzer with GPU acceleration and caching."""
    
    def __init__(self, use_gpu: bool = True, batch_size: int = None):
        self.pipeline = None
        self.tokenizer = None
        self.model_name = None
        self.fallback_mode = False
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.device = DEVICE if self.use_gpu else -1
        self.is_initialized = False
        
        # Cache for analyzed results
        self.analysis_cache = {}
        self.current_data_hash = None
        
        # Dynamic batch sizing based on available resources
        if batch_size is None:
            if self.use_gpu:
                # Larger batches for GPU
                if GPU_AVAILABLE and torch.cuda.get_device_properties(0).total_memory > 8e9:  # >8GB VRAM
                    self.batch_size = 64
                elif GPU_AVAILABLE and torch.cuda.get_device_properties(0).total_memory > 4e9:  # >4GB VRAM
                    self.batch_size = 32
                else:
                    self.batch_size = 16
            else:
                self.batch_size = 8  # Smaller batches for CPU
        else:
            self.batch_size = batch_size
    
    def _generate_data_hash(self, messages: List[str]) -> str:
        """Generate a hash for the input data to detect changes."""
        data_string = ''.join(sorted(messages))
        return hashlib.md5(data_string.encode()).hexdigest()
    
    def _initialize_model_once(self):
        """Initialize the model only once and cache it."""
        if self.is_initialized:
            print("📋 Model already initialized, reusing existing pipeline")
            return
            
        if not TRANSFORMERS_AVAILABLE:
            self.fallback_mode = True
            self.is_initialized = True
            print("Using fallback sentiment analysis (basic lexicon-based)")
            return
        
        # Try models in order of preference
        for model_key, model_name in MODEL_CONFIGS.items():
            try:
                print(f"🔄 Loading {model_name} on {'GPU' if self.use_gpu else 'CPU'}...")
                
                if self.use_gpu:
                    # GPU-optimized initialization
                    self.pipeline = pipeline(
                        "sentiment-analysis",
                        model=model_name,
                        tokenizer=model_name,
                        device=self.device,
                        torch_dtype=torch.float16,  # Use half precision for faster inference
                        return_all_scores=True,
                        framework="pt"  # Explicitly use PyTorch
                    )
                    
                    # Warm up the GPU model
                    try:
                        _ = self.pipeline(["test message"])
                        print(f"🚀 GPU warmup successful for {model_name}")
                    except Exception as e:
                        print(f"⚠️ GPU warmup failed: {e}")
                        
                else:
                    # CPU initialization
                    self.pipeline = pipeline(
                        "sentiment-analysis",
                        model=model_name,
                        tokenizer=model_name,
                        device=self.device,
                        return_all_scores=True
                    )
                
                self.model_name = model_name
                self.is_initialized = True
                print(f"✅ Successfully loaded {model_name} on {'GPU' if self.use_gpu else 'CPU'}")
                
                # Print memory usage if using GPU
                if self.use_gpu and GPU_AVAILABLE:
                    allocated = torch.cuda.memory_allocated(0) / 1024**3
                    cached = torch.cuda.memory_reserved(0) / 1024**3
                    print(f"📊 GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")
                
                return
                
            except Exception as e:
                print(f"❌ Failed to load {model_name}: {e}")
                # Clear GPU memory on failure
                if self.use_gpu and GPU_AVAILABLE:
                    torch.cuda.empty_cache()
                continue
        
        # If all models fail, use fallback
        print("All transformer models failed. Using fallback sentiment analysis.")
        self.fallback_mode = True
        self.is_initialized = True
    
    def analyze_batch(self, messages: List[str], custom_batch_size: int = None) -> List[Dict[str, Any]]:
        """Analyze sentiment for a batch of messages with caching."""
        if not messages:
            return []
        
        # Initialize model only once
        self._initialize_model_once()
        
        # Generate hash for caching
        data_hash = self._generate_data_hash(messages)
        
        # Check if we have cached results for this exact data
        if data_hash in self.analysis_cache:
            print("📋 Using cached sentiment analysis results")
            return self.analysis_cache[data_hash]
        
        # Clean messages
        cleaned_messages = [self._clean_message(msg) for msg in messages]
        cleaned_messages = [msg for msg in cleaned_messages if msg.strip()]
        
        if not cleaned_messages:
            results = [{"label": "neutral", "score": 0.5} for _ in messages]
            self.analysis_cache[data_hash] = results
            return results
        
        if self.fallback_mode:
            results = self._fallback_analysis(cleaned_messages)
            self.analysis_cache[data_hash] = results
            return results
        
        try:
            # Use custom batch size or default
            batch_size = custom_batch_size or self.batch_size
            
            # Process in batches to optimize GPU memory usage
            results = []
            total_batches = (len(cleaned_messages) + batch_size - 1) // batch_size
            
            for i in range(0, len(cleaned_messages), batch_size):
                batch = cleaned_messages[i:i + batch_size]
                batch_num = i // batch_size + 1
                
                if self.use_gpu and GPU_AVAILABLE:
                    # Monitor GPU memory before processing
                    if batch_num % 10 == 0:  # Check every 10 batches
                        allocated = torch.cuda.memory_allocated(0) / 1024**3
                        if allocated > 6:  # If using >6GB, reduce batch size
                            batch_size = max(batch_size // 2, 4)
                            print(f"⚠️ Reducing batch size to {batch_size} due to memory usage")
                
                print(f"📊 Processing batch {batch_num}/{total_batches} (size: {len(batch)})")
                
                try:
                    batch_results = self._analyze_transformer_batch(batch)
                    results.extend(batch_results)
                    
                    # Clear GPU cache periodically
                    if self.use_gpu and GPU_AVAILABLE and batch_num % 20 == 0:
                        torch.cuda.empty_cache()
                        
                except torch.cuda.OutOfMemoryError:
                    print("❌ GPU out of memory! Reducing batch size and retrying...")
                    torch.cuda.empty_cache()
                    
                    # Retry with smaller batch size
                    smaller_batch_size = max(batch_size // 4, 1)
                    for j in range(0, len(batch), smaller_batch_size):
                        mini_batch = batch[j:j + smaller_batch_size]
                        mini_results = self._analyze_transformer_batch(mini_batch)
                        results.extend(mini_results)
                
                except Exception as e:
                    print(f"❌ Error processing batch {batch_num}: {e}")
                    # Fallback for this batch
                    fallback_results = self._fallback_analysis(batch)
                    results.extend(fallback_results)
            
            # Ensure we have results for all original messages
            while len(results) < len(messages):
                results.append({"label": "neutral", "score": 0.5})
            
            final_results = results[:len(messages)]
            
            # Cache the results
            self.analysis_cache[data_hash] = final_results
            
            return final_results
            
        except Exception as e:
            print(f"Error in transformer analysis: {e}")
            if self.use_gpu and GPU_AVAILABLE:
                torch.cuda.empty_cache()
            results = self._fallback_analysis(cleaned_messages)
            self.analysis_cache[data_hash] = results
            return results
    
    def _analyze_transformer_batch(self, messages: List[str]) -> List[Dict[str, Any]]:
        """Analyze using transformer model with GPU optimization."""
        try:
            # Truncate messages to avoid token limits (GPU can handle longer sequences)
            max_length = 512 if self.use_gpu else 256
            truncated_messages = [msg[:max_length] for msg in messages]
            
            with torch.no_grad():  # Disable gradient computation for inference
                raw_results = self.pipeline(truncated_messages)
            
            processed_results = []
            for result in raw_results:
                if isinstance(result, list):
                    # Handle models that return all scores
                    best_result = max(result, key=lambda x: x['score'])
                    label = self._normalize_label(best_result['label'])
                    score = float(best_result['score'])
                else:
                    # Handle models that return single result
                    label = self._normalize_label(result['label'])
                    score = float(result['score'])
                
                processed_results.append({
                    "label": label,
                    "score": score
                })
            
            return processed_results
            
        except Exception as e:
            print(f"Transformer batch analysis failed: {e}")
            return self._fallback_analysis(messages)
    
    def _normalize_label(self, label: str) -> str:
        """Normalize different model label formats."""
        label_lower = label.lower()
        
        if label_lower in ['positive', 'pos', 'label_2', '2']:
            return "positive"
        elif label_lower in ['negative', 'neg', 'label_0', '0']:
            return "negative"
        else:
            return "neutral"
    
    def _clean_message(self, message: str) -> str:
        """Clean message for sentiment analysis."""
        if pd.isna(message):
            return ""
        
        message = str(message)
        
        # Remove media placeholders
        message = re.sub(r'<media omitted>|image omitted|video omitted|audio omitted|document omitted', '', message, flags=re.IGNORECASE)
        message = re.sub(r'this message was deleted', '', message, flags=re.IGNORECASE)
        
        # Remove URLs
        message = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', message)
        
        # Remove mentions and hashtags for cleaner sentiment
        message = re.sub(r'@\w+|#\w+', '', message)
        
        # Clean up whitespace
        message = re.sub(r'\s+', ' ', message).strip()
        
        return message
    
    def _fallback_analysis(self, messages: List[str]) -> List[Dict[str, Any]]:
        """Fallback sentiment analysis using simple methods."""
        results = []
        
        for message in messages:
            try:
                if TEXTBLOB_AVAILABLE:
                    # Use TextBlob if available
                    blob = TextBlob(message)
                    polarity = blob.sentiment.polarity
                    
                    if polarity > 0.1:
                        label = "positive"
                        score = min(0.5 + polarity/2, 1.0)
                    elif polarity < -0.1:
                        label = "negative"
                        score = max(0.5 + polarity/2, 0.0)
                    else:
                        label = "neutral"
                        score = 0.5
                else:
                    # Simple lexicon-based approach
                    sentiment_score = self._simple_lexicon_analysis(message)
                    
                    if sentiment_score > 0.6:
                        label = "positive"
                        score = sentiment_score
                    elif sentiment_score < 0.4:
                        label = "negative"
                        score = sentiment_score
                    else:
                        label = "neutral"
                        score = 0.5
                
                results.append({"label": label, "score": score})
                
            except Exception as e:
                print(f"Fallback analysis failed for message: {e}")
                results.append({"label": "neutral", "score": 0.5})
        
        return results
    
    def _simple_lexicon_analysis(self, message: str) -> float:
        """Simple lexicon-based sentiment analysis."""
        positive_words = {
            'good', 'great', 'awesome', 'amazing', 'excellent', 'wonderful', 'fantastic',
            'love', 'like', 'happy', 'joy', 'excited', 'glad', 'pleased', 'satisfied',
            'perfect', 'best', 'brilliant', 'cool', 'nice', 'beautiful', 'incredible',
            'outstanding', 'superb', 'magnificent', 'terrific', 'marvelous', 'fabulous',
            'yes', 'yeah', 'yay', 'congratulations', 'congrats', 'well done', 'bravo',
            '😊', '😄', '😃', '😀', '🙂', '😍', '🥰', '😘', '👍', '❤️', '💕', '🎉'
        }
        
        negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'disgusting', 'hate', 'dislike',
            'sad', 'angry', 'mad', 'upset', 'disappointed', 'frustrated', 'annoyed',
            'worst', 'stupid', 'dumb', 'crazy', 'ridiculous', 'pathetic', 'useless',
            'wrong', 'problem', 'issue', 'trouble', 'difficult', 'hard', 'impossible',
            'no', 'never', 'nothing', 'nobody', 'nowhere', 'failure', 'failed',
            '😢', '😭', '😞', '😔', '😟', '😕', '🙁', '😠', '😡', '🤬', '💔', '👎'
        }
        
        words = message.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        total_sentiment_words = positive_count + negative_count
        
        if total_sentiment_words == 0:
            return 0.5  # Neutral
        
        positive_ratio = positive_count / total_sentiment_words
        return positive_ratio
    
    def clear_cache(self):
        """Clear the analysis cache."""
        self.analysis_cache.clear()
        print("🧹 Analysis cache cleared")

# Global analyzer instance with GPU settings
_sentiment_analyzer = None

def get_sentiment_pipeline(use_gpu: bool = True, batch_size: int = None):
    """Get cached sentiment analyzer instance with GPU acceleration."""
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        print("🏗️ Creating new sentiment analyzer instance")
        _sentiment_analyzer = SentimentAnalyzer(use_gpu=use_gpu, batch_size=batch_size)
    else:
        print("♻️ Reusing existing sentiment analyzer instance")
    return _sentiment_analyzer

def perform_sentiment_analysis(df: pd.DataFrame, positive_threshold: float = 0.6, 
                               use_gpu: bool = True, progress_callback=None) -> pd.DataFrame:
    """
    Enhanced sentiment analysis with caching and single model loading.
    """
    try:
        if df.empty or "message" not in df.columns:
            return df
        
        # Check if sentiment analysis already exists and data hasn't changed
        if 'sentiment_label' in df.columns and 'sentiment_score' in df.columns:
            print("📋 Sentiment analysis already exists, checking if data changed...")
            
            # Generate hash of current messages to check if data changed
            messages = df["message"].fillna("").astype(str).tolist()
            current_hash = hashlib.md5(''.join(sorted(messages)).encode()).hexdigest()
            
            # Get analyzer to check if we have cached results
            analyzer = get_sentiment_pipeline(use_gpu=use_gpu)
            
            if current_hash in analyzer.analysis_cache or analyzer.current_data_hash == current_hash:
                print("✅ Data unchanged, using existing sentiment analysis")
                return df
            else:
                print("🔄 Data changed, performing new sentiment analysis")
                analyzer.current_data_hash = current_hash
        
        print("🚀 Starting sentiment analysis...")
        
        # Initialize analyzer with GPU settings (will reuse if already exists)
        analyzer = get_sentiment_pipeline(use_gpu=use_gpu)
        
        if analyzer.use_gpu and GPU_AVAILABLE:
            print(f"📊 Using GPU: {GPU_NAME}")
            print(f"🔧 Batch size: {analyzer.batch_size}")
        else:
            print("⚙️ Using CPU for sentiment analysis")
        
        # Prepare messages
        messages = df["message"].fillna("").astype(str).tolist()
        
        if not messages:
            df["sentiment_label"] = "neutral"
            df["sentiment_score"] = 0.5
            return df
        
        # Filter out very short messages and media notifications
        valid_messages = []
        valid_indices = []
        
        for i, msg in enumerate(messages):
            cleaned = analyzer._clean_message(msg)
            if len(cleaned.strip()) >= 3:  # Only analyze messages with at least 3 characters
                valid_messages.append(cleaned)
                valid_indices.append(i)
        
        # Initialize results arrays
        labels = ["neutral"] * len(messages)
        scores = [0.5] * len(messages)
        
        if valid_messages:
            print(f"🔍 Analyzing {len(valid_messages)} valid messages...")
            
            # Adaptive batch sizing based on message count and GPU memory
            total_messages = len(valid_messages)
            
            if analyzer.use_gpu and GPU_AVAILABLE:
                # Estimate memory usage and adjust batch size
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                if total_messages > 10000 and gpu_memory < 8:
                    batch_size = min(analyzer.batch_size, 16)
                elif total_messages > 50000:
                    batch_size = min(analyzer.batch_size, 32)
                else:
                    batch_size = analyzer.batch_size
            else:
                batch_size = min(analyzer.batch_size, 8)  # Conservative for CPU
            
            print(f"🔄 Processing with batch size: {batch_size}")
            
            # Process with progress tracking
            total_batches = (len(valid_messages) + batch_size - 1) // batch_size
            processed_messages = 0
            
            try:
                valid_results = []
                
                for batch_idx in range(0, len(valid_messages), batch_size):
                    batch_messages = valid_messages[batch_idx:batch_idx + batch_size]
                    current_batch = batch_idx // batch_size + 1
                    
                    # Update progress
                    if progress_callback:
                        progress = min(processed_messages / len(valid_messages), 1.0)
                        progress_callback(progress, f"Processing batch {current_batch}/{total_batches}")
                    
                    print(f"⚡ Batch {current_batch}/{total_batches} - {len(batch_messages)} messages")
                    
                    # Analyze batch (uses caching internally)
                    batch_results = analyzer.analyze_batch(batch_messages, custom_batch_size=len(batch_messages))
                    valid_results.extend(batch_results)
                    
                    processed_messages += len(batch_messages)
                    
                    # GPU memory management
                    if analyzer.use_gpu and GPU_AVAILABLE and current_batch % 10 == 0:
                        allocated = torch.cuda.memory_allocated(0) / 1024**3
                        print(f"📊 GPU Memory Usage: {allocated:.2f}GB")
                        
                        if allocated > gpu_memory * 0.8:  # If using >80% memory
                            torch.cuda.empty_cache()
                            print("🧹 Cleared GPU cache")
                
                # Map results back to original positions
                for i, result in enumerate(valid_results):
                    if i < len(valid_indices):
                        original_idx = valid_indices[i]
                        labels[original_idx] = result["label"]
                        scores[original_idx] = result["score"]
                
            except Exception as e:
                print(f"❌ Error during batch processing: {e}")
                if analyzer.use_gpu and GPU_AVAILABLE:
                    torch.cuda.empty_cache()
                    print("🧹 Cleared GPU cache after error")
                raise
        
        # Apply threshold for positive classification
        final_labels = []
        for i, (label, score) in enumerate(zip(labels, scores)):
            if label == "positive" and score < positive_threshold:
                final_labels.append("neutral")
            else:
                final_labels.append(label)
        
        # Add results to dataframe
        df = df.copy()
        df["sentiment_label"] = final_labels
        df["sentiment_score"] = scores
        
        print("✅ Sentiment analysis completed!")
        
        # Print summary with performance stats
        sentiment_counts = pd.Series(final_labels).value_counts()
        print(f"📊 Sentiment distribution: {sentiment_counts.to_dict()}")
        
        if analyzer.use_gpu and GPU_AVAILABLE:
            final_memory = torch.cuda.memory_allocated(0) / 1024**3
            print(f"📊 Final GPU Memory Usage: {final_memory:.2f}GB")
            
            # Clean up GPU memory
            torch.cuda.empty_cache()
            print("🧹 GPU memory cleaned up")
        
        return df
        
    except Exception as e:
        print(f"❌ Error in sentiment analysis: {e}")
        if GPU_AVAILABLE:
            torch.cuda.empty_cache()
        # Return dataframe with neutral sentiment as fallback
        df = df.copy()
        df["sentiment_label"] = "neutral"
        df["sentiment_score"] = 0.5
        return df

def get_gpu_info() -> Dict[str, Any]:
    """Get detailed GPU information for diagnostics."""
    gpu_info = {
        "available": GPU_AVAILABLE,
        "device_count": 0,
        "current_device": None,
        "memory_info": {},
        "device_properties": {}
    }
    
    if GPU_AVAILABLE:
        try:
            gpu_info["device_count"] = torch.cuda.device_count()
            gpu_info["current_device"] = torch.cuda.current_device()
            
            for i in range(gpu_info["device_count"]):
                props = torch.cuda.get_device_properties(i)
                gpu_info["device_properties"][i] = {
                    "name": props.name,
                    "total_memory": props.total_memory / 1024**3,  # GB
                    "major": props.major,
                    "minor": props.minor,
                    "multi_processor_count": props.multi_processor_count
                }
                
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated(i) / 1024**3
                    cached = torch.cuda.memory_reserved(i) / 1024**3
                    gpu_info["memory_info"][i] = {
                        "allocated": allocated,
                        "cached": cached,
                        "free": props.total_memory / 1024**3 - allocated
                    }
        except Exception as e:
            gpu_info["error"] = str(e)
    
    return gpu_info

def optimize_for_gpu(message_count: int) -> Dict[str, int]:
    """Optimize batch sizes and settings based on GPU capabilities and message count."""
    optimization = {
        "batch_size": 16,
        "max_length": 256,
        "use_fp16": False,
        "parallel_batches": 1
    }
    
    if not GPU_AVAILABLE:
        optimization["batch_size"] = 8
        optimization["max_length"] = 128
        return optimization
    
    try:
        gpu_props = torch.cuda.get_device_properties(0)
        total_memory_gb = gpu_props.total_memory / 1024**3
        
        # Optimize based on GPU memory
        if total_memory_gb >= 16:  # High-end GPU
            optimization["batch_size"] = 128 if message_count > 100000 else 64
            optimization["max_length"] = 512
            optimization["use_fp16"] = True
            optimization["parallel_batches"] = 2
        elif total_memory_gb >= 8:  # Mid-range GPU
            optimization["batch_size"] = 64 if message_count > 50000 else 32
            optimization["max_length"] = 384
            optimization["use_fp16"] = True
        elif total_memory_gb >= 4:  # Entry-level GPU
            optimization["batch_size"] = 32 if message_count > 20000 else 16
            optimization["max_length"] = 256
        else:  # Low memory GPU
            optimization["batch_size"] = 16
            optimization["max_length"] = 128
        
        # Adjust for very large datasets
        if message_count > 200000:
            optimization["batch_size"] = min(optimization["batch_size"], 32)
        elif message_count < 1000:
            optimization["batch_size"] = min(optimization["batch_size"], 8)
            
    except Exception as e:
        print(f"Error optimizing for GPU: {e}")
    
    return optimization

# Modified functions that now use cached results instead of re-running analysis
def sentiment_over_time(selected_user: str, df: pd.DataFrame, window_days: int = 30) -> pd.DataFrame:
    """Enhanced sentiment analysis over time using CACHED results."""
    try:
        # Check if sentiment data already exists
        if 'sentiment_score' not in df.columns or 'sentiment_label' not in df.columns:
            print("⚠️ No sentiment data found. Run sentiment analysis first.")
            return pd.DataFrame(columns=["date", "sentiment_score", "user"])
        
        print("📊 Using existing sentiment data for time analysis")
        
        df_use = df.copy() if selected_user == "Overall" else df[df["user"] == selected_user].copy()
        
        if df_use.empty or 'date' not in df_use.columns:
            return pd.DataFrame(columns=["date", "sentiment_score", "user"])
        
        # Ensure date column is datetime
        df_use["date"] = pd.to_datetime(df_use["date"], errors="coerce")
        df_use = df_use.dropna(subset=["date"])
        
        if df_use.empty:
            return pd.DataFrame(columns=["date", "sentiment_score", "user"])
        
        # Filter to recent days
        end_date = df_use["date"].max()
        start_date = end_date - pd.Timedelta(days=window_days)
        
        mask = (df_use["date"] >= start_date) & (df_use["date"] <= end_date)
        filtered_df = df_use.loc[mask].copy()
        
        if filtered_df.empty:
            return pd.DataFrame(columns=["date", "sentiment_score", "user"])
        
        # Group by date and calculate sentiment using existing scores
        sentiment_data = []
        
        if selected_user == "Overall":
            # Group by date and user
            for (date, user), group in filtered_df.groupby([filtered_df["date"].dt.date, "user"]):
                # Use existing sentiment scores
                avg_score = group["sentiment_score"].mean()
                sentiment_data.append({
                    "date": pd.to_datetime(date),
                    "sentiment_score": avg_score,
                    "user": user
                })
        else:
            # Group by date only
            for date, group in filtered_df.groupby(filtered_df["date"].dt.date):
                # Use existing sentiment scores
                avg_score = group["sentiment_score"].mean()
                sentiment_data.append({
                    "date": pd.to_datetime(date),
                    "sentiment_score": avg_score,
                    "user": selected_user
                })
        
        if not sentiment_data:
            return pd.DataFrame(columns=["date", "sentiment_score", "user"])
        
        return pd.DataFrame(sentiment_data)
    
    except Exception as e:
        print(f"Error in sentiment_over_time: {e}")
        return pd.DataFrame(columns=["date", "sentiment_score", "user"])

def analyze_sentiment_trends(df: pd.DataFrame, user: str = None, days: int = 30) -> Dict[str, Any]:
    """Analyze sentiment trends over time using CACHED results."""
    try:
        if df.empty or 'sentiment_score' not in df.columns:
            print("⚠️ No sentiment data available. Run sentiment analysis first.")
            return {}
        
        print("📊 Using existing sentiment data for trend analysis")
        
        # Filter data
        if user and user != "Overall":
            df_filtered = df[df['user'] == user].copy()
        else:
            df_filtered = df.copy()
        
        if df_filtered.empty:
            return {}
        
        # Ensure date column exists and is datetime
        if 'date' not in df_filtered.columns:
            return {}
        
        df_filtered['date'] = pd.to_datetime(df_filtered['date'], errors='coerce')
        df_filtered = df_filtered.dropna(subset=['date'])
        
        # Filter to recent days
        end_date = df_filtered['date'].max()
        start_date = end_date - pd.Timedelta(days=days)
        df_filtered = df_filtered[df_filtered['date'] >= start_date]
        
        if df_filtered.empty:
            return {}
        
        # Calculate trends using existing sentiment scores
        daily_sentiment = (df_filtered.groupby(df_filtered['date'].dt.date)['sentiment_score']
                          .mean().reset_index())
        
        # Calculate overall statistics
        positive_pct = (df_filtered['sentiment_label'] == 'positive').mean() * 100
        negative_pct = (df_filtered['sentiment_label'] == 'negative').mean() * 100
        neutral_pct = (df_filtered['sentiment_label'] == 'neutral').mean() * 100
        
        avg_sentiment = df_filtered['sentiment_score'].mean()
        
        # Find most positive and negative days
        if not daily_sentiment.empty:
            most_positive_day = daily_sentiment.loc[daily_sentiment['sentiment_score'].idxmax(), 'date']
            most_negative_day = daily_sentiment.loc[daily_sentiment['sentiment_score'].idxmin(), 'date']
        else:
            most_positive_day = None
            most_negative_day = None
        
        return {
            'daily_sentiment': daily_sentiment,
            'positive_percentage': positive_pct,
            'negative_percentage': negative_pct,
            'neutral_percentage': neutral_pct,
            'average_sentiment': avg_sentiment,
            'most_positive_day': most_positive_day,
            'most_negative_day': most_negative_day,
            'total_messages_analyzed': len(df_filtered)
        }
        
    except Exception as e:
        print(f"Error in sentiment trends analysis: {e}")
        return {}

def get_sentiment_summary(df: pd.DataFrame) -> str:
    """Generate a text summary of sentiment analysis results using CACHED data."""
    try:
        if df.empty or 'sentiment_label' not in df.columns:
            return "No sentiment data available. Run sentiment analysis first."
        
        total_messages = len(df)
        sentiment_counts = df['sentiment_label'].value_counts()
        
        positive_count = sentiment_counts.get('positive', 0)
        negative_count = sentiment_counts.get('negative', 0)
        neutral_count = sentiment_counts.get('neutral', 0)
        
        positive_pct = (positive_count / total_messages) * 100
        negative_pct = (negative_count / total_messages) * 100
        neutral_pct = (neutral_count / total_messages) * 100
        
        avg_sentiment = df['sentiment_score'].mean()
        
        summary = f"""
Sentiment Analysis Summary:
- Total messages analyzed: {total_messages:,}
- Positive messages: {positive_count:,} ({positive_pct:.1f}%)
- Negative messages: {negative_count:,} ({negative_pct:.1f}%)
- Neutral messages: {neutral_count:,} ({neutral_pct:.1f}%)
- Average sentiment score: {avg_sentiment:.3f}

Overall sentiment: {"Positive" if avg_sentiment > 0.6 else "Negative" if avg_sentiment < 0.4 else "Neutral"}
"""
        
        return summary
        
    except Exception as e:
        return f"Error generating sentiment summary: {e}"

def clear_sentiment_cache():
    """Clear the sentiment analysis cache - useful when data changes."""
    global _sentiment_analyzer
    if _sentiment_analyzer is not None:
        _sentiment_analyzer.clear_cache()
        print("🧹 Sentiment analysis cache cleared")
=======
# # sentiment_analysis.py
# from transformers import pipeline
# import pandas as pd
#
#
# def perform_sentiment_analysis(df, positive_threshold=0.5, negative_threshold=0.5):
# #Using the 'distilbert-base-uncased-finetuned-sst-2-english' model for sentiment analysis
#     classifier = pipeline('sentiment-analysis')
#
#     # Apply sentiment analysis to each message
#     results = classifier(df['message'].tolist())
#
#     # Extract sentiment labels and scores
#     sentiments = [result['label'] for result in results]
#     scores = [result['score'] for result in results]
#
#     # Add sentiment-related columns to the DataFrame
#     df['sentiment_label'] = sentiments
#     df['sentiment_score'] = scores
#
#     return df

from transformers import DistilBertTokenizer, DistilBertForSequenceClassification


def perform_sentiment_analysis(df, positive_threshold=0.9):
    # Use the 'distilbert-base-uncased-finetuned-sst-2-english' model for sentiment analysis
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"

    # Load the model and tokenizer
    model = DistilBertForSequenceClassification.from_pretrained(model_name)
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)

    # Define the sentiment analysis function
    def analyze_sentiment(message):
        inputs = tokenizer(message, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = logits.softmax(dim=1)
        return probabilities

    # Apply sentiment analysis to each message
    results = df['message'].apply(analyze_sentiment)

    # Initialize empty lists for positive and negative probabilities
    positive_probs = []
    negative_probs = []

    for result in results:
        # Extract positive and negative probabilities
        positive_prob = result[0][1].item()
        negative_prob = result[0][0].item()

        # Append probabilities to the respective lists
        positive_probs.append(positive_prob)
        negative_probs.append(negative_prob)

    # Add sentiment-related columns to the DataFrame
    df['sentiment_label'] = ["positive" if p >= positive_threshold else "negative" for p in positive_probs]
    df['sentiment_score'] = [max(p, n) for p, n in zip(positive_probs, negative_probs)]

    return df
>>>>>>> fcfbd584046b32005c908f931ae5d9ff4a42871a
