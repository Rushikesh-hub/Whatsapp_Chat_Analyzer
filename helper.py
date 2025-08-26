from datetime import timedelta
from collections import Counter
import pandas as pd
from urlextract import URLExtract
from wordcloud import WordCloud, STOPWORDS
import emoji
import re
import numpy as np

# Initialize URL extractor
extract = URLExtract()

# Enhanced stopwords list
CUSTOM_STOPWORDS = {
    'english': set(STOPWORDS),
    'hinglish': {
        'hai', 'hain', 'ka', 'ki', 'ke', 'ko', 'se', 'me', 'mein', 'main', 'tu', 'tum', 'aap', 
        'ye', 'yeh', 'vo', 'voh', 'kya', 'kyun', 'kaise', 'kahan', 'kab', 'jo', 'ji', 'nhi', 
        'nahi', 'han', 'haan', 'ok', 'okay', 'hmm', 'hm', 'lol', 'lmao', 'omg', 'wtf', 'tbh',
        'btw', 'imo', 'imho', 'fyi', 'asap', 'dm', 'pm', 'am', 'the', 'and', 'or', 'but',
        'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into',
        'through', 'during', 'before', 'after', 'above', 'below', 'between', 'among',
        'media', 'omitted', 'message', 'deleted', 'this', 'was', 'image', 'video', 'audio'
    }
}

def _is_emoji(char: str) -> bool:
    """Check if a character is an emoji."""
    try:
        return char in emoji.EMOJI_DATA
    except (AttributeError, Exception):
        try:
            return bool(emoji.emoji_count(char))
        except:
            # Fallback using Unicode ranges for emojis
            return any(ord(char) in range(start, end + 1) for start, end in [
                (0x1F600, 0x1F64F),  # Emoticons
                (0x1F300, 0x1F5FF),  # Misc Symbols and Pictographs
                (0x1F680, 0x1F6FF),  # Transport and Map
                (0x1F1E0, 0x1F1FF),  # Regional Indicators
                (0x2600, 0x26FF),    # Misc symbols
                (0x2700, 0x27BF),    # Dingbats
            ])

def _clean_message(message: str) -> str:
    """Clean and normalize message text."""
    if pd.isna(message):
        return ""
    
    message = str(message).lower()
    
    # Remove media placeholders
    message = re.sub(r'<media omitted>', '', message, flags=re.IGNORECASE)
    message = re.sub(r'image omitted', '', message, flags=re.IGNORECASE)
    message = re.sub(r'video omitted', '', message, flags=re.IGNORECASE)
    message = re.sub(r'audio omitted', '', message, flags=re.IGNORECASE)
    message = re.sub(r'document omitted', '', message, flags=re.IGNORECASE)
    message = re.sub(r'this message was deleted', '', message, flags=re.IGNORECASE)
    
    # Remove URLs
    message = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', message)
    
    # Remove extra whitespace and special characters
    message = re.sub(r'[^\w\s]', ' ', message)
    message = re.sub(r'\s+', ' ', message).strip()
    
    return message

def fetch_stats(selected_user: str, df: pd.DataFrame) -> tuple:
    """Fetch comprehensive statistics for selected user or overall."""
    try:
        df_use = df.copy() if selected_user == "Overall" else df[df["user"] == selected_user].copy()
        
        if df_use.empty:
            return 0, 0, 0, 0
        
        num_messages = len(df_use)
        
        # Count words more accurately
        words = []
        for message in df_use["message"].fillna(''):
            cleaned_msg = _clean_message(str(message))
            if cleaned_msg:
                words.extend(cleaned_msg.split())
        
        # Count media messages
        media_patterns = [
            r'<media omitted>',
            r'image omitted',
            r'video omitted',
            r'audio omitted',
            r'document omitted'
        ]
        
        num_media_messages = 0
        for pattern in media_patterns:
            num_media_messages += df_use["message"].astype(str).str.contains(pattern, case=False, na=False).sum()
        
        # Count links
        links = []
        for message in df_use["message"].fillna(''):
            try:
                links.extend(extract.find_urls(str(message)))
            except:
                continue
        
        return num_messages, len(words), num_media_messages, len(links)
    
    except Exception as e:
        print(f"Error in fetch_stats: {e}")
        return 0, 0, 0, 0

def most_busy_users(df: pd.DataFrame) -> tuple:
    """Get most active users in the chat."""
    try:
        # Filter out system notifications
        df_filtered = df[~df["user"].isin(["group_notification", "Group notification", "System"])]
        
        if df_filtered.empty:
            return pd.Series(dtype='int64'), pd.DataFrame()
        
        user_counts = df_filtered["user"].value_counts().head(10)
        
        df_pct = pd.DataFrame({
            'name': user_counts.index,
            'message_count': user_counts.values,
            'percent': round((user_counts.values / len(df_filtered)) * 100, 2)
        })
        
        return user_counts, df_pct
    
    except Exception as e:
        print(f"Error in most_busy_users: {e}")
        return pd.Series(dtype='int64'), pd.DataFrame()

def create_wordcloud(selected_user: str, df: pd.DataFrame) -> object:
    """Generate an enhanced word cloud."""
    try:
        df_use = df.copy() if selected_user == "Overall" else df[df["user"] == selected_user].copy()
        
        # Filter out system messages and media
        df_use = df_use[~df_use["user"].isin(["group_notification", "Group notification", "System"])]
        df_use = df_use[~df_use["message"].astype(str).str.contains(
            r'<media omitted>|image omitted|video omitted|audio omitted|document omitted|this message was deleted',
            case=False, na=False
        )]
        
        if df_use.empty:
            return None
        
        # Combine all custom stopwords
        all_stops = CUSTOM_STOPWORDS['english'] | CUSTOM_STOPWORDS['hinglish']
        
        # Load additional stopwords from file if available
        try:
            with open("stop_hinglish.txt", "r", encoding="utf-8") as f:
                file_stops = set(word.strip().lower() for word in f.read().split() if word.strip())
                all_stops.update(file_stops)
        except FileNotFoundError:
            pass
        
        def remove_stop_words(message: str) -> str:
            """Remove stopwords and clean text."""
            cleaned = _clean_message(message)
            tokens = [word for word in cleaned.split() 
                     if len(word) > 2 and word not in all_stops]
            return " ".join(tokens)
        
        # Process all messages
        processed_messages = df_use["message"].astype(str).apply(remove_stop_words)
        text_content = " ".join(processed_messages).strip()
        
        if not text_content or len(text_content.split()) < 5:
            return None
        
        # Create enhanced wordcloud
        wc = WordCloud(
            width=800, 
            height=400,
            min_font_size=10,
            max_font_size=100,
            background_color="white",
            colormap="viridis",
            max_words=100,
            relative_scaling=0.5,
            collocations=False
        )
        
        df_wc = wc.generate(text_content)
        return df_wc
    
    except Exception as e:
        print(f"Error in create_wordcloud: {e}")
        return None

def most_common_words(selected_user: str, df: pd.DataFrame) -> pd.DataFrame:
    """Get most frequently used words."""
    try:
        df_use = df.copy() if selected_user == "Overall" else df[df["user"] == selected_user].copy()
        
        # Filter out system messages
        df_use = df_use[~df_use["user"].isin(["group_notification", "Group notification", "System"])]
        
        if df_use.empty:
            return pd.DataFrame(columns=['word', 'frequency'])
        
        # Combine all stopwords
        all_stops = CUSTOM_STOPWORDS['english'] | CUSTOM_STOPWORDS['hinglish']
        
        try:
            with open("stop_hinglish.txt", "r", encoding="utf-8") as f:
                file_stops = set(word.strip().lower() for word in f.read().split() if word.strip())
                all_stops.update(file_stops)
        except FileNotFoundError:
            pass
        
        # Extract words
        words = []
        for message in df_use["message"].fillna(''):
            cleaned = _clean_message(str(message))
            for word in cleaned.split():
                if len(word) > 2 and word not in all_stops:
                    words.append(word)
        
        if not words:
            return pd.DataFrame(columns=['word', 'frequency'])
        
        # Get most common words
        word_freq = Counter(words).most_common(25)
        most_common_df = pd.DataFrame(word_freq, columns=['word', 'frequency'])
        
        return most_common_df
    
    except Exception as e:
        print(f"Error in most_common_words: {e}")
        return pd.DataFrame(columns=['word', 'frequency'])

def emoji_helper(selected_user: str, df: pd.DataFrame) -> pd.DataFrame:
    """Enhanced emoji analysis."""
    try:
        df_use = df.copy() if selected_user == "Overall" else df[df["user"] == selected_user].copy()
        
        if df_use.empty:
            return pd.DataFrame(columns=['emoji', 'count'])
        
        emojis = []
        for message in df_use["message"].fillna(''):
            message_str = str(message)
            for char in message_str:
                if _is_emoji(char):
                    emojis.append(char)
        
        if not emojis:
            return pd.DataFrame(columns=['emoji', 'count'])
        
        emoji_counts = Counter(emojis).most_common()
        emoji_df = pd.DataFrame(emoji_counts, columns=['emoji', 'count'])
        
        return emoji_df
    
    except Exception as e:
        print(f"Error in emoji_helper: {e}")
        return pd.DataFrame(columns=['emoji', 'count'])

def monthly_timeline(selected_user: str, df: pd.DataFrame) -> pd.DataFrame:
    """Generate monthly timeline data."""
    try:
        df_use = df.copy() if selected_user == "Overall" else df[df["user"] == selected_user].copy()
        
        if df_use.empty or 'Year' not in df_use.columns or 'month_num' not in df_use.columns:
            return pd.DataFrame(columns=["Year", "month_num", "month", "message", "time"])
        
        # Ensure numeric columns
        df_use['Year'] = pd.to_numeric(df_use['Year'], errors='coerce')
        df_use['month_num'] = pd.to_numeric(df_use['month_num'], errors='coerce')
        
        # Remove rows with invalid dates
        df_use = df_use.dropna(subset=['Year', 'month_num', 'month'])
        
        if df_use.empty:
            return pd.DataFrame(columns=["Year", "month_num", "month", "message", "time"])
        
        timeline = (df_use.groupby(["Year", "month_num", "month"], dropna=True)
                   .size()
                   .reset_index(name='message'))
        
        timeline["time"] = timeline["month"].astype(str) + "-" + timeline["Year"].astype(int).astype(str)
        
        return timeline
    
    except Exception as e:
        print(f"Error in monthly_timeline: {e}")
        return pd.DataFrame(columns=["Year", "month_num", "month", "message", "time"])

def daily_timeline(selected_user: str, df: pd.DataFrame) -> pd.DataFrame:
    """Generate daily timeline data."""
    try:
        df_use = df.copy() if selected_user == "Overall" else df[df["user"] == selected_user].copy()
        
        if df_use.empty or 'only_date' not in df_use.columns:
            return pd.DataFrame(columns=["only_date", "message"])
        
        daily_timeline = (df_use.groupby("only_date", dropna=True)
                         .size()
                         .reset_index(name='message'))
        
        return daily_timeline
    
    except Exception as e:
        print(f"Error in daily_timeline: {e}")
        return pd.DataFrame(columns=["only_date", "message"])

def week_activity_map(selected_user: str, df: pd.DataFrame) -> pd.Series:
    """Get activity by day of week."""
    try:
        df_use = df.copy() if selected_user == "Overall" else df[df["user"] == selected_user].copy()
        
        if df_use.empty or 'day_name' not in df_use.columns:
            return pd.Series(dtype="int64")
        
        # Ensure proper day order
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_counts = df_use["day_name"].value_counts()
        
        # Reindex to maintain day order
        day_counts = day_counts.reindex(day_order, fill_value=0)
        
        return day_counts
    
    except Exception as e:
        print(f"Error in week_activity_map: {e}")
        return pd.Series(dtype="int64")

def month_activity_map(selected_user: str, df: pd.DataFrame) -> pd.Series:
    """Get activity by month."""
    try:
        df_use = df.copy() if selected_user == "Overall" else df[df["user"] == selected_user].copy()
        
        if df_use.empty or 'month' not in df_use.columns:
            return pd.Series(dtype="int64")
        
        # Ensure proper month order
        month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                      'July', 'August', 'September', 'October', 'November', 'December']
        
        month_counts = df_use["month"].value_counts()
        month_counts = month_counts.reindex(month_order, fill_value=0)
        
        return month_counts
    
    except Exception as e:
        print(f"Error in month_activity_map: {e}")
        return pd.Series(dtype="int64")

def hourly_activity_map(selected_user: str, df: pd.DataFrame) -> pd.Series:
    """Get activity by hour of day."""
    try:
        df_use = df.copy() if selected_user == "Overall" else df[df["user"] == selected_user].copy()
        
        if df_use.empty or 'hours' not in df_use.columns:
            return pd.Series(dtype="int64")
        
        hour_counts = df_use["hours"].value_counts().sort_index()
        
        # Ensure all hours are represented
        all_hours = pd.Series(index=range(24), dtype='int64')
        hour_counts = hour_counts.reindex(all_hours.index, fill_value=0)
        
        return hour_counts
    
    except Exception as e:
        print(f"Error in hourly_activity_map: {e}")
        return pd.Series(dtype="int64")

def activity_heatmap(selected_user: str, df: pd.DataFrame) -> pd.DataFrame:
    """Generate activity heatmap data."""
    try:
        df_use = df.copy() if selected_user == "Overall" else df[df["user"] == selected_user].copy()
        
        if df_use.empty or 'day_name' not in df_use.columns or 'period' not in df_use.columns:
            return pd.DataFrame()
        
        # Create pivot table for heatmap
        user_heatmap = df_use.pivot_table(
            index="day_name", 
            columns="period", 
            values="message", 
            aggfunc="count",
            fill_value=0
        )
        
        # Ensure proper day order
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        user_heatmap = user_heatmap.reindex(day_order, fill_value=0)
        
        return user_heatmap
    
    except Exception as e:
        print(f"Error in activity_heatmap: {e}")
        return pd.DataFrame()

def sentiment_over_time(selected_user: str, df: pd.DataFrame, window_days: int = 30) -> pd.DataFrame:
    """Enhanced sentiment analysis over time using CACHED results - NO RE-ANALYSIS."""
    try:
        # Check if sentiment data already exists
        if 'sentiment_score' not in df.columns or 'sentiment_label' not in df.columns:
            print("⚠️ No sentiment data found. Run sentiment analysis first.")
            return pd.DataFrame(columns=["date", "sentiment_score", "user"])
        
        print("📊 Using existing sentiment data for time analysis (no re-analysis)")
        
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
        start_date = end_date - timedelta(days=window_days)
        
        mask = (df_use["date"] >= start_date) & (df_use["date"] <= end_date)
        filtered_df = df_use.loc[mask].copy()
        
        if filtered_df.empty:
            return pd.DataFrame(columns=["date", "sentiment_score", "user"])
        
        # Group by date and calculate sentiment using EXISTING scores (no re-analysis)
        sentiment_data = []
        
        if selected_user == "Overall":
            # Group by date and user
            for (date, user), group in filtered_df.groupby([filtered_df["date"].dt.date, "user"]):
                # Use existing sentiment scores - NO RE-ANALYSIS
                avg_score = group["sentiment_score"].mean()
                sentiment_data.append({
                    "date": pd.to_datetime(date),
                    "sentiment_score": avg_score,
                    "user": user
                })
        else:
            # Group by date only
            for date, group in filtered_df.groupby(filtered_df["date"].dt.date):
                # Use existing sentiment scores - NO RE-ANALYSIS
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

def generate_summary_report(selected_user: str, df: pd.DataFrame, num_messages: int, 
                          words: int, num_media_messages: int, num_links: int) -> str:
    """Generate a comprehensive summary report."""
    try:
        report = f"""
WhatsApp Chat Analysis Report
=============================

Analysis Subject: {selected_user}
Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

OVERVIEW STATISTICS
==================
Total Messages: {num_messages:,}
Total Words: {words:,}
Media Messages: {num_media_messages:,}
Links Shared: {num_links:,}
Average Words per Message: {words/num_messages:.1f if num_messages > 0 else 0}

"""
        
        if selected_user == "Overall":
            # Add user statistics
            user_counts, user_pct = most_busy_users(df)
            if not user_counts.empty:
                report += """
MOST ACTIVE USERS
================
"""
                for i, (user, count) in enumerate(user_counts.head(5).items(), 1):
                    pct = user_pct[user_pct['name'] == user]['percent'].iloc[0] if len(user_pct) > 0 else 0
                    report += f"{i}. {user}: {count:,} messages ({pct:.1f}%)\n"
        
        # Add time-based insights
        df_use = df.copy() if selected_user == "Overall" else df[df["user"] == selected_user].copy()
        
        if not df_use.empty:
            # Most active day
            if 'day_name' in df_use.columns:
                most_active_day = df_use['day_name'].mode().iloc[0] if not df_use['day_name'].empty else "N/A"
                report += f"""
ACTIVITY PATTERNS
================
Most Active Day: {most_active_day}
"""
            
            # Most active hour
            if 'hours' in df_use.columns:
                most_active_hour = df_use['hours'].mode().iloc[0] if not df_use['hours'].empty else "N/A"
                report += f"Most Active Hour: {most_active_hour}:00\n"
        
        # Add word analysis
        common_words = most_common_words(selected_user, df)
        if not common_words.empty:
            report += """
MOST COMMON WORDS
================
"""
            # Handle both column formats
            if 'word' in common_words.columns and 'frequency' in common_words.columns:
                for i, (word, freq) in enumerate(common_words.head(10)[['word', 'frequency']].values, 1):
                    report += f"{i}. {word}: {freq} times\n"
            else:
                for i, (word, freq) in enumerate(common_words.head(10).values, 1):
                    report += f"{i}. {word}: {freq} times\n"
        
        # Add emoji analysis
        emojis = emoji_helper(selected_user, df)
        if not emojis.empty:
            report += """
TOP EMOJIS
==========
"""
            # Handle both column formats
            if 'emoji' in emojis.columns and 'count' in emojis.columns:
                for i, (emoji_char, count) in enumerate(emojis.head(5)[['emoji', 'count']].values, 1):
                    report += f"{i}. {emoji_char}: {count} times\n"
            else:
                for i, (emoji_char, count) in enumerate(emojis.head(5).values, 1):
                    report += f"{i}. {emoji_char}: {count} times\n"
        
        # Add sentiment analysis summary if available
        if 'sentiment_label' in df.columns and 'sentiment_score' in df.columns:
            df_sentiment = df_use.copy()
            sentiment_counts = df_sentiment['sentiment_label'].value_counts()
            
            positive_count = sentiment_counts.get('positive', 0)
            negative_count = sentiment_counts.get('negative', 0)
            neutral_count = sentiment_counts.get('neutral', 0)
            
            total_sentiment_messages = len(df_sentiment)
            positive_pct = (positive_count / total_sentiment_messages) * 100 if total_sentiment_messages > 0 else 0
            negative_pct = (negative_count / total_sentiment_messages) * 100 if total_sentiment_messages > 0 else 0
            neutral_pct = (neutral_count / total_sentiment_messages) * 100 if total_sentiment_messages > 0 else 0
            
            avg_sentiment = df_sentiment['sentiment_score'].mean() if total_sentiment_messages > 0 else 0.5
            
            report += f"""
SENTIMENT ANALYSIS
==================
Total Messages Analyzed: {total_sentiment_messages:,}
Positive Messages: {positive_count:,} ({positive_pct:.1f}%)
Negative Messages: {negative_count:,} ({negative_pct:.1f}%)
Neutral Messages: {neutral_count:,} ({neutral_pct:.1f}%)
Average Sentiment Score: {avg_sentiment:.3f}
Overall Sentiment: {"Positive" if avg_sentiment > 0.6 else "Negative" if avg_sentiment < 0.4 else "Neutral"}
"""
        
        report += """
==========================================
Report generated by WhatsApp Chat Analyzer
"""
        
        return report
    
    except Exception as e:
        return f"Error generating report: {e}"
=======

from transformers import pipeline
from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import emoji

extract = URLExtract()


def fetch_stats(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    num_messages = df.shape[0]
    words = []
    for message in df['message']:
        words.extend(message.split())

    # fetch number of media messages
    num_media_messages = df[df['message'] == '<Media omitted>\n'].shape[0]

    # fetch number of links shared
    links = []
    for message in df['message']:
        links.extend(extract.find_urls(message))

    return num_messages, len(words), num_media_messages, len(links)


def most_busy_users(df):
    x = df['user'].value_counts().head()
    df = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'name', 'user': 'percent'})
    return x, df


def create_wordcloud(selected_user, df):
    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    df['message'] = df['message'].astype(str)

    def remove_stop_words(message):
        y = []
        for word in message.lower().split():
            if word not in stop_words:
                y.append(word)
        return " ".join(y)

    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
    temp['message'] = temp['message'].apply(remove_stop_words)
    df_wc = wc.generate(temp['message'].str.cat(sep=" "))
    return df_wc


def most_common_words(selected_user, df):
    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    df = temp

    words = []

    for message in df['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    return most_common_df


def emoji_helper(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    emojis = []
    for message in df['message']:
        emojis.extend([c for c in message if emoji.is_emoji(c)])

    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))

    return emoji_df


def monthly_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['Year', 'month_num', 'month']).count()['message'].reset_index()

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['Year'][i]))

    timeline['time'] = time

    return timeline


def daily_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    daily_timeline = df.groupby('only_date').count()['message'].reset_index()

    return daily_timeline


def week_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['day_name'].value_counts()


def month_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['month'].value_counts()


def activity_heatmap(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    user_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)

    return user_heatmap


# setting window day to 10
def sentiment_over_time(selected_user, df, window_days):
    classifier = pipeline('sentiment-analysis')

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    df['date'] = pd.to_datetime(df['date'])
    end_date = df['date'].max()
    start_date = end_date - timedelta(days=window_days)

    filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

    # Print the values of start_date and end_date
    print("Start Date:", start_date)
    print("End Date:", end_date)

    sentiment_scores = []
    dates = []

    for date, group_df in filtered_df.groupby('only_date'):
        results = classifier(group_df['message'].tolist())
        average_score = sum(result['score'] for result in results) / len(results)
        dates.append(date)
        sentiment_scores.append(average_score)

    sentiment_over_time_df = pd.DataFrame({
        'date': dates,
        'sentiment_score': sentiment_scores,
        'user': selected_user  # Assuming you want to store the selected user in the DataFrame
    })

    return sentiment_over_time_df
