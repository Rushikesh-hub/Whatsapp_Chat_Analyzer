import re
import pandas as pd
<<<<<<< HEAD
from datetime import datetime
import numpy as np

def preprocess(data: str) -> pd.DataFrame:
    """
    Enhanced WhatsApp text export parser with improved error handling and flexibility.
    Supports multiple date formats, 24h and 12h times, and various regional formats.
    """
    try:
        if not data or not isinstance(data, str):
            return _empty_dataframe()
        
        # Clean the input data
        data = data.strip()
        if not data:
            return _empty_dataframe()
        
        # Enhanced timestamp patterns for different WhatsApp export formats
        patterns = [
            # Standard formats: dd/mm/yyyy, hh:mm - 
            r'\d{1,2}/\d{1,2}/\d{4},\s\d{1,2}:\d{2}\s-\s',
            # With AM/PM: dd/mm/yyyy, hh:mm AM/PM - 
            r'\d{1,2}/\d{1,2}/\d{4},\s\d{1,2}:\d{2}\s[APMapm]{2}\s-\s',
            # Short year: dd/mm/yy, hh:mm - 
            r'\d{1,2}/\d{1,2}/\d{2},\s\d{1,2}:\d{2}\s-\s',
            # Short year with AM/PM: dd/mm/yy, hh:mm AM/PM - 
            r'\d{1,2}/\d{1,2}/\d{2},\s\d{1,2}:\d{2}\s[APMapm]{2}\s-\s',
            # Alternative formats with different separators
            r'\d{1,2}-\d{1,2}-\d{4},\s\d{1,2}:\d{2}\s-\s',
            r'\d{1,2}\.\d{1,2}\.\d{4},\s\d{1,2}:\d{2}\s-\s',
            # Formats without comma
            r'\d{1,2}/\d{1,2}/\d{4}\s\d{1,2}:\d{2}\s-\s',
            r'\d{1,2}/\d{1,2}/\d{2}\s\d{1,2}:\d{2}\s-\s'
        ]
        
        messages = []
        dates = []
        
        # Try each pattern until we find one that works
        for pattern in patterns:
            try:
                temp_messages = re.split(pattern, data)
                temp_dates = re.findall(pattern, data)
                
                # Check if this pattern found reasonable results
                if len(temp_dates) > 0 and len(temp_messages) > len(temp_dates):
                    messages = temp_messages[1:]  # Skip first empty element
                    dates = temp_dates
                    break
            except Exception as e:
                print(f"Pattern {pattern} failed: {e}")
                continue
        
        # If no pattern worked, try a more flexible approach
        if not dates or not messages:
            return _fallback_parsing(data)
        
        # Ensure we have matching messages and dates
        min_length = min(len(messages), len(dates))
        messages = messages[:min_length]
        dates = dates[:min_length]
        
        if not messages or not dates:
            return _empty_dataframe()
        
        # Create initial dataframe
        df = pd.DataFrame({
            "user_message": messages,
            "message_date": dates
        })
        
        # Clean the date strings
        df["message_date"] = df["message_date"].str.replace(r"\s-\s*$", "", regex=True)
        df["message_date"] = df["message_date"].str.strip()
        
        # Parse dates with multiple format attempts
        df["date"] = df["message_date"].apply(_parse_flexible_date)
        
        # Remove rows where date parsing failed
        df = df.dropna(subset=["date"]).copy()
        
        if df.empty:
            return _empty_dataframe()
        
        # Extract users and messages
        users = []
        msgs = []
        
        for user_message in df["user_message"]:
            try:
                # More flexible user extraction
                user_message = str(user_message).strip()
                
                # Try different patterns for user:message separation
                patterns = [
                    r"^(.*?):\s(.*)$",  # Standard: "User: message"
                    r"^(.*?)-\s(.*)$",  # Alternative: "User- message"
                    r"^([^:]+):\s*(.*)$"  # Flexible colon separation
                ]
                
                parsed = False
                for pattern in patterns:
                    match = re.match(pattern, user_message, re.DOTALL)
                    if match:
                        user = match.group(1).strip()
                        message = match.group(2).strip()
                        
                        # Validate user name (shouldn't be too long or contain special patterns)
                        if len(user) <= 50 and not re.search(r'[\n\r]', user):
                            users.append(user)
                            msgs.append(message)
                            parsed = True
                            break
                
                if not parsed:
                    # Treat as system message
                    users.append("group_notification")
                    msgs.append(user_message)
                    
            except Exception as e:
                print(f"Error parsing user message: {e}")
                users.append("group_notification")
                msgs.append(str(user_message))
        
        df["user"] = users
        df["message"] = msgs
        df = df.drop(columns=["user_message", "message_date"])
        
        # Generate date-related columns with error handling
        df = _add_date_columns(df)
        
        # Final cleanup
        df = df.dropna(subset=["date"]).copy()
        
        # Remove obviously invalid entries
        df = df[df["message"].str.len() < 10000]  # Remove extremely long messages
        df = df[df["user"].str.len() < 100]      # Remove extremely long usernames
        
        return df
        
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return _empty_dataframe()

def _empty_dataframe() -> pd.DataFrame:
    """Return an empty dataframe with proper columns."""
    return pd.DataFrame(columns=[
        "date", "user", "message", "only_date", "Year", "month_num", "month",
        "day", "day_name", "hours", "minute", "period"
    ])

def _parse_flexible_date(date_str: str) -> pd.Timestamp:
    """Parse date string with multiple format attempts."""
    if pd.isna(date_str) or not date_str:
        return pd.NaT
    
    date_str = str(date_str).strip()
    
    # Common WhatsApp date formats
    formats = [
        "%d/%m/%Y, %H:%M",
        "%d/%m/%y, %H:%M", 
        "%m/%d/%Y, %H:%M",
        "%m/%d/%y, %H:%M",
        "%d/%m/%Y, %I:%M %p",
        "%d/%m/%y, %I:%M %p",
        "%m/%d/%Y, %I:%M %p", 
        "%m/%d/%y, %I:%M %p",
        "%d-%m-%Y, %H:%M",
        "%d-%m-%y, %H:%M",
        "%d.%m.%Y, %H:%M",
        "%d.%m.%y, %H:%M",
        "%d/%m/%Y %H:%M",
        "%d/%m/%y %H:%M",
        "%Y-%m-%d, %H:%M",
        "%Y/%m/%d, %H:%M"
    ]
    
    # Try each format
    for fmt in formats:
        try:
            parsed = pd.to_datetime(date_str, format=fmt)
            # Validate the result
            if parsed.year > 1900 and parsed.year <= datetime.now().year + 1:
                return parsed
        except (ValueError, TypeError):
            continue
    
    # Fallback: let pandas infer the format
    try:
        parsed = pd.to_datetime(date_str, dayfirst=True, errors='coerce')
        if pd.notna(parsed) and parsed.year > 1900:
            return parsed
    except:
        pass
    
    return pd.NaT

def _add_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add date-related columns with error handling."""
    try:
        # Basic date components
        df["only_date"] = df["date"].dt.date
        df["Year"] = df["date"].dt.year
        df["month_num"] = df["date"].dt.month
        df["month"] = df["date"].dt.month_name()
        df["day"] = df["date"].dt.day
        df["day_name"] = df["date"].dt.day_name()
        df["hours"] = df["date"].dt.hour
        df["minute"] = df["date"].dt.minute
        
        # Generate hour periods for heatmap
        periods = []
        for hour in df["hours"]:
            try:
                if pd.isna(hour):
                    periods.append("Unknown")
                elif hour == 23:
                    periods.append("23-00")
                elif hour == 0:
                    periods.append("00-01")
                else:
                    periods.append(f"{hour:02d}-{hour+1:02d}")
            except:
                periods.append("Unknown")
        
        df["period"] = periods
        
        return df
        
    except Exception as e:
        print(f"Error adding date columns: {e}")
        # Fill with default values if error occurs
        df["only_date"] = pd.NaT
        df["Year"] = 0
        df["month_num"] = 0
        df["month"] = "Unknown"
        df["day"] = 0
        df["day_name"] = "Unknown"
        df["hours"] = 0
        df["minute"] = 0
        df["period"] = "00-01"
        
        return df

def _fallback_parsing(data: str) -> pd.DataFrame:
    """Fallback parsing method for non-standard formats."""
    try:
        lines = data.split('\n')
        parsed_data = []
        
        # Pattern to match any reasonable timestamp
        timestamp_pattern = r'(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}.*?\d{1,2}:\d{2})'
        
        current_message = ""
        current_date = None
        current_user = "unknown"
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line starts with a timestamp
            timestamp_match = re.match(timestamp_pattern, line)
            
            if timestamp_match:
                # Save previous message if exists
                if current_message and current_date:
                    parsed_data.append({
                        'date': current_date,
                        'user': current_user,
                        'message': current_message.strip()
                    })
                
                # Extract new timestamp and message
                timestamp_str = timestamp_match.group(1)
                remaining = line[len(timestamp_str):].strip()
                
                # Remove common separators
                remaining = re.sub(r'^[\-\s]+', '', remaining)
                
                # Try to extract user and message
                if ':' in remaining:
                    parts = remaining.split(':', 1)
                    current_user = parts[0].strip()
                    current_message = parts[1].strip()
                else:
                    current_user = "group_notification"
                    current_message = remaining
                
                current_date = _parse_flexible_date(timestamp_str)
                
            else:
                # This is a continuation of the previous message
                current_message += " " + line
        
        # Don't forget the last message
        if current_message and current_date:
            parsed_data.append({
                'date': current_date,
                'user': current_user,
                'message': current_message.strip()
            })
        
        if not parsed_data:
            return _empty_dataframe()
        
        df = pd.DataFrame(parsed_data)
        df = df.dropna(subset=['date'])
        
        if df.empty:
            return _empty_dataframe()
        
        return _add_date_columns(df)
        
    except Exception as e:
        print(f"Fallback parsing failed: {e}")
        return _empty_dataframe()

def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean the processed dataframe."""
    try:
        if df.empty:
            return df
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Remove messages that are too short or too long
        df = df[df["message"].str.len().between(1, 5000)]
        
        # Remove users with invalid names
        df = df[~df["user"].str.contains(r'[\n\r\t]', na=False)]
        df = df[df["user"].str.len().between(1, 50)]
        
        # Ensure dates are reasonable
        current_year = datetime.now().year
        df = df[(df["Year"] >= 2009) & (df["Year"] <= current_year + 1)]
        
        # Sort by date
        df = df.sort_values("date").reset_index(drop=True)
        
        return df
        
    except Exception as e:
        print(f"Error in validation: {e}")
        return df
=======


def preprocess(data):
    pattern = '\d{1,2}\/\d{1,2}\/\d{2,4}\,\s\d{1,2}\:\d{2}\s-\s'

    messages = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)

    # print("Messages found:", messages)
    # print("Dates found:", dates)

    df = pd.DataFrame({'user_message': messages, 'message_date': dates})
    # convert message_date type
    df['message_date'] = pd.to_datetime(df['message_date'], format='%d/%m/%Y, %H:%M - ')

    df.rename(columns={'message_date': 'date'}, inplace=True)

    # separate users and messages
    users = []
    messages = []
    for message in df['user_message']:
        entry = re.split('([\w\W]+?):\s', message)
        if entry[1:]:
            users.append(entry[1])
            messages.append(entry[2])
        else:
            users.append('group_notification')
            messages.append(entry[0])

    df['user'] = users
    df['message'] = messages
    df.drop(columns=['user_message'], inplace=True)

    df['only_date'] = df['date'].dt.date
    df['Year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['hours'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute

    period = []
    for hours in df[['day_name', 'hours']]['hours']:
        if hours == 23:
            period.append(str(hours) + "-" + str('00'))
        elif hours == 0:
            period.append(str('00') + "-" + str(hours + 1))
        else:
            period.append(str(hours) + "-" + str(hours + 1))
    # print("DataFrame after preprocessing:")
    # print(df.head())

    df['period'] = period

    return df
>>>>>>> fcfbd584046b32005c908f931ae5d9ff4a42871a
