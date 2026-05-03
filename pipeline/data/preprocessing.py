import pandas as pd
import re
from datetime import datetime

## Finding gibberish generated content
# We will find content that is not a URL and is more than 30 characters long.

def is_gibberish(text):
   matches = re.findall(r'\b(?!<[^>]+>)[^\s]{30,}\b', text)
   
   is_error = False
   if matches:
       for match in matches:
           if 'www' not in match and 'http' not in match and "<" not in match and ">" not in match and ("@" not in match and (".com" not in match) or (".org" not in match) or (".net" not in match) or (".edu" not in match) or (".gov" not in match)):
               is_error = True
               break

   if is_error:
       return True
   
   # if &lt; or &gt; in the text, return True
   if "&lt;" in text or "&gt;" in text:
       return True
   
   return False

## Fix space in closing tags
# We will find space in closing tags and remove it.

def fix_space_in_closing_tags(text): 
    text = re.sub(r'</ *(\w+) *>', r'</\1>', text)
    return text

## Fixing space in tags
# We will find space in tags and remove it (e.g. <date>2023-01-01 </date> -> <date>2023-01-01</date>)

def fix_space_in_tags(text): 
    # Find all XML-style tags and remove extra spaces inside them
    text = re.sub(r'<(\w+)>\s*(.*?)\s*</\1>', r'<\1>\2</\1>', text)
    return text

## Fixing issues with phone numbers
# We will find phone numbers and replace them with <phone>1234567890</phone>


def fix_unformatted_phone_numbers(text):
    patterns = [r'\b\d{3}-\d{3}-\d{4}\b', r'\b\d{3}\s\d{3}\s\d{4}\b', r'\(\d{3}\)\s\d{3}[-\s]\d{4}']
    total_changes = 0
    for pattern in patterns:
        text_matches = re.findall(pattern, text)
        for match in text_matches:
            if not re.search(f"<phone>{re.escape(match)}</phone>", text):
                text = text.replace(match, f"<phone>{match}</phone>")
                total_changes += 1
    return text, total_changes

## Fixing issues with dates
# We will find dates (YYYY-MM-DD) and replace them with <date>YYYY-MM-DD</date>

def fix_unformatted_dates(text):
    pattern = r'\b\d{4}-\d{2}-\d{2}\b'
    total_changes = 0
    text_matches = re.findall(pattern, text)
    for match in text_matches:
        if not re.search(f"<date>{match}</date>", text):
            text = text.replace(match, f'<date>{match}</date>')
            total_changes += 1
    return text, total_changes

## Fixing age issue
# Sometimes the LLM will use "1-year-old" instead of "<age>1</age>-year-old". We will correct for this.

def format_ages(text):
    # Regular expression to find ages
    age_pattern = r'(\d+)[-\s]year[-\s]old'
    
    # Function to replace matches with formatted age
    def replace_age(match):
        age = match.group(1)
        return f'<age>{age}</age>-year-old'
    
    # Replace ages that are already formatted
    text = re.sub(r'<age>\d+</age>[-\s]year[-\s]old', lambda m: m.group(0), text)
    
    # Format unformatted ages
    text = re.sub(age_pattern, replace_age, text)
    
    return text

def search_unprocessed_ages(df, column_name):
    # Regular expression to find unprocessed ages
    unprocessed_age_pattern = r'\b\d+[-\s]year[-\s]old\b'
    
    # Function to check if a row contains unprocessed ages
    def contains_unprocessed_age(text):
        return bool(re.search(unprocessed_age_pattern, text))
    
    # Filter rows containing unprocessed ages
    unprocessed_rows = df[df[column_name].apply(contains_unprocessed_age)]
    
    return unprocessed_rows

def process_ages_in_dataframe(df, column_name):
    # Search for unprocessed ages
    unprocessed_rows = search_unprocessed_ages(df, column_name)
    
    if not unprocessed_rows.empty:
        df[column_name] = df[column_name].apply(format_ages)
    
    return df

## Fixing gender issue
# Sometimes the LLM will use "individual" instead of "man" or "woman". We will correct for this.

def correct_gender_terms(df, original_col='default_response_0', processed_col='default_response_1'):
    def get_first_paragraph(text):
        paragraphs = text.split('\n')
        return paragraphs[0] if paragraphs else ''

    def find_gender_term(text):
        gender_terms = ['man', 'woman', 'male', 'female']
        for term in gender_terms:
            if term in text.lower():
                return term
        return None

    def replace_individual(original_para, processed_para):
        gender_term = find_gender_term(original_para)
        if gender_term and 'individual' in processed_para.lower():
            return re.sub(r'\bindividual\b', gender_term, processed_para, flags=re.IGNORECASE)
        return processed_para

    corrections = 0
    for index, row in df.iterrows():
        original_first_para = get_first_paragraph(row[original_col])
        processed_first_para = get_first_paragraph(row[processed_col])
        
        corrected_para = replace_individual(original_first_para, processed_first_para)
        
        if corrected_para != processed_first_para:
            corrections += 1
            df.at[index, processed_col] = row[processed_col].replace(processed_first_para, corrected_para, 1)

    return df, corrections

## Fixing name prefix issue
# The LLM has considered the name prefix as inside the name tag. We will correct for this.

def correct_name_tag_prefix(text):
    # List of titles to check for, ordered to prevent Mr/Mrs confusion
    titles = ['Mrs', 'Mr', 'Ms', 'Dr', 'Miss', 'Prof', 'NP', 'Nurse Practitioner', 'Nurse', 'Doctor']
    
    # Create a regex pattern for titles, optionally followed by a dot and optional space
    title_pattern = r'(' + '|'.join(titles) + r')(\.?)(\s?)(<name[^>]*>)'
    
    def reposition_title(match):
        title = match.group(1)
        dot = match.group(2)  # This will be '.' if present, or '' if not
        space = match.group(3)  # This will be ' ' if present, or '' if not
        name_tag = match.group(4)  # This captures the entire name tag with its properties
        return f"{title}{dot}{space}{name_tag}"
    
    # Correct titles that are immediately before or inside the name tags
    corrected_text, count = re.subn(title_pattern, reposition_title, text)
    
    # Correct cases where the title is inside the name tag
    inside_pattern = r'(<name[^>]*>)(' + '|'.join(titles) + r')(\.?)(\s?)'
    corrected_text, count2 = re.subn(inside_pattern, lambda m: f"{m.group(2)}{m.group(3)}{m.group(4)}{m.group(1)}", corrected_text)
    
    # Total number of corrections
    total_corrections = count + count2
    
    return corrected_text, total_corrections

def correct_name_tag_suffix(text):
    # List of suffixes to check for
    suffixes = ['MD', 'DO', 'NP', 'RN', 'PA', 'PT', 'OT', 'PTA', 'NP']
    
    # Create a regex pattern for suffixes, capturing any comma and/or space
    suffix_pattern = r'<name([^>]*)>(.*?)(,?\s*)?(' + '|'.join(suffixes) + r')</name>'
    
    def reposition_suffix(match):
        attrs = match.group(1)    # Capture the tag attributes
        name = match.group(2)     # Capture the name
        separator = match.group(3) or ''  # Capture the separator (comma/space) or empty string
        suffix = match.group(4)   # Capture the suffix
        return f'<name{attrs}>{name}</name>{separator}{suffix}'
    
    # Correct suffixes that are inside the name tags
    corrected_text, count = re.subn(suffix_pattern, reposition_suffix, text)
    
    return corrected_text, count

# Function to process a DataFrame
def process_names_tags(df, column_name):
    total_fixes = 0
    
    def apply_correction(text):
        nonlocal total_fixes
        corrected, count = correct_name_tag_prefix(text)
        total_fixes += count
        corrected, count = correct_name_tag_suffix(corrected)
        total_fixes += count
        return corrected
    
    df[column_name] = df[column_name].apply(apply_correction)
    return df, total_fixes

## Fixing provider name type issue
# We have tagged the provider name with type="doctor" or type="nurse" or type="other" or type="patient" etc. Now we will change that to wither provider or patient.

def simplify_name_tags(df, column_name='default_response_1'):
    def process_text(text):
        pattern = r'<name id="(\d+)" type="(doctor|nurse|other|patient)">(.*?)</name>'
        
        def replace_tag(match):
            id_num, type_, name = match.groups()
            new_type = "provider" if type_ in ["doctor", "nurse", "nurse practitioner", "other"] else "patient"
            return f'<name id="{id_num}" type="{new_type}">{name}</name>'
        
        new_text, count = re.subn(pattern, replace_tag, text)
        return new_text, count

    total_changes = 0
    for index, row in df.iterrows():
        new_text, changes = process_text(row[column_name])
        df.at[index, column_name] = new_text
        total_changes += changes

    return df, total_changes

## Fix ID issues
# Some of the IDs are not consistent. We will fix them by adding a suffix to the IDs.

def reformat_id_tags(df, column_name='default_response_1'):
    def process_text(text):
        pattern = r'<id type="([^"]+)">([^:]+):(\s?)(\S+)</id>'
        
        def replace_tag(match):
            type_, prefix, space, id_num = match.groups()
            return f'{prefix}:{space}<id type="{type_}">{id_num}</id>'
        
        new_text, count = re.subn(pattern, replace_tag, text)
        return new_text, count

    total_changes = 0
    for index, row in df.iterrows():
        new_text, changes = process_text(row[column_name])
        df.at[index, column_name] = new_text
        total_changes += changes

    return df, total_changes


## Fixing dates
# We will find missed dates and replace them with <date>YYYY-MM-DD</date>

def extract_and_format_dates(text):
    months = ["January", "February", "March", "April", "June", "July", "August", "September", "October", "November", "December"]
    changes = 0
    
    # Define a regex pattern to find dates in the format "Month Day, Year"
    pattern = r'\b(?:' + '|'.join(months) + r')\s+\d{1,2},\s+\d{4}\b'
    matches = re.findall(pattern, text, re.IGNORECASE)
    for date_str in matches:
        try:
            date_obj = datetime.strptime(date_str.strip(), '%B %d, %Y')
            text = text.replace(date_str, f'<date>{date_obj.strftime("%Y-%m-%d")}</date>')
            changes += 1
        except ValueError as e:
            print(f"Error parsing date '{date_str}': {e}")

    # Define a regex pattern to find dates in the format "Month Day(st/nd/rd/th), year"
    pattern = r'\b((?:' + '|'.join(months) + r'))\s+(\d{1,2})(?:st|nd|rd|th),\s+(\d{4})\b'
    matches = re.findall(pattern, text, re.IGNORECASE)
    for match in matches:
        try:
            month, day, year = match
            date_str = f"{month} {day}, {year}"
            date_obj = datetime.strptime(date_str, '%B %d, %Y')
            original_date = f"{month} {day}th, {year}"
            text = text.replace(original_date, f'<date>{date_obj.strftime("%Y-%m-%d")}</date>')
            changes += 1
        except ValueError as e:
            print(f"Error parsing date '{date_str}': {e}")
                    
    # Define a regex pattern to find dates in the format "Month Year"
    pattern = r'\b(?:' + '|'.join(months) + r')\s+\d{4}\b'
    matches = re.findall(pattern, text, re.IGNORECASE)
    for date_str in matches:
        try:
            date_obj = datetime.strptime(date_str.strip(), '%B %Y')
            text = text.replace(date_str, f'<date>{date_obj.strftime("%Y-%m-??")}</date>')
            changes += 1
        except ValueError as e:
            print(f"Error parsing date '{date_str}': {e}")

    # Define a regex pattern to find dates in the format "Month Day"
    pattern = r'\b(?:' + '|'.join(months) + r')\s+\d{1,2}\b'
    matches = re.findall(pattern, text, re.IGNORECASE)
    for date_str in matches:
        try:
            date_obj = datetime.strptime(date_str.strip(), '%B %d')
            text = text.replace(date_str, f'<date>{date_obj.strftime("????-%m-%d")}</date>')
            changes += 1
        except ValueError as e:
            print(f"Error parsing date '{date_str}': {e}")

    # Define a regex pattern to find dates in the format "Month Day(st/nd/rd/th)"
    pattern = r'\b((?:' + '|'.join(months) + r'))\s+(\d{1,2})(?:st|nd|rd|th)\b'
    matches = re.findall(pattern, text, re.IGNORECASE)
    for match in matches:
        try:
            month, day = match
            date_str = f"{month} {day}"
            date_obj = datetime.strptime(date_str, '%B %d')
            original_date = f"{month} {day}th"
            text = text.replace(original_date, f'<date>????-{date_obj.strftime("%m-%d")}</date>')
            changes += 1
        except ValueError as e:
            print(f"Error parsing date '{date_str}': {e}")

    # Define a regex pattern to find standalone months (not at start if "May")
    pattern = r'(?<!^)(?<![\.\!\?\n])(?<![\.\!\?\n]\s)(?<![\.\!\?\n]\s\s)\b((?:' + '|'.join(months) + r'))\b(?![a-zA-Z])'
    matches = re.findall(pattern, text)
    for match in matches:
        if isinstance(match, tuple):
            month = match[0]
        else:
            month = match
        
        date_obj = datetime.strptime(month.strip(), '%B')
        text = text.replace(month, f'<date>????-{date_obj.strftime("%m")}-??</date>')
        changes += 1

    # remove double date tags <date><date>2023-04-15</date></date>
    text = re.sub(r'<date><date>(.*?)</date></date>', r'<date>\1</date>', text)
    return text, changes

## Removing Markdowns
# We will remove the markdown tags from the text.

def remove_markdown(df, column_name='default_response_1'):
    """
    Remove markdown formatting from text while preserving the content.
    Handles bold (**), italic (*), underscores (_), and URLs.
    
    Args:
        df (DataFrame): The input DataFrame
        column_name (str): Name of the column to process
        
    Returns:
        tuple: (DataFrame, int) - Modified DataFrame and number of changes made
    """
    def find_matching_close(text: str, start_index: int, symbol: str) -> int:
        i = start_index + len(symbol)
        while i < len(text):
            if text.startswith(symbol, i):
                if i + len(symbol) >= len(text) or text[i + len(symbol)] != symbol[0]:
                    return i
            i += 1
        return -1

    def process_single_text(text):
        original_text = text
        result = text
        start_index = 0
        
        while start_index < len(result):
            patterns = ['**', '__', '*', '_']
            
            found = False
            for pattern in patterns:
                if result.startswith(pattern, start_index):
                    close_index = find_matching_close(result, start_index, pattern)
                    
                    if close_index != -1:
                        content = result[start_index + len(pattern):close_index]
                        result = result[:start_index] + content + result[close_index + len(pattern):]
                        found = True
                        break

            if not found:
                url_match = re.search(r'\[([^\]]+)\]\(([^)]+)\)', result[start_index:])
                if url_match:
                    full_match = url_match.group(0)
                    link_text = url_match.group(1)
                    match_index = start_index + url_match.start()
                    result = result[:match_index] + link_text + result[match_index + len(full_match):]
                    found = True

            if not found:
                start_index += 1
                
        return result, 1 if result != original_text else 0

    total_changes = 0
    for index, row in df.iterrows():
        new_text, changes = process_single_text(row[column_name])
        df.at[index, column_name] = new_text
        total_changes += changes

    return df, total_changes


def preprocess_text(df, input_col='default_response_0', output_col='default_response_1'):
    """
    Process text data through all available preprocessing steps
    """
    print("Starting text processing pipeline...")
    
    # Remove gibberish content
    gibberish_indices = df[df[output_col].apply(is_gibberish)].index
    df.drop(index=gibberish_indices, inplace=True)
    df.reset_index(inplace=True, drop=True)
    
    # Initialize counters
    phone_changes = 0
    date_changes = 0
    age_changes = 0
    gender_changes = 0
    name_tag_changes = 0
    name_type_changes = 0
    id_tag_changes = 0
    remaining_date_changes = 0
    
    # Fix spaces in closing tags
    print("Fixing spaces in closing tags...")
    df[output_col] = df[output_col].apply(fix_space_in_closing_tags)
    
    # Fix spaces in tags
    print("Fixing spaces in tags...")
    df[output_col] = df[output_col].apply(fix_space_in_tags)
    
    # Apply all formatting functions
    print("Processing phone numbers...")
    for idx, row in df.iterrows():
        text, changes = fix_unformatted_phone_numbers(row[output_col])
        df.at[idx, output_col] = text
        phone_changes += changes
    
    print("Processing dates...")
    for idx, row in df.iterrows():
        text, changes = fix_unformatted_dates(row[output_col])
        df.at[idx, output_col] = text
        date_changes += changes
    
    print("Processing ages...")
    pre_change_ages = len(df[df[output_col].str.contains(r'<age>\d+</age>')])
    df = process_ages_in_dataframe(df, output_col)
    age_changes = len(df[df[output_col].str.contains(r'<age>\d+</age>')]) - pre_change_ages
    
    print("Correcting gender terms...")
    df, gender_changes = correct_gender_terms(df, input_col, output_col)
    
    print("Processing name tags...")
    df, name_tag_changes = process_names_tags(df, output_col)
    
    print("Simplifying name tags...")
    df, name_type_changes = simplify_name_tags(df, output_col)
    
    print("Reformatting ID tags...")
    df, id_tag_changes = reformat_id_tags(df, output_col)
    
    # Process remaining dates
    months = ["January", "February", "March", "April", "June", "July", "August", 
             "September", "October", "November", "December"]
    
    print("Processing remaining dates...")
    non_date_rows = df[df[output_col].str.contains('|'.join(months))]
    initial_count = len(non_date_rows)
    
    if not non_date_rows.empty:
        for idx, row in non_date_rows.iterrows():
            text, changes = extract_and_format_dates(row[output_col])
            df.at[idx, output_col] = text
            remaining_date_changes += changes
    
    print("Removing markdown tags...")
    df, markdown_changes = remove_markdown(df, output_col)
    
    # Print summary of all changes
    print("\nProcessing Summary:")
    print(f"Phone numbers formatted: {phone_changes}")
    print(f"Dates formatted: {date_changes} + {remaining_date_changes}")
    print(f"Ages processed: {age_changes}")
    print(f"Gender terms corrected: {gender_changes}")
    print(f"Name tags processed: {name_tag_changes}")
    print(f"Name types simplified: {name_type_changes}")
    print(f"ID tags reformatted: {id_tag_changes}")
    print(f"Markdown tags removed: {markdown_changes}")
    
    print("\nText processing pipeline completed!")
    return df

