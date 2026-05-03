import random
import re
from datetime import datetime, timedelta

def drop_leading_zero(date_str: str, probability: float = 0.3) -> str:
    def maybe_drop_zero(match):
        if random.random() < probability:
            return match.group(1) or match.group(2)  # Return the non-zero digit or the lone zero
        return match.group(0)  # Return the original match

    # Pattern to match leading zeros in day or month
    pattern = r'\b0(\d)|\b00'
    
    # Apply the transformation
    transformed_date = re.sub(pattern, maybe_drop_zero, date_str)
    
    return transformed_date

def choose_century(probability: float = 0.9) -> str:
    if random.random() < probability:
        return "%Y"
    return "%y"

def add_ordinal_suffix(day, probability: float = 0.5) -> str:
    if random.random() < probability:
        if 10 <= day % 100 <= 20:
            suffix = 'th'
        else:
            suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(day % 10, 'th')
        return f"{day}{suffix}"
    return day

def add_comma(probability: float = 0.5) -> str:
    if random.random() < probability:
        return ","
    return ""

def add_international_code_to_phone_number(phone_number: str, probability: float = 0.1) -> str:
    if random.random() < probability:
        return f'+{random.randint(1, 999)} {phone_number}'
    return phone_number

def ymd_to_mdy(date_str: str) -> str: # Example: 2024-12-01 -> 12/01/(20)24
    try:
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        return drop_leading_zero(date_obj.strftime(f'%m/%d/{choose_century()}')), date_obj.strftime('%Y-%m-%d')
    except ValueError:
        return date_str, date_str
    
def ymd_to_mdy2(date_str: str) -> str: # Example: 2024-12-01 -> 12/01/(20)24
    try:
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        return drop_leading_zero(date_obj.strftime(f'%m-%d-{choose_century()}')), date_obj.strftime('%Y-%m-%d')
    except ValueError:
        return date_str, date_str

def ymd_to_dmy(date_str: str) -> str: # Example: 2024-12-01 -> 01/12/(20)24
    try:
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        return drop_leading_zero(date_obj.strftime(f'%d-%m-{choose_century()}')), date_obj.strftime('%Y-%m-%d')
    except ValueError:
        return date_str, date_str

def ymd_remove_year(date_str: str) -> str: # Example: 2024-12-01 -> 12/01
    try:
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        return drop_leading_zero(date_obj.strftime('%m/%d')), f"????-{date_obj.strftime('%m-%d')}"
    except ValueError:
        return date_str, date_str

def ymd_remove_year2(date_str: str) -> str: # Example: 2024-12-01 -> 12-01
    try:
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        return drop_leading_zero(date_obj.strftime('%m-%d')), f"????-{date_obj.strftime('%m-%d')}"
    except ValueError:
        return date_str, date_str

def ymd_to_wordy(date_str: str) -> str: # Example: 2024-12-01 -> December 1(st)
    try:
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        return drop_leading_zero(date_obj.strftime(f'%B {add_ordinal_suffix(date_obj.day)}')), f"????-{date_obj.strftime('%m-%d')}"
    except ValueError:
        return date_str, date_str
    
def ymd_to_wordy2(date_str: str) -> str: # Example: 2024-12-01 -> December 1(st)(,) (20)24
    try:
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        return drop_leading_zero(date_obj.strftime(f'%B {add_ordinal_suffix(date_obj.day)}{add_comma()} {choose_century()}')), date_obj.strftime('%Y-%m-%d')
    except ValueError:
        return date_str, date_str
    
def ymd_to_wordy3(date_str: str) -> str: # Example: 2024-12-01 -> 1 December(,) (20)24
    try:
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        return drop_leading_zero(date_obj.strftime(f'%d %B{add_comma()} {choose_century()}')), date_obj.strftime('%Y-%m-%d')
    except ValueError:
        return date_str, date_str

def ymd_remove_day(date_str: str) -> str: # Example: 2024-12-01 -> 12/(20)24
    try:
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        return drop_leading_zero(date_obj.strftime(f'%m/{choose_century()}')), f"{date_obj.strftime('%Y-%m')}-??"
    except ValueError:
        return date_str, date_str

def ymd_remove_day_and_month(date_str: str) -> str: # Example: 2024-12-01 -> 2024
    try:
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        return drop_leading_zero(date_obj.strftime(f'%Y')), f"{date_obj.strftime('%Y')}-??-??"
    except ValueError:
        return date_str, date_str
    
def ymd_remove_day_and_year(date_str: str) -> str: # Example: 2024-12-01 -> December
    try:
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        return drop_leading_zero(date_obj.strftime(f'%B')), f"????-{date_obj.strftime('%m')}-??"
    except ValueError:
        return date_str, date_str

def ymd_to_wordy_month_only(date_str: str) -> str:  # Example: 2024-12-01 -> December 2024
    try:
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        return drop_leading_zero(date_obj.strftime(f'%B {choose_century()}')), f"{date_obj.strftime('%Y-%m')}-??"
    except ValueError:
        return date_str, date_str

def ymd_to_wordy_short_month(date_str: str) -> str:  # Example: 2024-12-01 -> Dec 1(st), (20)24 
    try:
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        return drop_leading_zero(date_obj.strftime(f'%b {add_ordinal_suffix(date_obj.day)}, {choose_century()}')), date_obj.strftime('%Y-%m-%d')
    except ValueError:
        return date_str, date_str

def ymd_to_wordy_short_month_no_year(date_str: str) -> str:  # Example: 2024-12-01 -> Dec 1(st)
    try:
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        return drop_leading_zero(date_obj.strftime(f'%b {add_ordinal_suffix(date_obj.day)}')), f"????-{date_obj.strftime('%m-%d')}"
    except ValueError:
        return date_str, date_str

def ymd_to_wordy_comma_style(date_str: str) -> str:  # Example: 2024-12-01 -> December 1, 2024
    try:
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        return drop_leading_zero(date_obj.strftime(f'%B {date_obj.day}, %Y')), date_obj.strftime('%Y-%m-%d')
    except ValueError:
        return date_str, date_str

def ymd_to_wordy_no_comma(date_str: str) -> str:  # Example: 2024-12-01 -> December 1 2024
    try:
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        return drop_leading_zero(date_obj.strftime(f'%B {date_obj.day} %Y')), date_obj.strftime('%Y-%m-%d')
    except ValueError:
        return date_str, date_str
    
def introduce_misspelling_in_name(text: str) -> str:
    words = text.split()
    if not words:
        return text.group(0)
    
    word_to_misspell = random.choice(words)
    
    if random.choice([True, False]):  # Randomly choose between addition and removal
        # Addition
        position = random.randint(0, len(word_to_misspell))
        letter_to_add = random.choice('abcdefghijklmnopqrstuvwxyz')
        misspelled_word = word_to_misspell[:position] + letter_to_add + word_to_misspell[position:]
    else:
        # Removal
        if len(word_to_misspell) > 1:
            position = random.randint(0, len(word_to_misspell) - 1)
            misspelled_word = word_to_misspell[:position] + word_to_misspell[position + 1:]
        else:
            misspelled_word = word_to_misspell
    
    words[words.index(word_to_misspell)] = misspelled_word
    
    return " ".join(words)

def global_remove_comma_before_suffixes(text: str) -> str:
    suffixes = ['MD', 'DO', 'NP', 'RN', 'PA', 'PT', 'OT', 'PTA', 'NP']

    def remove_comma_before_suffix(match):
        return match.group(1) + " " + match.group(3)
    
    pattern = r'(\b\w+\s*),(\s*)(' + '|'.join(suffixes) + r')'
    
    for match in re.finditer(pattern, text):
        if random.random() < 0.1:
            text = text.replace(match.group(0), remove_comma_before_suffix(match))
    return text

def global_sanitize_date(text: str) -> str:
    def get_random_day():
        return str(random.randint(1, 28)).zfill(2)
    
    def get_random_year():
        return str(random.randint(1980, 2024))
    
    # Extract date tags from the text
    date_tags = re.findall(r'<date.*?>(.*?)</date>', text)
    
    for date_tag in date_tags:
        # Remove all characters except digits, hyphens, and question marks
        sanitized_date = re.sub(r'[^\d\-\?]', '', date_tag)

        # Check if the sanitized date is in YYYY-MM or YYYY-MM-?? format and add '-01' if necessary
        if re.match(r'^\d{4}-\d{2}$', sanitized_date):
            sanitized_date += f'-{get_random_day()}'
        elif re.match(r'^(\d{4}|\?{4})-(\d{2}|\?{2})-\?{2}$', sanitized_date):
            sanitized_date = sanitized_date[:-3] + f'-{get_random_day()}'
        
        # Check if the sanitized date is in MM-DD or ????-MM-DD format and add '1990-' if necessary
        if re.match(r'^\d{2}-\d{2}$', sanitized_date):
            sanitized_date = f'{get_random_year()}-{sanitized_date}'
        elif re.match(r'^\?{4}-\d{2}-\d{2}$', sanitized_date):
            sanitized_date = f'{get_random_year()}-{sanitized_date[-5:]}'
        
        # Replace the original date with the sanitized version
        text = text.replace(f'<date>{date_tag}</date>', f'<date>{sanitized_date}</date>')
    
    return text

def remove_spaces_in_phone_number(text: str) -> str: # Example: 512-231-1234 -> 5073198906
    phone_numbers = re.findall(r'\d+', text)
    if len(phone_numbers) == 1:
        phone_number = phone_numbers[0]
        if len(phone_number) == 10:
            text = add_international_code_to_phone_number(phone_number)
    return text

def add_parantheses_to_phone_number(text: str) -> str: # Example: 512-231-1234 -> (512) 231-1234
    phone_numbers = re.findall(r'\d+', text)
    if len(phone_numbers) == 1:
        phone_number = phone_numbers[0]
        if len(phone_number) == 10:
            text = add_international_code_to_phone_number(f'({phone_number[:3]}) {phone_number[3:6]}-{phone_number[6:]}')
    return text

def add_dash_to_phone_number(text: str) -> str: # Example: 512-231-1234 -> 512-231-1234
    phone_numbers = re.findall(r'\d+', text)
    if len(phone_numbers) == 1:
        phone_number = phone_numbers[0]
        if len(phone_number) == 10:
            text = add_international_code_to_phone_number(f'{phone_number[:3]}-{phone_number[3:6]}-{phone_number[6:]}')
    return text

def add_space_between_phone_number_parts(text: str) -> str: # Example: 512-231-1234 -> 512 231 1234
    phone_numbers = re.findall(r'\d+', text)
    if len(phone_numbers) == 1:
        phone_number = phone_numbers[0]
        if len(phone_number) == 10:
            text = add_international_code_to_phone_number(f'{phone_number[:3]} {phone_number[3:6]} {phone_number[6:]}')
    return text

def global_uniform_date(text: str) -> str:
    # find all dates (<date>), select one of the augmentations by random and apply it to all dates
    dates = re.findall(r'<date>(.*?)</date>', text)
    augmentations = [ # One of
        (ymd_to_mdy, 0.15),
        (ymd_to_dmy, 0.15),
        (ymd_to_mdy2, 0.15),
        (ymd_remove_year, 0.45),
        (ymd_remove_year2, 0.45),
        (ymd_to_wordy, 0.15),
        (ymd_to_wordy2, 0.15),
        (ymd_to_wordy3, 0.15),
        (ymd_remove_day, 0.30),
        (ymd_remove_day_and_month, 0.30),
        (ymd_remove_day_and_year, 0.30),
        (ymd_to_wordy_short_month_no_year, 0.15),
        (ymd_to_wordy_comma_style, 0.15),
        (ymd_to_wordy_no_comma, 0.15),
        (ymd_to_wordy_short_month, 0.15)
    ]
    selected_augmentation, prob = random.choices(augmentations, weights=[a[1] for a in augmentations])[0]
    for date in dates:
        new_date, ymd_date = selected_augmentation(date)
        text = text.replace(f'<date>{date}</date>', f'<date formatted="{ymd_date}">{new_date}</date>')
    return text

def global_date_random_shift(text: str) -> str:
    days_shift = random.randint(-3650, 3650)
    dates = re.findall(r'\b\d{4}-\d{2}-\d{2}\b', text)

    for date in dates:
        try:
            shifted_date = datetime.strptime(date, '%Y-%m-%d') + timedelta(days=days_shift)
            text = text.replace(date, shifted_date.strftime('%Y-%m-%d'))
        except ValueError:
            continue
    
    return text

def global_add_time_to_date(text: str) -> str:
    # add time to date
    def get_random_time_24h():
        return f'{random.randint(0, 23):02d}:{random.randint(0, 59):02d}{":{random.randint(0, 59):02d}" if random.random() < 0.5 else ""}'
    
    def get_random_time_12h():
        return f'{random.randint(1, 12):02d}:{random.randint(0, 59):02d}{":{random.randint(0, 59):02d}" if random.random() < 0.5 else ""}{" " if random.random() < 0.8 else ""}{random.choice(["AM", "PM", "am", "pm", "a.m.", "p.m.", "A.M.", "P.M.", "a.m", "p.m", "A.M", "P.M", "AM.", "PM.", "am.", "pm."])}.'
    
    dates = re.findall(r'(<date>.*?</date>)', text)
    for date in dates:
        if random.random() < 0.5:
            text = text.replace(date, f'{date}{" " if random.random() < 0.8 else ""}{get_random_time_24h() if random.random() < 0.5 else get_random_time_12h()}')
    return text

def global_remove_on_before_date(text: str) -> str:
    def process_date(match):
        full_match = match.group(0)
        date = match.group(1)
        
        if random.random() < 0.5:
            # Remove "on" before the date if it exists
            # also replace in with other prepositions
            if random.random() < 0.33:
                return re.sub(r'\bon\s+', 'in', full_match, flags=re.IGNORECASE)
            elif random.random() < 0.66:
                return re.sub(r'\bon\s+', 'at', full_match, flags=re.IGNORECASE)
            else:
                return re.sub(r'\bon\s+', '', full_match, flags=re.IGNORECASE)
        else:
            # Remove "on" before the date if it exists and put date in parentheses
            without_on = re.sub(r'\bon\s+', '', full_match, flags=re.IGNORECASE)
            return re.sub(r'<date>(.*?)</date>', r'(<date>\1</date>)', without_on)
        
    # Process all date tags in the text
    processed_text = re.sub(r'\bon\s+<date>(.*?)</date>', process_date, text, flags=re.IGNORECASE)
    
    return processed_text

def global_join_paragraphs(text: str) -> str:
    paragraphs = re.split(r'(?:\n){1,3}', text)
    return ''.join(paragraphs)

def global_drop_prepositions(text: str) -> str:
    prepositions = ['a', 'an', 'the', 'at', 'by', 'for', 'in', 'of', 'on', 'up', 'down', 'out', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how']
    words = text.split()
    if random.random() < 0.1:  # 10% of the times a component is dropped
        filtered_words = [word for word in words if word.lower() not in prepositions]
    else:
        filtered_words = words
    return ' '.join(filtered_words)

def global_convert_yo(text: str) -> str:
    prob = random.random()
    if prob < 0.1:
        text = text.replace("-year-old", "y.o.")
    elif prob < 0.2:
        text = text.replace("-year-old", "years old")
    elif prob < 0.3:
        text = text.replace("-year-old", "yo")
    elif prob < 0.4:
        text = text.replace("-year-old", "y")
    elif prob < 0.5:
        text = text.replace("-year-old", " years old")
    elif prob < 0.6:
        text = text.replace("-year-old", " y.o.")
    elif prob < 0.7:
        text = text.replace("-year-old", " yo")
    elif prob < 0.8:
        text = text.replace("-year-old", " y")
    elif prob < 0.9:
        text = text.replace("-year-old", " years-old")
    elif prob < 1.0:
        text = text.replace("-year-old", "year-old")
    
    return text

def global_replace_patient_names(text: str) -> str:
    def replace_name(match):
        full_match = match.group(0)
        name = match.group(3)
        
        # 50% chance of replacement
        if random.random() < 0.5:
            # Choose between "patient" and "PT" (50% chance each)
            replacement = "patient" if random.random() < 0.5 else "pt"
            
            # Capitalize if at the beginning of a sentence
            if re.match(r'(^|[.!?]\s+)[<\w]', full_match):
                replacement = replacement.title()
            
            return replacement
        else:
            # If not replacing, return the original full match
            return full_match

    # Pattern to match patient name tags
    pattern = r'(<name id="(\d+)" type="patient">)(.*?)(</name>)'
    
    # Apply the replacement
    processed_text = re.sub(pattern, replace_name, text)
    
    return processed_text

def global_update_ids(text: str) -> str:
    def replace_id(match):
        full_tag = match.group(0)
        current_id = match.group(2)
        tag_type = match.group(1)
        
        # Create separate counters for each tag type
        if tag_type not in id_mapping:
            id_mapping[tag_type] = {}
            
        if current_id not in id_mapping[tag_type]:
            id_mapping[tag_type][current_id] = len(id_mapping[tag_type].keys()) + 1
        
        new_id = id_mapping[tag_type][current_id]
        return full_tag.replace(f'id="{current_id}"', f'id="{new_id}"')

    id_mapping = {}
    
    # Pattern to match complete tags with id attribute
    pattern = r'<(name|place)[^>]*\bid="([^"]*)"[^>]*>'
    # Apply the replacement
    processed_text = re.sub(pattern, replace_id, text)
    return processed_text

def global_drop_title_before_name(text: str) -> str:
    titles = ['Mrs', 'Mr', 'Ms', 'Dr', 'Miss', 'Prof']
    
    def remove_title(match):
        title = match.group(1)
        name_tag = match.group(2)
        
        return name_tag

    # Pattern to match titles before name tags
    pattern = rf'\b({"|".join(titles)})\.?\s*(<name[^>]*>.*?</name>)'
    
    # Apply the replacement
    processed_text = re.sub(pattern, remove_title, text)
    
    return processed_text

def global_drop_dot_in_title(text: str) -> str:
    titles = ['Mrs', 'Mr', 'Ms', 'Dr', 'Miss', 'Prof']
    
    def remove_dot_after_title(match):
        title = match.group(1)
        rest = match.group(2)
        return f"{title}{rest}"

    # Pattern to match titles with optional dot
    pattern = rf'\b({"|".join(titles)})\.?(\s*)'
    
    # Apply the replacement
    processed_text = re.sub(pattern, remove_dot_after_title, text)
    
    return processed_text

def fix_lastname_firstname(text: str, probability: float = 0.8) -> str:
    name = text
    if " " in name:
        parts = len(name.split(" "))
        if parts == 2:
            lastname, firstname = name.rsplit(" ", 1)
            name = f'{lastname}, {firstname}'
        elif parts == 3:
            lastname, firstname, middlename = name.rsplit(" ", 2)
            if random.random() < probability:
                name = f'{lastname}, {firstname} {middlename.capitalize()}.'
            else:
                name = f'{lastname}, {firstname.capitalize()}. {middlename.capitalize()}.'
        else:
            return name
    
    return name

def global_remove_phis(text: str) -> str:
    # remove all phi tags and replace the content with *
    tags = ['name', 'age', 'date', 'place', 'address', 'phone', 'email', 'id', 'url', 'device']
    pattern = rf'<({"|".join(tags)})[^>]*>(.*?)</\1>'
    return re.sub(pattern, '', text)

def global_remove_spaces_inside_tags(text: str) -> str:
    tags = ['name', 'age', 'date', 'place', 'address', 'phone', 'email', 'id', 'url', 'device']
    pattern = rf'<({"|".join(tags)})[^>]*>(.*?)</\1>'
    # return re.sub(pattern, lambda m: f'<{m.group(1)}{m.group(0).split(">")[0].split(m.group(1))[1]}>{m.group(2).replace(" ", "")}</{m.group(1)}>', text)
    # do this per matched tag
    matches = re.findall(pattern, text)
    for match in matches:
        tag, content = match
        if random.random() < 0.05:
            pattern = rf'<{tag}([^>]*)>{content}</{tag}>'
            tag_match = re.search(pattern, text)
            if tag_match:
                attributes = tag_match.group(1)
                text = text.replace(f'<{tag}{attributes}>{content}</{tag}>', f'<{tag}{attributes}>{content.replace(" ", "")}</{tag}>')
    return text

def global_make_sentence_uppercase(text: str) -> str:
    # Split by line breaks first
    lines = text.split("\n")
    result = []
    
    def uppercase_preserving_tags(text: str) -> str:
        # Store tags and replace with placeholders
        tags = []
        def store_tag(match):
            tags.append(match.group(0))
            return f"__TAG_{len(tags)-1}__"
        
        # Replace tags with placeholders
        text_with_placeholders = re.sub(r'<[^>]+>', store_tag, text)
        
        # Uppercase the text
        text_with_placeholders = text_with_placeholders.upper()
        
        # Restore tags
        for i, tag in enumerate(tags):
            text_with_placeholders = text_with_placeholders.replace(f"__TAG_{i}__", tag)
        
        return text_with_placeholders
    
    for line in lines:
        # Split each line by periods
        sentences = line.split(".")
        # Process each sentence
        for i, sentence in enumerate(sentences):
            if sentence.strip():  # Only process non-empty sentences
                if random.random() < 0.01:
                    sentences[i] = uppercase_preserving_tags(sentence)
        # Join sentences back with periods
        result.append(".".join(sentences))
    
    # Join lines back with line breaks
    return "\n".join(result)