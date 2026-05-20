import difflib
import pandas as pd
import re
from datetime import datetime

# Matches any XML/HTML tag — used by XML-aware helpers below.
_XML_TAG = re.compile(r'<[^>]+>')

def _apply_to_text_segments(func, text, skip_element_content=False):
    """Apply func(segment) -> (segment, int) only to plain-text parts between XML tags.

    Splits text on XML tags, applies func to each non-tag segment, then
    reassembles. This prevents any text-munging function from accidentally
    modifying tag names, attribute values, or already-tagged content.

    skip_element_content=True: also skip segments that are the direct content
    of an XML element (i.e. the segment immediately follows an opening tag).
    Use this for functions that should only process top-level / inter-element
    text, not values stored inside existing tags (e.g. <original>Feb 13</original>).
    """
    parts    = _XML_TAG.split(text)
    xml_tags = _XML_TAG.findall(text)
    total    = 0
    new_parts = []
    for i, part in enumerate(parts):
        if skip_element_content and i > 0:
            preceding = xml_tags[i - 1]
            # Skip if immediately inside an opening tag (not a closing or self-closing tag)
            if not preceding.startswith('</') and not preceding.endswith('/>'):
                new_parts.append(part)
                continue
        new_part, n = func(part)
        new_parts.append(new_part)
        total += n
    result = new_parts[0]
    for tag, part in zip(xml_tags, new_parts[1:]):
        result += tag + part
    return result, total


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


_PHONE_PATTERNS = [
    r'\b\d{3}-\d{3}-\d{4}\b',
    r'\b\d{3}\s\d{3}\s\d{4}\b',
    r'\(\d{3}\)\s\d{3}[-\s]\d{4}',
]

def fix_unformatted_phone_numbers(text):
    def _fix(segment):
        changes = 0
        for pattern in _PHONE_PATTERNS:
            for match in re.findall(pattern, segment):
                if f"<phone>{match}</phone>" not in segment:
                    segment = segment.replace(match, f"<phone>{match}</phone>", 1)
                    changes += 1
        return segment, changes
    return _apply_to_text_segments(_fix, text, skip_element_content=True)

## Fixing issues with dates
# We will find dates (YYYY-MM-DD) and replace them with <date>YYYY-MM-DD</date>

def fix_unformatted_dates(text):
    _DATE_PATTERN = re.compile(r'\b\d{4}-\d{2}-\d{2}\b')

    def _fix(segment):
        changes = 0
        for match in _DATE_PATTERN.findall(segment):
            if f"<date>{match}</date>" not in segment:
                segment = segment.replace(match, f'<date>{match}</date>', 1)
                changes += 1
        return segment, changes

    return _apply_to_text_segments(_fix, text, skip_element_content=True)

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
    corrected_text, count = re.subn(suffix_pattern, reposition_suffix, text, flags=re.DOTALL)
    
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
        changes = [0]

        def replace_tag(match):
            id_num, type_, name = match.groups()
            new_type = "provider" if type_ in ["doctor", "nurse", "nurse practitioner", "other"] else "patient"
            result = f'<name id="{id_num}" type="{new_type}">{name}</name>'
            if result != match.group(0):
                changes[0] += 1
            return result

        new_text = re.sub(pattern, replace_tag, text, flags=re.DOTALL)
        return new_text, changes[0]

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
        
        new_text, count = re.subn(pattern, replace_tag, text, flags=re.DOTALL)
        return new_text, count

    total_changes = 0
    for index, row in df.iterrows():
        new_text, changes = process_text(row[column_name])
        df.at[index, column_name] = new_text
        total_changes += changes

    return df, total_changes


## Fixing dates
# We will find missed dates and replace them with <date>YYYY-MM-DD</date>

_MONTHS = ["January", "February", "March", "April", "June", "July",
           "August", "September", "October", "November", "December"]

def _format_dates_in_segment(text):
    """Format bare month-name date strings in a plain-text segment (no XML tags)."""
    changes = 0
    months = _MONTHS
    M = '|'.join(months)

    # "Month Day, Year"
    for date_str in re.findall(rf'\b(?:{M})\s+\d{{1,2}},\s+\d{{4}}\b', text, re.IGNORECASE):
        try:
            date_obj = datetime.strptime(date_str.strip(), '%B %d, %Y')
            text = text.replace(date_str, f'<date>{date_obj.strftime("%Y-%m-%d")}</date>')
            changes += 1
        except ValueError:
            pass

    # "Month Dayth, Year"
    for mo, day, yr in re.findall(rf'\b((?:{M}))\s+(\d{{1,2}})(?:st|nd|rd|th),\s+(\d{{4}})\b', text, re.IGNORECASE):
        try:
            date_obj = datetime.strptime(f"{mo} {day}, {yr}", '%B %d, %Y')
            text = text.replace(f"{mo} {day}th, {yr}", f'<date>{date_obj.strftime("%Y-%m-%d")}</date>')
            changes += 1
        except ValueError:
            pass

    # "Month Year"
    for date_str in re.findall(rf'\b(?:{M})\s+\d{{4}}\b', text, re.IGNORECASE):
        try:
            date_obj = datetime.strptime(date_str.strip(), '%B %Y')
            text = text.replace(date_str, f'<date>{date_obj.strftime("%Y-%m-??")}</date>')
            changes += 1
        except ValueError:
            pass

    # "Month Day"
    for date_str in re.findall(rf'\b(?:{M})\s+\d{{1,2}}\b', text, re.IGNORECASE):
        try:
            date_obj = datetime.strptime(date_str.strip(), '%B %d')
            text = text.replace(date_str, f'<date>{date_obj.strftime("????-%m-%d")}</date>')
            changes += 1
        except ValueError:
            pass

    # "Month Dayth"
    for mo, day in re.findall(rf'\b((?:{M}))\s+(\d{{1,2}})(?:st|nd|rd|th)\b', text, re.IGNORECASE):
        try:
            date_obj = datetime.strptime(f"{mo} {day}", '%B %d')
            text = text.replace(f"{mo} {day}th", f'<date>????-{date_obj.strftime("%m-%d")}</date>')
            changes += 1
        except ValueError:
            pass

    # Standalone month names (fallback)
    standalone = rf'(?<!^)(?<![\.\!\?\n])(?<![\.\!\?\n]\s)(?<![\.\!\?\n]\s\s)\b((?:{M}))\b(?![a-zA-Z])'
    for match in re.findall(standalone, text):
        month = match[0] if isinstance(match, tuple) else match
        date_obj = datetime.strptime(month.strip(), '%B')
        text = text.replace(month, f'<date>????-{date_obj.strftime("%m")}-??</date>')
        changes += 1

    return text, changes


def extract_and_format_dates(text):
    return _apply_to_text_segments(_format_dates_in_segment, text, skip_element_content=True)

## Removing Markdowns
# We will remove the markdown tags from the text.

# _XML_TAG and _apply_to_text_segments are defined at the top of this module.
_MD_PATTERNS = [
    (re.compile(r'\*\*(.+?)\*\*', re.DOTALL), r'\1'),
    (re.compile(r'__(.+?)__',     re.DOTALL), r'\1'),
    (re.compile(r'\*(.+?)\*',     re.DOTALL), r'\1'),
    (re.compile(r'_(.+?)_',       re.DOTALL), r'\1'),
    (re.compile(r'\[([^\]]+)\]\([^)]+\)'),    r'\1'),
]

def remove_markdown(df, column_name='default_response_1'):
    """Remove markdown formatting from plain-text segments, skipping XML tags."""
    def process_single_text(text):
        # Split into alternating [text, tag, text, tag, ...] segments.
        # Only process the text (non-tag) segments.
        parts = _XML_TAG.split(text)
        tags  = _XML_TAG.findall(text)
        tokens = 0
        new_parts = []
        for part in parts:
            for pattern, repl in _MD_PATTERNS:
                part, n = pattern.subn(repl, part)
                tokens += n
            new_parts.append(part)
        # Re-interleave text and tag segments.
        result = new_parts[0]
        for tag, part in zip(tags, new_parts[1:]):
            result += tag + part
        return result, tokens

    total_removed = 0
    for index, row in df.iterrows():
        new_text, count = process_single_text(row[column_name])
        df.at[index, column_name] = new_text
        total_removed += count

    return df, total_removed


_ENTITY_TAG_PATTERNS: dict[str, re.Pattern] = {
    "name":    re.compile(r"<name[\s>]"),
    "age":     re.compile(r"<age>"),
    "date":    re.compile(r"<date>"),
    "place":   re.compile(r"<place>"),
    "address": re.compile(r"<address>"),
    "phone":   re.compile(r"<phone>"),
    "email":   re.compile(r"<email>"),
    "id":      re.compile(r"<id[\s>]"),
    "url":     re.compile(r"<url>"),
    "device":  re.compile(r"<device>"),
}

def _count_entity_tags(df: pd.DataFrame, column: str) -> dict[str, int]:
    """Count individual entity tags of each PHI type across all rows."""
    return {
        entity: int(df[column].apply(lambda x: len(pat.findall(x))).sum())
        for entity, pat in _ENTITY_TAG_PATTERNS.items()
    }


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
    
    # Per-entity reformatting totals (keyed to the same entity names as tag counts)
    reformatted: dict[str, int] = {
        "name":  name_tag_changes + name_type_changes,
        "age":   age_changes,
        "date":  date_changes + remaining_date_changes,
        "phone": phone_changes,
        "id":    id_tag_changes,
    }
    other_changes = {"gender terms corrected": gender_changes,
                     "markdown tokens removed": markdown_changes}

    entity_counts = _count_entity_tags(df, output_col)

    print("\nProcessing Summary:")
    print(f"  {'Entity':<10}  {'Detected':>9}  {'Reformatted':>11}")
    print(f"  {'-'*10}  {'-'*9}  {'-'*11}")
    for entity, detected in entity_counts.items():
        ref = reformatted.get(entity, 0)
        ref_str = str(ref) if ref else "-"
        print(f"  {entity:<10}  {detected:>9}  {ref_str:>11}")

    extras = [(label, n) for label, n in other_changes.items() if n]
    if extras:
        print()
        for label, n in extras:
            print(f"  {label}: {n}")

    print("\nText processing pipeline completed!")
    return df


def inspect_changes(
    before: pd.Series,
    after: pd.Series,
    entity: str | None = None,
    n: int = 5,
    context: int = 2,
) -> None:
    """Show unified diffs for rows where preprocessing made changes.

    Usage
    -----
    Snapshot the column *before* calling preprocess_text, then call this
    function afterwards:

        before_col = df['anonymized'].copy()
        df = preprocess_text(df, input_col='original', output_col='anonymized')
        inspect_changes(before_col, df['anonymized'], entity='date', n=3)

    Parameters
    ----------
    before  : Series of column values before preprocessing.
    after   : Series of column values after preprocessing.
    entity  : Optional PHI type to filter on — only shows rows where the
              tag count for that entity changed.
              One of: name, age, date, place, address, phone, email, id, url, device.
    n       : Max number of samples to display (default 5).
    context : Diff context lines shown around each change (default 2).
    """
    # Align by position — preprocess_text may drop rows and reset the index,
    # so the two series might not share the same labels.
    before = before.reset_index(drop=True)
    after  = after.reset_index(drop=True)
    # Truncate to the shorter series (rows dropped by gibberish filter won't appear in after)
    min_len = min(len(before), len(after))
    before  = before.iloc[:min_len]
    after   = after.iloc[:min_len]

    changed = before != after

    if entity is not None:
        pat = _ENTITY_TAG_PATTERNS.get(entity)
        if pat is None:
            raise ValueError(
                f"Unknown entity {entity!r}. "
                f"Valid options: {sorted(_ENTITY_TAG_PATTERNS)}"
            )
        b_cnt = before.apply(lambda x: len(pat.findall(str(x))))
        a_cnt = after.apply(lambda x: len(pat.findall(str(x))))
        changed = changed & (b_cnt != a_cnt)

    indices = changed[changed].index.tolist()
    total   = len(indices)
    label   = f" for entity='{entity}'" if entity else ""
    print(f"{total} row(s) changed{label}.")

    for rank, idx in enumerate(indices[:n], 1):
        b_lines = str(before.loc[idx]).splitlines(keepends=True)
        a_lines = str(after.loc[idx]).splitlines(keepends=True)
        diff    = list(difflib.unified_diff(
            b_lines, a_lines,
            fromfile="before", tofile="after",
            n=context, lineterm="",
        ))
        print(f"\n{'─' * 64}")
        print(f"  [{rank}/{min(n, total)}]  row index {idx}")
        if diff:
            # Skip the @@-header lines; just print changed lines with context
            for line in diff[2:]:
                print(line)
        else:
            print("  (content changed but no line-level diff found)")

    if total > n:
        print(f"\n  … {total - n} more row(s) not shown — increase n= to see them.")
