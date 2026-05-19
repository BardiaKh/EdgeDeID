"""
Utility functions for the pseudolabeling pipeline:
data loading, deanonymization, model inference, XML merging, and text annotation.
"""

import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from glob import glob
from typing import Dict, List

import Levenshtein
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

sys.path.insert(0, "..")
from pipeline.prompts import SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(name: str) -> pd.DataFrame:
    """Load and normalize a generated dataset by name ('hermes', 'haiku', or 'deepseek')."""
    if name == "hermes":
        frames = []
        for path in glob("../data/hermes/output_*_hermes.csv"):
            df = pd.read_csv(path)
            df.drop(index=df[df["features_to_include"].isna()].index, inplace=True)
            df.drop(index=df[df["default_response_1"].isna()].index, inplace=True)
            df["model"] = "Hermes"
            frames.append(df)
        out = pd.concat(frames)
        out.drop(columns=["index"], inplace=True)
        out.reset_index(inplace=True, drop=True)
        faulty_indices = [18765, 19717, 21618, 21362, 21524, 21507, 22274, 22506]
        out.drop(index=faulty_indices, inplace=True)
        out.reset_index(inplace=True, drop=True)
        return out[["default_response_0", "default_response_1", "report_length", "features_to_include", "model"]]

    if name == "haiku":
        frames = []
        for path in glob("../data/haiku/output_*_haiku.csv"):
            df = pd.read_csv(path)
            df.rename(columns={"raw_report": "default_response_0", "anonymized_report": "default_response_1"}, inplace=True)
            df["model"] = "Haiku"
            frames.append(df)
        out = pd.concat(frames)
        out["report_length"] = out["criterion"].apply(lambda x: eval(x)["report_length"])
        out["features_to_include"] = out["criterion"].apply(lambda x: eval(x)["features_to_include"])
        out.drop(columns=["criterion"], inplace=True)
        out.reset_index(inplace=True, drop=True)
        return out[["default_response_0", "default_response_1", "report_length", "features_to_include", "model"]]

    if name == "deepseek":
        frames = []
        for path in glob("../data/deepseek/output_*_deepseek.csv"):
            df = pd.read_csv(path)
            df.rename(columns={"raw_report": "default_response_0", "anonymized_report": "default_response_1"}, inplace=True)
            df["model"] = "DeepSeek"
            frames.append(df)
        out = pd.concat(frames)
        out["report_length"] = out["criterion"].apply(lambda x: eval(x)["report_length"])
        out["features_to_include"] = out["criterion"].apply(lambda x: eval(x)["features_to_include"])
        out.drop(columns=["criterion"], inplace=True)
        out.reset_index(inplace=True, drop=True)
        return out[["default_response_0", "default_response_1", "report_length", "features_to_include", "model"]]

    raise ValueError(f"Unknown dataset {name!r}. Choose 'hermes', 'haiku', or 'deepseek'.")


def clean_responses(df: pd.DataFrame) -> pd.DataFrame:
    """Strip XML wrapper tags from both response columns and drop null/empty rows."""
    for col in ["default_response_1", "default_response_0"]:
        df[col] = (
            df[col]
            .str.replace("<anonymized_report>", "", regex=False)
            .str.replace("</anonymized_report>", "", regex=False)
            .str.strip()
            .str.replace("<anonymized_report/>", "", regex=False)
            .str.replace("<raw_report>", "", regex=False)
            .str.replace("</raw_report>", "", regex=False)
            .str.strip()
            .str.replace("<raw_report/>", "", regex=False)
        )
    for col in ["default_response_0", "default_response_1"]:
        df.drop(index=df[df[col].isna()].index, inplace=True)
        df.drop(index=df[df[col] == ""].index, inplace=True)
    df.reset_index(inplace=True, drop=True)
    return df


# ---------------------------------------------------------------------------
# Deanonymization
# ---------------------------------------------------------------------------

def remove_anonymization_tags(text: str) -> str:
    """Strip PHI XML tags, keeping inner content."""
    tags = ["name", "age", "date", "place", "address", "phone", "email", "id", "url", "device"]
    for tag in tags:
        text = re.sub(rf"<{tag}(?:\s+[^>]*)?>(.*?)</{tag}>", r"\1", text)
    return text


def handle_formatted_prop(text: str) -> str:
    """Replace tag content with the `formatted` attribute value when present."""
    tags = ["name", "age", "date", "place", "address", "phone", "email", "id", "url", "device"]

    def process_tag(tag, text):
        pattern = rf'<{tag}(\s+[^>]*?)?(formatted="([^"]*)")?(.*?)>(.*?)</{tag}>'

        def replace(match):
            prefix = match.group(1) or ""
            formatted_value = match.group(3)
            other_attrs = match.group(4) or ""
            content = match.group(5)
            new_content = formatted_value if formatted_value else content
            attrs = (prefix + " " + other_attrs).strip()
            return f'<{tag}{" " + attrs if attrs else ""}>{new_content}</{tag}>'

        return re.sub(pattern, replace, text)

    for tag in tags:
        text = process_tag(tag, text)
    return text


def deanonymize_dataframe_v1(
    df: pd.DataFrame, column_name: str, new_column_name: str
) -> pd.DataFrame:
    """Add a deanonymized column and apply formatted-prop handling in-place."""
    df[new_column_name] = df[column_name].apply(remove_anonymization_tags)
    df[column_name] = df[column_name].apply(handle_formatted_prop)
    return df


def deanonymize_dataframe_v2(
    df: pd.DataFrame, column_name: str, xml_column_name: str, native_column_name: str
) -> pd.DataFrame:
    """Add XML-converted and plain-text columns derived from an annotated column."""
    from pipeline.data.xml_utils import convert_to_xml
    df[xml_column_name] = df[column_name].apply(convert_to_xml)
    df[native_column_name] = df[column_name].apply(remove_anonymization_tags)
    return df


# ---------------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------------

def run_model(note: str, client, model: str) -> str:
    """Run the pseudo-label model on a single clinical note and return annotated XML."""
    input_text = (
        f"<instructions>\n{SYSTEM_PROMPT}\n</instructions>\n\n"
        f"<note>\n{note.strip()}\n</note>\n\n"
        "<phi_entities>\n"
    )
    response = client.completions.create(
        model=model,
        prompt=input_text,
        max_tokens=2048,
        temperature=0,
        seed=1995,
        stop=["</phi_entities>"],
    )
    response_text = "<phi_entities>\n" + response.choices[0].text
    return response_text.split("</phi_entities>", 1)[0] + "</phi_entities>"


# ---------------------------------------------------------------------------
# XML merging utilities
# ---------------------------------------------------------------------------

def clean_xml(xml_string: str) -> str:
    """Normalize content between <phi_entities> tags."""
    start_tag, end_tag = "<phi_entities>", "</phi_entities>"
    content = xml_string[xml_string.find(start_tag) + len(start_tag):xml_string.find(end_tag)].strip()
    return f"<phi_entities>\n{content}\n</phi_entities>" if content else create_empty_xml_with_parents()


def create_empty_xml_with_parents() -> str:
    """Return a <phi_entities> skeleton with all parent tags as self-closing."""
    parent_tags = ["names", "ages", "dates", "places", "addresses", "phones", "emails", "ids", "urls", "devices"]
    return "<phi_entities>\n" + "\n".join(f"<{t}/>" for t in parent_tags) + "\n</phi_entities>"


@dataclass(frozen=True)
class EntityMention:
    text: str
    start: int
    end: int
    tag: str
    attributes: tuple


def _extract_mentions_from_tag(tag, text: str) -> List[EntityMention]:
    """Extract EntityMention objects from a single BeautifulSoup tag."""
    mentions = []
    if tag.name == "name":
        variations = tag.find("variations")
        if variations:
            for val in variations.find_all("value"):
                content = val.text.strip() if val.text else ""
                if content:
                    for match in re.finditer(re.escape(content), text, re.IGNORECASE):
                        mentions.append(EntityMention(
                            text=content, start=match.start(), end=match.end(),
                            tag="name", attributes=tuple(sorted(tag.attrs.items())),
                        ))
    elif tag.name == "date":
        original = tag.find("original")
        if original and original.text:
            content = original.text.strip()
            if content:
                for match in re.finditer(re.escape(content), text, re.IGNORECASE):
                    formatted = tag.find("formatted")
                    attrs = (("formatted", formatted.text.strip()),) if formatted and formatted.text else ()
                    mentions.append(EntityMention(
                        text=content, start=match.start(), end=match.end(),
                        tag="date", attributes=attrs,
                    ))
    else:
        content = tag.text.strip() if tag.text else ""
        if content:
            for match in re.finditer(re.escape(content), text, re.IGNORECASE):
                mentions.append(EntityMention(
                    text=content, start=match.start(), end=match.end(),
                    tag=tag.name, attributes=tuple(sorted(tag.attrs.items())),
                ))
    return mentions


def extract_mentions(text: str, xml: str) -> List[EntityMention]:
    """Extract all entity mentions with text positions from an annotated XML."""
    mentions = []
    try:
        soup = BeautifulSoup(xml, "xml").find("phi_entities")
        if not soup:
            return mentions
        parent_tags = ["names", "ages", "dates", "places", "addresses", "phones", "emails", "ids", "urls", "devices"]
        for pt in parent_tags:
            parent_el = soup.find(pt)
            if parent_el:
                for tag in parent_el.find_all(recursive=False):
                    mentions.extend(_extract_mentions_from_tag(tag, text))
    except Exception as e:
        print(f"Error extracting mentions: {e}")
        return mentions

    mentions.sort(key=lambda x: (-len(x.text), x.start))
    final: List[EntityMention] = []
    for mention in mentions:
        if not any(
            not (mention.end <= ex.start or mention.start >= ex.end)
            for ex in final
        ):
            final.append(mention)
    return final


def resolve_overlapping_mentions(mentions: List[EntityMention]) -> List[EntityMention]:
    """Resolve overlapping mentions using tag-priority rules."""
    tag_priority = {
        "name": 5, "date": 4, "id": 3, "phone": 3, "email": 3,
        "device": 2, "address": 2, "place": 2, "url": 1, "age": 1,
    }
    mentions.sort(key=lambda x: (-len(x.text), x.start))
    final: List[EntityMention] = []
    for mention in mentions:
        for existing in final:
            if not (mention.end <= existing.start or mention.start >= existing.end):
                if tag_priority.get(mention.tag, 0) > tag_priority.get(existing.tag, 0):
                    final.remove(existing)
                    final.append(mention)
                break
        else:
            final.append(mention)
    return final


def generate_xml_with_parent_structure(
    final_mentions: List[EntityMention], mention_groups: Dict
) -> str:
    """Render a merged set of entity mentions as a <phi_entities> XML string."""
    root = BeautifulSoup("<phi_entities></phi_entities>", "xml")
    parent_tags = ["names", "ages", "dates", "places", "addresses", "phones", "emails", "ids", "urls", "devices"]
    tags = ["name", "age", "date", "place", "address", "phone", "email", "id", "url", "device"]
    tag_to_parent = dict(zip(tags, parent_tags))

    parent_elements = {}
    for pt in parent_tags:
        parent_elements[pt] = root.new_tag(pt)
        root.phi_entities.append(parent_elements[pt])

    final_mentions.sort(key=lambda x: (x.tag, x.start))

    name_mentions: Dict = defaultdict(set)
    non_name_mentions = []
    for mention in final_mentions:
        if mention.tag == "name":
            name_mentions[mention.attributes].add(mention)
        else:
            non_name_mentions.append(mention)

    for attrs, mentions in name_mentions.items():
        tag = root.new_tag("name")
        for key, value in attrs:
            tag[key] = value
        variations = root.new_tag("variations")
        seen: set = set()
        for mention in mentions:
            for m in mention_groups[mention]:
                if m.text.lower() not in seen:
                    v = root.new_tag("value")
                    v.string = m.text
                    variations.append(v)
                    seen.add(m.text.lower())
        tag.append(variations)
        parent_elements["names"].append(tag)

    for mention in non_name_mentions:
        tag = root.new_tag(mention.tag)
        parent_tag = tag_to_parent.get(mention.tag, "names")
        for key, value in mention.attributes:
            if key == "formatted":
                continue
            tag[key] = value
        if mention.tag == "date":
            original = root.new_tag("original")
            original.string = mention.text
            formatted = root.new_tag("formatted")
            formatted.string = dict(mention.attributes).get("formatted", mention.text)
            tag.append(original)
            tag.append(formatted)
        else:
            tag.string = mention.text
        parent_elements[parent_tag].append(tag)

    return str(root)


def _validate_tag_in_text(tag, text: str) -> bool:
    """Return True if the tag's content appears in text."""
    if tag.name == "name":
        variations = tag.find("variations")
        if variations:
            return any(
                val.text and val.text.lower() in text.lower()
                for val in variations.find_all("value")
            )
    elif tag.name == "date":
        original = tag.find("original")
        if original and original.text:
            return original.text.lower() in text.lower()
    elif tag.text:
        return tag.text.lower() in text.lower()
    return True


def validate_merged_xml(text: str, merged_xml: str) -> bool:
    """Return True if every entity in merged_xml appears in text."""
    try:
        soup = BeautifulSoup(merged_xml, "xml")
        parent_tags = ["names", "ages", "dates", "places", "addresses", "phones", "emails", "ids", "urls", "devices"]
        for pt in parent_tags:
            parent_el = soup.find(pt)
            if parent_el:
                for tag in parent_el.find_all(recursive=False):
                    if not _validate_tag_in_text(tag, text):
                        return False
        return True
    except Exception:
        return False


def merge_xmls(text: str, gt_xml: str, llm_xml: str) -> str:
    """Merge ground-truth and LLM-generated XML annotations for a single text."""
    gt_xml = clean_xml(gt_xml)
    llm_xml = clean_xml(llm_xml)

    all_mentions = extract_mentions(text, gt_xml) + extract_mentions(text, llm_xml)

    mention_groups: Dict = defaultdict(list)
    for mention in all_mentions:
        added = False
        for key in mention_groups:
            if mention.tag == key.tag and Levenshtein.ratio(mention.text.lower(), key.text.lower()) > 0.8:
                mention_groups[key].append(mention)
                added = True
                break
        if not added:
            mention_groups[mention].append(mention)

    final_mentions = resolve_overlapping_mentions(list(mention_groups.keys()))
    return generate_xml_with_parent_structure(final_mentions, mention_groups)


def handle_xml_merging(df: pd.DataFrame) -> List[str]:
    """Merge xml_haiku and xml_pseudolabel for every row; fall back to gt on failure."""
    merged_xmls = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        merged = merge_xmls(row["native_text"], row["xml_haiku"], row["xml_pseudolabel"])
        if not validate_merged_xml(row["native_text"], merged):
            print(f"Warning: Invalid merged XML for text: {row['native_text'][:100]}...")
            merged = clean_xml(row["xml_haiku"])
        merged_xmls.append(merged)
    return merged_xmls


# ---------------------------------------------------------------------------
# Text annotation
# ---------------------------------------------------------------------------

def simple_text_anonymization(native_text: str, xml_string: str) -> str:
    """Replace PHI entities in native_text with placeholder tokens."""
    soup = BeautifulSoup(xml_string, "xml").find("phi_entities")
    to_be_replaced = {}
    parent_tags = ["names", "ages", "dates", "places", "addresses", "phones", "emails", "ids", "urls", "devices"]
    for pt in parent_tags:
        parent_el = soup.find(pt)
        if parent_el:
            for tag in parent_el.find_all(recursive=False):
                if tag.name == "name":
                    for val in tag.find("variations").find_all("value"):
                        to_be_replaced[val.text] = "***Name***"
                elif tag.name == "date":
                    to_be_replaced[tag.find("original").text] = "***Date***"
                else:
                    to_be_replaced[tag.text] = f"***{tag.name}***"
    for key, value in sorted(to_be_replaced.items(), key=lambda x: len(x[0]), reverse=True):
        native_text = native_text.replace(key, value)
    return native_text


def verify_name_variations(native_text: str, xml_string: str) -> str:
    """Expand name variation lists to include individual words found in the anonymized text."""
    soup = BeautifulSoup(xml_string, "xml")
    native_text_anon = simple_text_anonymization(native_text, xml_string)
    for name in soup.find_all("name"):
        variation_values = [val.text for val in name.find("variations").find_all("value")]
        for variation in variation_values:
            words = variation.split(" ")
            if len(words) > 1:
                for word in words:
                    if word in native_text_anon and word not in variation_values:
                        name.find("variations").append(BeautifulSoup(f"<value>{word}</value>", "xml"))
                    if word.lower() in native_text_anon:
                        name.find("variations").append(BeautifulSoup(f"<value>{word.lower()}</value>", "xml"))
    return str(soup)


def annotate_text_with_xml(native_text: str, xml_string: str) -> str:
    """Re-insert XML tags into native_text at the positions of matched entities."""
    parent_tags = ["names", "ages", "dates", "places", "addresses", "phones", "emails", "ids", "urls", "devices"]
    soup = BeautifulSoup(xml_string, "lxml-xml")

    for tag in soup.find_all():
        if tag.name not in ["name", "date"]:
            if tag.find_parent(tag.name):
                tag.decompose()
    for tag in soup.find_all():
        if not tag.text.strip():
            tag.decompose()

    replacements = []
    for tag in soup.find_all():
        if tag.name in ["phi_entities", "variations", "value", "original", "formatted"] + parent_tags:
            continue

        all_variations = []
        formatted_content = None
        if tag.name == "name":
            for variation in tag.find("variations").find_all("value"):
                all_variations.append(variation.text.strip())
        elif tag.name == "date":
            all_variations.append(tag.find("original").text.strip())
            formatted_content = tag.find("formatted").text.strip()
        else:
            all_variations.append(tag.text.strip())

        all_variations.sort(key=len, reverse=True)

        for content in all_variations:
            for match in re.finditer(re.escape(content), native_text, re.IGNORECASE):
                opening_tag = f"<{tag.name}" + "".join(f' {k}="{v}"' for k, v in tag.attrs.items()) + ">"
                replacement = (
                    f"{opening_tag}{formatted_content}</{tag.name}>"
                    if formatted_content is not None
                    else f"{opening_tag}{match.group()}</{tag.name}>"
                )
                replacements.append((match.start(), match.end(), replacement))

    non_overlapping = []
    for start, end, replacement in sorted(replacements, key=lambda x: (x[1] - x[0]), reverse=True):
        if not non_overlapping:
            non_overlapping.append((start, end, replacement))
            continue
        for i in range(len(non_overlapping)):
            if non_overlapping[i][0] <= start and non_overlapping[i][1] >= end:
                break
        else:
            non_overlapping.append((start, end, replacement))

    for start, end, replacement in sorted(non_overlapping, key=lambda x: x[0], reverse=True):
        native_text = native_text[:start] + replacement + native_text[end:]

    return native_text
