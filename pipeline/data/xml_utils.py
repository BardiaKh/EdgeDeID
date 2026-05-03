import re
import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import Dict, List, Tuple

def convert_to_xml(text, sort_by_position=False):
    tags = ['name', 'age', 'date', 'place', 'address', 'phone', 'email', 'id', 'url', 'device']
    parent_tags = ['names', 'ages', 'dates', 'places', 'addresses', 'phones', 'emails', 'ids', 'urls', 'devices']
    
    # Create mapping from singular to plural tags
    tag_to_parent = dict(zip(tags, parent_tags))
    
    def extract_attributes(tag_string: str) -> Dict[str, str]:
        return dict(re.findall(r'(\w+)="([^"]*)"', tag_string))

    # Find all tags with their positions and create unique identifiers
    tag_positions: List[Tuple[int, str, Dict[str, str], str]] = []  # [(position, tag, attributes, content)]
    
    for tag in tags:
        pattern = rf'<{tag}((?:\s+\w+="[^"]*")*)\s*>(.*?)</{tag}>'
        for match in re.finditer(pattern, text):
            position = match.start()
            attr_string, content = match.groups()
            attrs = extract_attributes(attr_string)
            
            # Create unique identifier based on tag type
            unique_id = None
            if tag == 'name':
                unique_id = f"name_{attrs.get('id')}_{attrs.get('type')}"
            elif tag == 'date':
                unique_id = f"date_{content}_{attrs.get('formatted', '')}"
            elif tag in ['place', 'id']:
                unique_id = f"{tag}_{attrs.get('id') or attrs.get('type')}"
            else:
                unique_id = f"{tag}_{content}"
                
            tag_positions.append((position, tag, attrs, content, unique_id))
    
    # Sort by position to preserve original order
    if sort_by_position:
        tag_positions.sort(key=lambda x: x[0])
    
    # Create the root element for the XML
    root = ET.Element("phi_entities")
    
    # Create parent elements for all tag types (empty by default)
    parent_elements = {}
    for parent_tag in parent_tags:
        parent_elements[parent_tag] = ET.SubElement(root, parent_tag)
    
    # Dictionary to store unique entities
    unique_entities: Dict[str, ET.Element] = {}
    
    # Process tags in order of appearance
    for _, tag, attrs, content, unique_id in tag_positions:
        parent_tag = tag_to_parent[tag]
        parent_element = parent_elements[parent_tag]
        
        if unique_id not in unique_entities:
            element = ET.SubElement(parent_element, tag)
            for k, v in attrs.items():
                if k != 'formatted':
                    element.set(k, v)

            if tag == 'date':
                original = ET.SubElement(element, "original")
                original.text = content
                formatted = ET.SubElement(element, "formatted")
                formatted.text = attrs.get('formatted', content)
            elif tag == 'name':
                variations = ET.SubElement(element, "variations")
                value = ET.SubElement(variations, "value")
                value.text = content
            else:
                element.text = content

            unique_entities[unique_id] = element
        elif tag == 'name':
            # Add variation for existing name
            variations = unique_entities[unique_id].find('variations')
            value = ET.SubElement(variations, "value")
            value.text = content

    # Remove duplicate variations
    for element in root.findall('.//variations'):
        unique_values = list(dict.fromkeys(child.text for child in element))
        element.clear()
        for unique_value in unique_values:
            value = ET.SubElement(element, "value")
            value.text = unique_value
            
    # Convert the XML to a pretty-printed string
    xml_string = minidom.parseString(ET.tostring(root)).toprettyxml(indent="")

    # Remove the XML declaration line
    xml_string = '\n'.join(xml_string.split('\n')[1:])
    
    return xml_string.strip()

def get_entity_positions(text, xml_string):
    """
    Extract entity positions from text based on XML output.
    Returns list of dictionaries with start, end, type, and metadata.
    """
    tags = ['name', 'age', 'date', 'place', 'address', 'phone', 'email', 'id', 'url', 'device']
    parent_tags = ['names', 'ages', 'dates', 'places', 'addresses', 'phones', 'emails', 'ids', 'urls', 'devices']
    
    # Create mapping from singular to plural tags
    tag_to_parent = dict(zip(tags, parent_tags))
    
    def extract_attributes(tag_string: str) -> Dict[str, str]:
        return dict(re.findall(r'(\w+)="([^"]*)"', tag_string))

    try:
        root = ET.fromstring(xml_string)
    except ET.ParseError:
        return []
    
    positions = []
    
    # Process each parent tag type
    for parent_tag in parent_tags:
        parent_element = root.find(parent_tag)
        if parent_element is not None:
            for tag in parent_element.findall('*'):
                tag_name = tag.tag
                attrs = dict(tag.attrib)
                
                # Extract content based on tag type
                content = None
                if tag_name == 'name':
                    variations = tag.find('variations')
                    if variations is not None:
                        for value_elem in variations.findall('value'):
                            if value_elem.text:
                                content = value_elem.text
                                break
                elif tag_name == 'date':
                    original = tag.find('original')
                    if original is not None and original.text:
                        content = original.text
                else:
                    content = tag.text
                
                if content:
                    # Find all occurrences of this content in the original text
                    pattern = re.escape(content)
                    for match in re.finditer(pattern, text):
                        positions.append({
                            'start': match.start(),
                            'end': match.end(),
                            'type': tag_name,
                            'metadata': attrs,
                            'content': content
                        })
    
    # Sort by start position, then by length (longest first) to handle overlaps
    positions.sort(key=lambda x: (x['start'], -(x['end'] - x['start'])))
    
    # Remove overlapping positions (keep the first/longest one for each position)
    filtered_positions = []
    for pos in positions:
        overlaps = False
        for accepted in filtered_positions:
            if not (pos['end'] <= accepted['start'] or pos['start'] >= accepted['end']):
                overlaps = True
                break
        
        if not overlaps:
            filtered_positions.append(pos)
    
    return filtered_positions

def mask_text(text, positions, mask_char='*'):
    """
    Replace specified positions in text with mask character.
    Returns the masked text.
    """
    if not positions:
        return text
    
    # Sort positions by start position (reverse order for replacement)
    sorted_positions = sorted(positions, key=lambda x: x['start'], reverse=True)
    
    masked_text = text
    for pos in sorted_positions:
        start = pos['start']
        end = pos['end']
        replacement = mask_char * (end - start)
        masked_text = masked_text[:start] + replacement + masked_text[end:]
    
    return masked_text