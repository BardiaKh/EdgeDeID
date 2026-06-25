#!/usr/bin/env python3
"""
EdgeDeID CLI: Medical Note Anonymization using llama.cpp
"""

import subprocess
import json
import re
import time
import argparse
import sys
from pathlib import Path
from bs4 import BeautifulSoup
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich.syntax import Syntax
from rich.table import Table

# Add the repo root to the path so `pipeline` is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from edgedeid.data.xml_utils import get_entity_positions

# Initialize rich console
console = Console()

# Load system prompt
def load_system_prompt():
    """Load the system prompt from prompts.txt"""
    prompts_file = Path(__file__).parent / "prompts.txt"
    if not prompts_file.exists():
        console.print("[red]Error: prompts.txt not found![/red]")
        sys.exit(1)
    
    with open(prompts_file, 'r') as f:
        return f.read().strip()

SYSTEM_PROMPT = load_system_prompt()

def run_llama_model(note, model_path, use_gpu=False, temperature=0):
    """
    Run llama.cpp model with the given note and return the PHI entities
    """
    global SYSTEM_PROMPT
    
    # Prepare the full prompt (no examples for simplified interface)
    input_text = f"""<instructions>
{SYSTEM_PROMPT}
</instructions>

<note>
{note.strip()}
</note>

<phi_entities>
"""
    # Prepare llama.cpp command
    llama_cli_path = Path(__file__).parent / "llama.cpp" / "bin" / "llama-cli"
    if not llama_cli_path.exists():
        console.print(f"[red]Error: llama-cli not found at {llama_cli_path}[/red]")
        sys.exit(1)

    cmd = [
        str(llama_cli_path),
        "-m", model_path,
        "-p", input_text,
        "-n", "2048",
        "-no-cnv",
        "--temp", str(temperature),
        "--seed", "2025"
    ]
    
    # Add GPU acceleration if requested
    if use_gpu:
        cmd.extend(["-ngl", "99"])  # Offload all layers to GPU
    else:
        cmd.extend(["-ngl", "0"])   # Force CPU only

    try:
        # Run the command with progress indicator
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task("Processing with LLaMA model...", total=None)
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
        
        if result.returncode != 0:
            console.print(f"[red]Error running llama-cli: {result.stderr}[/red]")
            return None
            
        # Extract the response and format it
        response_text = result.stdout.strip()[len(input_text):]
        
        # Add the opening tag and extract until closing tag
        full_response = "<phi_entities>\n" + response_text
        if "</phi_entities>" not in full_response:
            full_response += "\n</phi_entities>"
        else:
            full_response = full_response.split("</phi_entities>", 1)[0] + "</phi_entities>"
            
        return full_response
        
    except subprocess.TimeoutExpired:
        console.print("[red]Error: Model inference timed out[/red]")
        return None
    except Exception as e:
        console.print(f"[red]Error running model: {str(e)}[/red]")
        return None

def simple_text_anonymization(native_text, xml_string):
    """
    Anonymize text based on PHI entities XML using new positioning functions
    Returns tuple: (anonymized_text_with_entities, entity_mapping)
    """
    try:
        positions = get_entity_positions(native_text, xml_string)
    except Exception as e:
        console.print(f"[yellow]Warning: Could not parse XML: {e}[/yellow]")
        return [(native_text, "")], {}
    
    if not positions:
        return [(native_text, "")], {}
    
    # Create replacement mapping
    to_be_replaced = {}
    annotated_entities = {}
    
    for pos in positions:
        content = pos['content']
        tag_type = pos['type']
        metadata = pos['metadata']
        
        # Create replacement value based on tag type and metadata
        if tag_type == 'name':
            id_val = metadata.get('id', '1')
            replacement = f"***NAME-{id_val}***"
        elif tag_type == 'date':
            formatted = metadata.get('formatted', content)
            replacement = f"***{formatted}***"
        else:
            id_val = metadata.get('id')
            if id_val:
                replacement = f"***{tag_type.upper()}-{id_val}***"
            else:
                replacement = f"***{tag_type.upper()}***"
        
        to_be_replaced[content] = {
            'replacement': replacement,
            'type': tag_type
        }
        
        # Track entity mapping
        if replacement not in annotated_entities:
            annotated_entities[replacement] = [content]
        else:
            annotated_entities[replacement].append(content)
    
    # Find all matches in the original text
    all_matches = []
    for original_text, replacement_info in to_be_replaced.items():
        pattern = re.escape(original_text)
        for match in re.finditer(f'(?<!\\*){pattern}(?!\\*)', native_text):
            all_matches.append({
                'start': match.start(),
                'end': match.end(),
                'original': original_text,
                'replacement': replacement_info['replacement'],
                'type': replacement_info['type'],
                'length': len(original_text)
            })
    
    # Sort matches by start position, then by length (longest first) to handle overlaps
    all_matches.sort(key=lambda x: (x['start'], -x['length']))
    
    # Remove overlapping matches (keep the first/longest one for each position)
    filtered_matches = []
    for match in all_matches:
        # Check if this match overlaps with any previously accepted match
        overlaps = False
        for accepted in filtered_matches:
            if not (match['end'] <= accepted['start'] or match['start'] >= accepted['end']):
                overlaps = True
                break
        
        if not overlaps:
            filtered_matches.append(match)
    
    # Sort filtered matches by start position for processing
    filtered_matches.sort(key=lambda x: x['start'])
    
    # Build the result by replacing from end to start (to maintain positions)
    working_text = native_text
    highlighted_segments = []
    
    # Process replacements from end to start
    for match in reversed(filtered_matches):
        start, end = match['start'], match['end']
        replacement = match['replacement']
        working_text = working_text[:start] + replacement + working_text[end:]
    
    # Now build highlighted segments by going through the final text
    last_pos = 0
    final_matches = []
    
    # Calculate new positions after all replacements
    offset = 0
    for match in filtered_matches:
        new_start = match['start'] + offset
        new_end = new_start + len(match['replacement'])
        final_matches.append({
            'start': new_start,
            'end': new_end,
            'replacement': match['replacement'],
            'type': match['type']
        })
        # Update offset for next match
        offset += len(match['replacement']) - (match['end'] - match['start'])
    
    # Build highlighted segments
    for match in final_matches:
        # Add text before this match
        if last_pos < match['start']:
            highlighted_segments.append((working_text[last_pos:match['start']], ""))
        
        # Add the highlighted replacement
        highlighted_segments.append((match['replacement'], match['type']))
        last_pos = match['end']
    
    # Add remaining text
    if last_pos < len(working_text):
        highlighted_segments.append((working_text[last_pos:], ""))
    
    return highlighted_segments, annotated_entities


def display_highlighted_text(segments):
    """Display text with highlighted PHI entities using rich"""
    text = Text()
    
    for segment, entity_type in segments:
        if entity_type:
            # Different colors for different entity types - optimized for visibility on white backgrounds
            colors = {
                'name': 'white on red',
                'date': 'white on blue', 
                'place': 'white on green',
                'age': 'black on yellow',
                'phone': 'white on magenta',
                'email': 'white on cyan',
                'address': 'white on bright_red',
                'id': 'white on bright_blue',
                'url': 'white on bright_green',
                'device': 'black on bright_yellow'
            }
            style = colors.get(entity_type, 'white on black')
            text.append(segment, style=f"bold {style}")
        else:
            text.append(segment)
    
    return text

def sanitize_note(note, model_path, use_gpu=False):
    """Main function to sanitize a medical note"""
    note = note.strip()
    if len(note) < 50:
        console.print("[red]Error: Note must be at least 50 characters long.[/red]")
        return None, None, None
    
    start_time = time.time()
    
    # Run the model to get PHI entities
    phi_entities = run_llama_model(note, model_path, use_gpu=use_gpu, temperature=0)
    
    if phi_entities is None:
        return None, None, None
    
    end_time = time.time()
    
    # Display the raw XML output for debugging
    console.print("\n" + "="*60)
    console.print(Panel(Syntax(phi_entities, "xml", theme="monokai"), title="[bold blue]Extracted PHI Entities (XML)[/bold blue]"))
    
    # Anonymize the text
    highlighted_segments, entity_mapping = simple_text_anonymization(note, phi_entities)
    
    return end_time - start_time, highlighted_segments, entity_mapping

def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(description="EdgeDeID: Medical Note Anonymization using llama.cpp")
    parser.add_argument("--cpu", action="store_true", help="Force CPU-only processing")
    parser.add_argument("--gpu", action="store_true", help="Use GPU acceleration")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to a GGUF model file. Defaults to ./Qwen3-0.6B-Base_SFT-Q4_K_M.gguf next to this script.")

    args = parser.parse_args()

    # Set default model path (next to this script)
    if args.model:
        model_path = Path(args.model)
    else:
        model_path = Path(__file__).parent / "Qwen3-0.6B-Base_SFT-Q4_K_M.gguf"
    
    # Check if model exists
    if not model_path.exists():
        console.print(f"[red]Error: Model file not found: {model_path}[/red]")
        console.print("[yellow]Please ensure the model is at the expected location.[/yellow]")
        sys.exit(1)
    
    # Display header
    console.print(Panel.fit(
        "[bold blue]EdgeDeID: Medical Note Anonymization[/bold blue]\n"
        "[dim]Powered by llama.cpp and Rich[/dim]",
        border_style="blue"
    ))
    
    # Ask for acceleration preference if not specified
    use_gpu = True  # Default to GPU
    if args.cpu:
        use_gpu = False
    elif args.gpu:
        use_gpu = True
    else:
        # Interactive choice
        choice = Prompt.ask(
            "\n[bold cyan]Choose processing mode[/bold cyan]",
            choices=["cpu", "gpu"], 
            default="gpu"
        )
        use_gpu = (choice == "gpu")
    
    acceleration_mode = "🚀 GPU" if use_gpu else "🖥️  CPU"
    console.print(f"\n[bold green]Using {acceleration_mode} acceleration[/bold green]")
    
    # Get input note interactively
    console.print("\n[bold cyan]Please enter your medical note:[/bold cyan]")
    console.print("[dim](Press Ctrl+D when finished, or Ctrl+C to cancel)[/dim]\n")
    
    lines = []
    try:
        while True:
            line = input()
            lines.append(line)
    except EOFError:
        pass
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled.[/yellow]")
        sys.exit(0)
    
    note = '\n'.join(lines)
    
    if not note.strip():
        console.print("[red]Error: No input provided[/red]")
        sys.exit(1)
    
    # Display original note
    console.print("\n" + "="*60)
    console.print(Panel(note, title="[bold green]Original Medical Note[/bold green]", border_style="green"))
    
    # Process the note
    time_taken, highlighted_segments, entity_mapping = sanitize_note(note, str(model_path), use_gpu)
    
    if time_taken is None:
        console.print("[red]Error: Failed to process the note[/red]")
        sys.exit(1)
    
    # Display timing prominently
    console.print("\n" + "="*60)
    timing_text = f"⏱️  Processing completed in {time_taken:.2f} seconds using {acceleration_mode}"
    console.print(Panel.fit(
        f"[bold yellow]{timing_text}[/bold yellow]",
        border_style="yellow"
    ))
    
    # Display anonymized note with highlighting
    console.print("\n" + "="*60)
    highlighted_text = display_highlighted_text(highlighted_segments)
    console.print(Panel(highlighted_text, title="[bold red]Anonymized Medical Note[/bold red]", border_style="red"))
    
    # Display entity mapping table
    if entity_mapping:
        console.print("\n" + "="*60)
        table = Table(title="[bold cyan]Entity Mapping[/bold cyan]", show_header=True, header_style="bold magenta")
        table.add_column("Anonymized Value", style="cyan")
        table.add_column("Original Values", style="green")
        
        for anon_value, original_list in entity_mapping.items():
            table.add_row(anon_value, ", ".join(original_list))
        
        console.print(table)
    else:
        console.print("\n[dim]No PHI entities found in the note.[/dim]")

if __name__ == "__main__":
    main()
