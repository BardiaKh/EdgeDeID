from transformers import AutoTokenizer, AutoModelForCausalLM

def expand_tokenizer(tokenizer: AutoTokenizer, model: AutoModelForCausalLM = None):
    """
    Expand tokenizer with PHI schema-specific tokens
    """
    new_tokens = []
    
    # 1. Container tags (need \n after opening, contain other elements)
    container_tags = [
        "names", "ages", "dates", "places", "addresses", 
        "phones", "emails", "ids", "urls", "devices", "variations"
    ]
    
    for tag in container_tags:
        new_tokens.extend([
            f"<{tag}>\n",      # Opening with newline
            f"</{tag}>\n",       # Closing without newline (content handles spacing)
            f"<{tag}/>\n"      # Self-closing with newline for empty containers
        ])
    
    # 2. Simple content tags (inline content, with \n after closing)
    simple_content_tags = [
        "age", "address", "phone", "email", "url", "device", 
        "value", "original", "formatted"
    ]
    
    for tag in simple_content_tags:
        new_tokens.extend([
            f"<{tag}>",        # Opening without newline (content is inline)
            f"</{tag}>\n",     # Closing with newline
        ])
    
    # 3. Tags with attributes (need attribute tokens)
    # name: id="X" type="patient/provider/other"
    new_tokens.extend([
        "<name",           # Opening tag start
        "</name>\n",        # Closing
        ' id="',            # Attribute start
        ' type="',          # Attribute start
    ])
    
    # place: id="X" type="hospital/clinic/other"  
    new_tokens.extend([
        "<place",          # Opening tag start
        "</place>\n",      # Closing
                           # Already has ID attribute
    ])
    
    # id: type="MRN/SSN/NPI/health_plan/account/billing_number/license/other"
    new_tokens.extend([
        "<id",             # Opening tag start
        "</id>\n",         # Closing
                           # Already has ID attribute
    ])
    
    # 4. Special structure tags
    new_tokens.extend([
        "<date>\n",          # Date can contain children
        "</date>\n",
        "<examples>\n",
        "</examples>\n",
        "<example_note>\n",  # Example content
        "</example_note>\n",
        "<example_phi_entities>\n",
        "</example_phi_entities>\n",
    ])
    
    # 5. PHI entities tag
    new_tokens.extend([
        "<phi_entities>\n",
        "</phi_entities>",
    ])
        
    # Add all new tokens
    print(f"Adding {len(new_tokens)} new tokens to tokenizer")
    tokenizer.add_tokens(new_tokens)
    
    # Resize model embeddings
    if model is not None:
        model.resize_token_embeddings(len(tokenizer), mean_resizing=True)
        print(f"Tokenizer vocabulary size: {len(tokenizer)}")
        
        return tokenizer, model
    else:
        print(f"Tokenizer vocabulary size: {len(tokenizer)}")
        return tokenizer
