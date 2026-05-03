from typing import List, Tuple, Union, Callable, Optional
import random
import re

class TextAugmentor:
    def __init__(self):
        self.augmentations: List[Tuple[Union[Callable, List[Tuple[Callable, float]]], float, Optional[str]]] = []

    def add_augmentation(self, func: Union[Callable, List[Tuple[Callable, float]]], prob: float, field: str = None):
        self.augmentations.append((func, prob, field))

    def _apply_global_augmentation(self, text: str, aug: Callable) -> str:
        try:
            return aug(text)
        except Exception as e:
            print(f"An error occurred during global augmentation: {e}")
            return text

    def _get_formatted_attribute(self, tag_attrs: str) -> Optional[str]:
        formatted_match = re.search(r'formatted="([^"]*)"', tag_attrs)
        return formatted_match.group(1) if formatted_match else None

    def _apply_augmentation_function(self, aug: Union[Callable, List[Tuple[Callable, float]]], original_value: str) -> Tuple[str, Optional[str]]:
        if isinstance(aug, list):  # OneOf group
            chosen_aug, chosen_prob = random.choices(aug, weights=[a[1] for a in aug])[0]
            augmented_value = chosen_aug(original_value)
        else:  # Single augmentation function
            augmented_value = aug(original_value)

        formatted_value = None
        if not isinstance(augmented_value, str):
            augmented_value, formatted_value = augmented_value
        
        return augmented_value, formatted_value

    def _update_formatted_value(self, text: str, field: str, augmented_value: str, 
                                formatted_value: str, existing_formatted: Optional[str]) -> str:
        # Find the opening '<' of the tag containing the augmented value
        tag_start = text.rfind(f'<{field}', 0, text.rfind(augmented_value))
        # Find the closing '>' of the tag
        tag_end = text.find('>', tag_start)
        
        # Remove existing formatted attribute if present
        if existing_formatted:
            text = text[:tag_start] + text[tag_start:tag_end].replace(f' formatted="{existing_formatted}"', '') + text[tag_end:]
            tag_end = text.find('>', tag_start)  # Recalculate tag_end
            
        # Insert the new formatted attribute
        return text[:tag_end] + f' formatted="{formatted_value}"' + text[tag_end:]

    def _apply_field_augmentation(self, text: str, field: str, aug: Union[Callable, List[Tuple[Callable, float]]]) -> str:
        pattern = rf'<{field}(.*?)>(.*?)</{field}>'
        matches = list(re.finditer(pattern, text))

        for match in matches:
            try:
                tag_attrs = match.group(1)
                original_value = match.group(2)
                existing_formatted = self._get_formatted_attribute(tag_attrs)

                augmented_value, formatted_value = self._apply_augmentation_function(aug, original_value)

                if formatted_value is not None and augmented_value == formatted_value and existing_formatted is not None:
                    formatted_value = existing_formatted

                # Update the text with augmented value
                text = text.replace(f'{original_value}', f'{augmented_value}', 1)

                if formatted_value is not None:
                    text = self._update_formatted_value(text, field, augmented_value, 
                                                        formatted_value, existing_formatted)

            except Exception as e:
                print(f"An error occurred while augmenting field {field}: {e}")

        return text

    def augment(self, text: str) -> str:
        # Process augmentations in the order they were added
        for aug, prob, field in self.augmentations:
            if random.random() >= prob:
                continue

            if field is None:
                text = self._apply_global_augmentation(text, aug)
            else:
                text = self._apply_field_augmentation(text, field, aug)

        return text