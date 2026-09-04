import re

def extract_method(source, method_name):
    # This is a simple extractor that finds the method and assumes it ends when the next top-level indent (4 spaces) def or class appears, or EOF
    pattern = re.compile(rf'^( {4}def {method_name}\b.*?(?=\n {4}(?:def|class|@|\Z)))', re.MULTILINE | re.DOTALL)
    match = pattern.search(source)
    if match:
        return match.group(1), match.span(1)
    
    # Try async def
    pattern = re.compile(rf'^( {4}async def {method_name}\b.*?(?=\n {4}(?:def|async def|class|@|\Z)))', re.MULTILINE | re.DOTALL)
    match = pattern.search(source)
    if match:
        return match.group(1), match.span(1)
    return None, None

def main():
    with open('src/training/unified_orchestrator.py', 'r', encoding='utf-8') as f:
        source = f.read()

    mixins = {
        'MetricsMixin': ['_log_metrics', '_get_memory_utilization'],
        'CheckpointMixin': ['_save_checkpoint', 'load_checkpoint'],
        'SelfPlayMixin': ['_generate_self_play_data'],
        'TrainingMixin': ['_train_policy_value_network', '_compute_gradient_norm', '_train_hrm_agent', '_train_trm_agent', '_evaluate']
    }

    import os
    os.makedirs('src/training/orchestrator_components', exist_ok=True)
    
    with open('src/training/orchestrator_components/__init__.py', 'w', encoding='utf-8') as f:
        pass
        
    extracted_source = source
    
    for mixin_name, methods in mixins.items():
        mixin_code = f"from typing import Any\nimport time\nfrom pathlib import Path\nimport torch\nimport torch.nn as nn\nfrom ...observability.logging import get_structured_logger\n\nlogger = get_structured_logger(__name__)\n\nclass {mixin_name}:\n"
        
        for method in methods:
            code, span = extract_method(extracted_source, method)
            if code:
                mixin_code += code + "\n"
                # Remove code from original, replace with a small comment to keep things sane, or just remove it
                # Actually, wait, replacing by index is better. 
                # But multiple replacements will shift indices.
                # So we replace with empty spaces of same length
                extracted_source = extracted_source[:span[0]] + (' ' * (span[1] - span[0])) + extracted_source[span[1]:]
            else:
                print(f"Failed to find {method}")
                
        # Write mixin file
        filename = f"src/training/orchestrator_components/{mixin_name.lower().replace('mixin', '_mixin')}.py"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(mixin_code)
            
    # Clean up empty lines from extracted_source
    cleaned = []
    for line in extracted_source.split('\n'):
        if line.strip() != '' or line == '': # keep intentional newlines but drop the huge empty gaps?
            # actually just replacing spaces means we have lines with just spaces
            if line.strip() == '' and len(line) > 0:
                continue
            cleaned.append(line)
            
    # Add imports to the top of unified_orchestrator.py
    imports = """from .orchestrator_components.metrics_mixin import MetricsMixin
from .orchestrator_components.checkpoint_mixin import CheckpointMixin
from .orchestrator_components.self_play_mixin import SelfPlayMixin
from .orchestrator_components.training_mixin import TrainingMixin
"""
    final_source = imports + "\n" + "\n".join(cleaned)
    final_source = final_source.replace('class UnifiedTrainingOrchestrator:', 'class UnifiedTrainingOrchestrator(MetricsMixin, CheckpointMixin, SelfPlayMixin, TrainingMixin):')
    
    with open('src/training/unified_orchestrator_refactored.py', 'w', encoding='utf-8') as f:
        f.write(final_source)
        
    print("Done refactoring.")

if __name__ == '__main__':
    main()
