# Dataset Attribution

This project uses the following open-source datasets for training and testing. Attribution is required by their licenses.

---

## DABStep - Multi-Step Reasoning Dataset

**License**: CC-BY-4.0 (Creative Commons Attribution 4.0 International)

**Source**: Hugging Face
**Dataset**: `adyen/DABstep`
**URL**: https://huggingface.co/datasets/adyen/DABstep

**Description**: DABStep contains 450+ data analysis tasks requiring sequential, iterative problem-solving. It is used for training HRM (Hierarchical Reasoning Module) and TRM (Task Refinement Module) agents.

**Citation**:
```
@dataset{dabstep2024,
  title={DABStep: Data Agent Benchmark for Multi-step Reasoning},
  author={Adyen},
  year={2024},
  publisher={Hugging Face},
  url={https://huggingface.co/datasets/adyen/DABstep}
}
```

**License Terms** (CC-BY-4.0):
- ✅ Free to use for any purpose
- ✅ Free to modify and create derivatives
- ✅ Free for commercial use
- ⚠️ **Must give appropriate credit**
- ⚠️ **Must indicate if changes were made**
- ⚠️ **Must provide link to license**

---

## PRIMUS - Cybersecurity Intelligence Suite

**License**: ODC-BY (Open Data Commons Attribution License)

**Source**: Trend Micro AI Lab
**Datasets**:
- `trendmicro-ailab/Primus-Seed` - Knowledge base (674,848 documents)
- `trendmicro-ailab/Primus-Instruct` - Instruction tuning (835 samples)

**URL**: https://huggingface.co/datasets/trendmicro-ailab/Primus-Seed

**Description**: PRIMUS provides cybersecurity domain knowledge including MITRE ATT&CK techniques, threat intelligence, and security documentation. Used for RAG knowledge base and cybersecurity tactical analysis.

**Citation**:
```
@dataset{primus2025,
  title={PRIMUS: Pre-training Cybersecurity Intelligence},
  author={Trend Micro AI Lab},
  year={2025},
  publisher={Hugging Face},
  url={https://huggingface.co/datasets/trendmicro-ailab/Primus-Seed}
}
```

**License Terms** (ODC-BY):
- ✅ Free to share and redistribute
- ✅ Free to create and distribute derivative works
- ✅ Free to adapt and combine with other datasets
- ⚠️ **Must attribute the source**
- ⚠️ **Must keep license notices intact**

---

## Additional Resources

### MITRE ATT&CK Framework
**License**: Free for use with attribution
**URL**: https://attack.mitre.org/
**Use**: Threat intelligence and tactical analysis

---

## How to Properly Attribute

### In Documentation
Include this attribution section in your project documentation:
```markdown
This project uses the following open-source datasets:
- DABStep (CC-BY-4.0): Multi-step reasoning dataset from Hugging Face
- PRIMUS (ODC-BY): Cybersecurity intelligence from Trend Micro AI Lab
```

### In Academic Papers
Include citations in your references section:
```
[1] DABStep: Data Agent Benchmark for Multi-step Reasoning. Hugging Face, 2024.
[2] PRIMUS: Pre-training Cybersecurity Intelligence. Trend Micro AI Lab, 2025.
```

### In Source Code
Include license headers in files that use the datasets:
```python
# This module uses data from:
# - DABStep (CC-BY-4.0) - https://huggingface.co/datasets/adyen/DABstep
# - PRIMUS (ODC-BY) - https://huggingface.co/datasets/trendmicro-ailab/Primus-Seed
```

---

## License Compliance Checklist

### CC-BY-4.0 (DABStep)
- [x] Attribution provided (this file)
- [x] License identified (CC-BY-4.0)
- [x] Source linked (Hugging Face URL)
- [x] Changes indicated (preprocessed for training)
- [x] No additional restrictions added

### ODC-BY (PRIMUS)
- [x] Attribution provided (this file)
- [x] License identified (ODC-BY)
- [x] Source linked (Hugging Face URL)
- [x] License notice preserved (in metadata)
- [x] Distribution compliant

---

## Usage in This Project

### Dataset Loading (`src/data/dataset_loader.py`)
- DABStep loaded via `DABStepLoader` class
- PRIMUS loaded via `PRIMUSLoader` class
- License information preserved in sample metadata

### Training Pipeline
- Multi-step reasoning: DABStep samples
- Domain knowledge: PRIMUS-Seed documents
- Instruction tuning: PRIMUS-Instruct samples

### Preprocessing (`src/data/preprocessing.py`)
- Text cleaning and normalization
- Feature extraction for meta-controller
- Tokenization and encoding

### Augmentation (`src/data/tactical_augmentation.py`)
- Tactical scenario variations
- Urgency level modifications
- Constraint additions

---

## Contact

For questions about dataset usage and licensing:
- DABStep: Adyen via Hugging Face
- PRIMUS: Trend Micro AI Lab via Hugging Face
- MITRE ATT&CK: mitre.org

---

**Last Updated**: November 2025
