# Installation Guide

## Quick Installation

To install all required packages, run one of the following commands:

### Option 1: Using pip (Recommended)
```bash
pip install -r requirements.txt
```

### Option 2: Install individually
```bash
pip install numpy pandas librosa matplotlib scipy soundfile
```

### Option 3: In Jupyter Notebook
Add a new cell at the beginning and run:
```python
!pip install numpy pandas librosa matplotlib scipy soundfile
```

## Troubleshooting

### If you get "ModuleNotFoundError: No module named 'librosa'"

1. **In Jupyter Notebook:**
   - Go to the first cell (Install Required Packages)
   - Uncomment the `!pip install` line
   - Run that cell first

2. **In Terminal/Command Prompt:**
   ```bash
   pip install librosa
   ```

### If librosa installation fails

Librosa requires some system dependencies. On Windows:
- Make sure you have Visual C++ Redistributable installed
- Try installing with: `pip install librosa --no-cache-dir`

### If you get errors about soundfile

On Windows, you might need:
```bash
pip install soundfile
```

If that fails, try:
```bash
pip install soundfile --no-binary soundfile
```

## Verify Installation

After installation, test if everything works:
```python
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
print("All packages installed successfully!")
```





