\# Augmentation Pipeline - Person C



\## Files

\- `augmentations.py` - Main augmentation pipeline

\- `run\_robustness\_table.py` - Script for Person D to run on Sunday

\- `demo\_augmentation.py` - Demo script for video



\## How to Use



\### For Person B (Training)

```python

from augmentations import get\_train\_augmentation



train\_transform = get\_train\_augmentation()

\# Use with Person A's ImageDataset



\### For Person D

from augmentations import get\_test\_conditions

test\_conditions = get\_test\_conditions()

\# Run each condition through evaluate()



\### To run robustness\_table

python run\_robustness\_table.py

\# (After filling in paths and model loader)

