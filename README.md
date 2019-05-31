# One Shot Video Object Segmentation with GCN-based contour propagation
> If you have any question or want to report a bug, please open an issue instead of emailing me directly.

## Repository Structure
- **download_DAVIS_2016.sh**: Downloads DAVIS 2016 dataset and saves it to **DAVIS_2016**
- **contour_extraction.ipynb**: Extracts contours from DAVIS 2016 annotations for ground truth generation

## Dependencies
> See requirements.txt for condaenv

## Remarks

## References

## ToDos
[x] Check which sequences to use [Christoph]
```python
bad_sequences = ['bmx-trees', 'bus', 'cows', 'dog-agility', 'horsejump-high', 'horsejump-low', 
                 'kite-walk', 'lucia', 'libby', 'motorbike', 'paragliding', 'rhino', 'scooter-gray', 
                 'swing']
```
[ ] Extract OSVOS feature vectors [Max]
[ ] Start with Graph implementation [both]
