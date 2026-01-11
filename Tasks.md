1. Modify decoder to output single frame (3 channels, not past_length*3)
    - Test changes on regular e2c
    - If pred_length is >1 should wrap x_next in list?
2. Change dataset generation
    - pred_length parameter
    - Create new buffers, modify control buffer shape
    - Buffer window must be past_length + pred_length
3. Modify dataset loading to work with pred_length shape
4. Modify RSSM code to sequentially predict pred_length horizon
5. Modify eval code to work with pred_length?