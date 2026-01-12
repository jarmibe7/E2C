1. [X] Modify decoder to output single frame (3 channels, not past_length*3)
    - [X] Test changes on regular e2c
    - [X] If pred_length is >1 should wrap x_next in list?
        * No, we should just predict a trajectory of predictions inside the world model, and output them as a single tensor.
2. Change dataset generation
    - [X] pred_length parameter
    - [X] Create new buffers, modify control buffer shape
    - [X] Buffer window must be past_length + pred_length
3. [X] Modify dataset loading to work with pred_length shape
4. Modify RSSM code to sequentially predict pred_length horizon
    - Decoder uncertainty for predictions? Should I accumulate it, average it what? How to train NLL loss here.
5. Modify eval code to work with pred_length?
    - Should we eval metrics for entire pred_length
    - Only traj vis for immediate next img
6. Modify closed loop data collection to support pred_length parameter