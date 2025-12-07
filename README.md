## E2C: Embed to Control
**Author: Jared Berry**

#### Objective
The goal of this project was to implement the architecture of Embed to Control: A Locally Linear Latent Dynamics Model for Control from Raw Images.

#### Software Format
- e2c.py<br>
Main E2C model architecture.

- encode.py<br>
Convolutional encoder/decoder architecture

- gen_dummy_dataset.py<br>
Script for generating a dummy dataset to test E2C dimensions/pipeline

- train.py<br>
Main training script for E2C.

- utils.py<br>
Utility functions.

#### Python Notetbooks
- E2C_Dataset_Generation.ipynb<br>
Generate a simple particle in gravity image dataset with lagrangian dynamics.

To run `embed_to_control_V6.ipynb`, make sure to first collect data using `rl_collect_data.ipynb`, saving things in accordance with the ImageDatasetV2 class in `models.py`.

- Embed_to_Control.ipynb<br>
Main E2C architecture.

#### Running on server
```ssh -v jarmibe7@dingo.mech.northwestern.edu```

edit file /etc/motd_bash

Write log on which GPU is being used

Check ```nvtop``` to see GPU resources

To run overnight and prevent from closing:

```screen -S my_session_name```

And run the training script. Then to detach:

```Ctrl + A```   then   ```D```

To resume:

```screen -r my_session_name```

#### Citations
> M. Watter, J. T. Springenberg, J. Boedecker, and M. Riedmiller, “Embed to Control: A Locally Linear Latent Dynamics Model for Control from Raw Images,” in *Advances in Neural Information Processing Systems 28 (NIPS 2015)*, Montréal, Canada, Dec. 2015, pp. 2746–2754. [Online]. Available: [https://arxiv.org/abs/1506.07365](https://arxiv.org/abs/1506.07365)


