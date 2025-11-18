## E2C: Embed to Control
**Author: Jared Berry**

#### Objective
The goal of this project was to implement the architecture of Embed to Control: A Locally Linear Latent Dynamics Model for Control from Raw Images.

#### Software Format
- E2C_Dataset_Generation.ipynb<br>
Generate a simple particle in gravity image dataset with lagrangian dynamics.

To run `embed_to_control_V6.ipynb`, make sure to first collect data using `rl_collect_data.ipynb`, saving things in accordance with the ImageDatasetV2 class in `models.py`.

- Embed_to_Control.ipynb<br>
Main E2C architecture.

#### Citations
> M. Watter, J. T. Springenberg, J. Boedecker, and M. Riedmiller, “Embed to Control: A Locally Linear Latent Dynamics Model for Control from Raw Images,” in *Advances in Neural Information Processing Systems 28 (NIPS 2015)*, Montréal, Canada, Dec. 2015, pp. 2746–2754. [Online]. Available: [https://arxiv.org/abs/1506.07365](https://arxiv.org/abs/1506.07365)


