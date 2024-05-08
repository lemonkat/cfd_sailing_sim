### A WIP project for simulating a sailboat using CFD, by LemonKat.

![example screenshot](images/screenshot1.png)

This program approximates the Navier-Stokes equations with 3 grids of float values, storing velocity and density of a fluid. The TensorFlow library is used to optimize everything as much as possible.

Inspired by these papers:
1. [_The Story of Airplane Wings_](https://arxiv.org/abs/2010.07446)
2. [_Real-Time Fluid Dynamics for Games_](http://graphics.cs.cmu.edu/nsp/course/15-464/Fall09/papers/StamFluidforGames.pdf)

The sailboat part is a work in progress, but an example of the fluid flowing around an object can be run by running `python3 fluid.py`.

#### Required libraries:
1. `numpy`
2. `pygame`
3. `pymunk`
4. `TensorFlow`
5. `skimage` (`scikit-image`)
