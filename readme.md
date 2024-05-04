### A WIP project for simulating a sailboat using CFD, by LemonKat.

![example screenshot](images/screenshot1.png)

Each fluid is represented by 3 large arrays of floats, storing the velocity and density of the fluid.
The values in the grids are moved along the velocity vectors via Forward Advection.
There is also a projection step where velocities are adjusted to flow from high to low pressure.

The sailboat part is a work in progress, but an example of the fluid flowing around an object can be run by running `python3 fluid.py`.

#### Required libraries:
1. `numpy`
2. `pygame`
3. `pymunk`
4. `TensorFlow`
5. `skimage` (`scikit-image`)
