# craft-env
OpenAI gym-like environment for simple craft game

## Requirements

### Spot requirement
The step function of this craft environment is using a Buchi automaton to validate if an input action follows the specified LTL formula. The conversion from LTL to Buchi automaton is done by Spot. Please follow the [Spot installation guide](https://spot.lrde.epita.fr/install.html) to install required packages and the Python bindings. 

