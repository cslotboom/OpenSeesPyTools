# OpenseesPyPostprocessing
This library provides data processing tool and animation tools for opensees.

Notably, the earthquake plotting functions and animation functions are currently limited to models in 2D, that use 1D elements 
(i.e. trusses, beams, ZLE, etc.).

The earthquake animation may run slowly for larger models - I haven't stressed tested it much.
Eventually my goal is to switch away from using a matplotlib back end into something like OpenGL.
