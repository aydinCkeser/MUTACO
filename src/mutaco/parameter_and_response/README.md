# Model coordinates and response channels

[parameter_response_specifications.py](parameter_response_specifications.py)
defines what each column of a physical parameter or response tensor means.
A `ParameterAxis` declares one name, finite closed bounds, unit, transformation
and role. `ParameterSpec` holds its axes in scientific column order and exposes
the corresponding lower and upper tuples. A `ResponseAxis` declares one measured
channel and unit. `ResponseSpec` preserves its order on the response/task axis.

These immutable declarations validate their metadata. They do not generate
design points, apply the declared parameter transformation, standardize observed
responses or fit a GP. Their order is shared by the simulator interface, frozen
response transform, multitask training records and target vector. Reordering
axes changes the scientific interpretation even when a tensor retains its shape.

Import the four classes from `mutaco`, `mutaco.parameter_and_response` or their
descriptive module. `mutaco.types` and `mutaco.types.specifications` remain
supported and expose the same class objects, including for historical pickles.
