# This acts as a static point of reference for objects that are used all over for ease of access.

# The Class can store values statically
class Interface():
    # Initialize variables to None
    path_manager = None

# Sets the static values
def init_values(pm):
    Interface.path_manager = pm