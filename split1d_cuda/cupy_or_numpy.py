try:
    # import cupy as xp
    import numpy as xp
    print("using numpy")
    # Using cupy didn't help as a direct drop in replacement
except:
    import numpy as xp
    print("using numpy")
