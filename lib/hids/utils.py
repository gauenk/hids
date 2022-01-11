

def optional(pydict,field,default):
    if pydict is None: return default
    if field in pydict: return pydict[field]
    else: return default
