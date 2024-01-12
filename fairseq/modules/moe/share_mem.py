
def share_mem(key, value=None):
    if not hasattr(share_mem, '_mem'):
        share_mem._mem = {}
    if value is not None:
        share_mem._mem[key] = value
    else:
        return share_mem._mem.get(key, None)