class DispatchData:
    __slots__ = ('index', 'params')

    def __init__(self, index: int, params=None):
        self.index = index
        self.params = params if params is not None else {}
