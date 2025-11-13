from pathlib import Path as _Path
class TabPrint:
    def __init__(self, tab:str="  ", tabval:int=1,*,file:_Path=None):
        self.__indt = 0
        self.__tab = tab
        self.__tabval = tabval
        self.__tabctx = self._TabContext(self)
        self.__file = file
        if self.__file is not None:
            # Clear the file at the start
            with open(self.__file, 'w') as f:
                pass
    def print(self, *args, **kwargs):
        newargs = [self.__tab * self.__indt * self.__tabval, *args] if self.__indt > 0 else args
        if self.__file is None:
            print(*newargs, **kwargs)
        else:
            with open(self.__file.resolve(), 'a') as f:
                print(*newargs, file=f, **kwargs)
    def indent(self):
        self.__indt += 1
    def unindent(self):
        self.__indt -= 1
    def __call__(self, *args, **kwargs):
        self.print(*args, **kwargs)
    
    class _TabContext:
        def __init__(self, parent:'TabPrint'):
            self.parent = parent
        def __enter__(self):
            self.parent.indent()
            return self
        def __exit__(self, exc_type, exc_value, traceback):
            self.parent.unindent()

    @property
    def tab(self):
        return self.__tabctx
    
