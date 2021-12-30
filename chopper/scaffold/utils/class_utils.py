__all__ = [
    "classproperty",
    "memoized_classproperty",
]

# classproperty for singeton pass registry


class classproperty(object):
    """@classmethod+@property"""
    def __init__(self, f):
        self.f = classmethod(f)

    def __get__(self, *a):
        return self.f.__get__(*a)()


class memoized_classproperty(object):
    """@classmethod+@property"""
    def __init__(self, f):
        self.f = classmethod(f)

    def __get__(self, instance, owner):
        # get the value:
        value = self.f.__get__(instance, owner)()
        # inject the value into class's __dict__ before returning:
        attr = self.f.__func__.__name__
        setattr(owner, attr, value)
        return value
