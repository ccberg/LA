def get_custom_attributes(obj):
    return [a for a in dir(obj) if not a.startswith('__')]
