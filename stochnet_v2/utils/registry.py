def normalize_to_list(data):
    if data is None:
        return []

    if isinstance(data, list):
        return data

    return [data]


class Registry(dict):

    def __init__(self, key_type=str, value_type=None, name='Registry'):
        self._key_type = key_type
        self._value_type = value_type
        self._name = name
        super().__init__()

    def _check_key_type(self, key):
        if self._key_type is None or isinstance(key, self._key_type):
            return

        raise TypeError(f'registry key must be type {self._key_type}')

    def _check_value_type(self, value):
        if self._value_type is None or isinstance(value, self._value_type):
            return

        raise TypeError(f'registry value must be type {self._value_type}')

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError as exc:
            raise KeyError(f'key "{key}" not registered in {self._name}') from exc

    def __setitem__(self, key, value):
        self._check_key_type(key)
        self._check_value_type(value)
        if key in self:
            raise ValueError(f'item with key "{key}" already exists')

        super().__setitem__(key, value)

    def register(self, aliases):
        aliases = normalize_to_list(aliases)

        def add_item(item):
            for alias in aliases:
                self.__setitem__(alias, item)

            return item

        return add_item


