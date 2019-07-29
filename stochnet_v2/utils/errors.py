class ShapeError(Exception):
    """Exception raised for errors in the input shape.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message='Wrong shape!'):
        self.message = message


class DimensionError(Exception):
    """Exception raised for errors concerning the dimension of the used spaces.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message='Wrong dimensions!'):
        self.message = message


class NotRestoredVariables(Exception):
    """Exception raised when trying to use model without loading trained variables.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message='Model variables not restored!'):
        self.message = message
