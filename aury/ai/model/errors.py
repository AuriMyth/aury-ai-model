class ModelError(Exception):
    pass

class ModelTimeoutError(ModelError):
    pass

class RateLimitError(ModelError):
    pass

class ModelOverloadedError(ModelError):
    pass

class InvalidRequestError(ModelError):
    pass

class TransportError(ModelError):
    pass

class SchemaMismatchError(ModelError):
    pass

class StreamBrokenError(ModelError):
    pass

class ProviderNotInstalledError(ModelError):
    pass
