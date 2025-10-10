from .inline.provider import get_provider_spec as get_inline_provider_spec
from .remote.provider import get_provider_spec as get_remote_provider_spec


def get_provider_spec():
    return [
        get_inline_provider_spec(),
        get_remote_provider_spec(),
    ]


__all__ = ["get_provider_spec"]
