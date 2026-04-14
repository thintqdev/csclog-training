from .drain import LogParser
from .base import BaseParser
from .linux import LinuxParser
from .windows import WindowsParser
from .mac import MacParser
from .network import NetworkParser

PARSERS = {
    "linux": LinuxParser,
    "windows": WindowsParser,
    "mac": MacParser,
    "network": NetworkParser,
}


def get_parser(name: str, cfg) -> BaseParser:
    if name not in PARSERS:
        raise ValueError(f"Unknown parser '{name}'. Choose from {list(PARSERS)}")
    return PARSERS[name](cfg)


__all__ = ["LogParser", "BaseParser", "LinuxParser", "WindowsParser", "MacParser", "NetworkParser", "get_parser"]
