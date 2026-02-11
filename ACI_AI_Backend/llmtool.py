from typing import Callable, get_type_hints, Any
import inspect

class Tool:
    def __init__(self, func: Callable, description: str):
        self.func = func
        self.name = func.__name__
        self.description = description

        sig = inspect.signature(self.func)
        type_hints = get_type_hints(self.func)

        self.parameters: dict[str, dict[str, Any]] = {}
        for name, param in sig.parameters.items():
            self.parameters[name] = {
                "type": type_hints.get(name, Any),
                "required": param.default is inspect._empty,
                "default": None if param.default is inspect._empty else param.default,
                "kind": param.kind.name,
            }

    def __str__(self) -> str:
        lines = [
            f"Tool name: {self.name}",
            f"Description: {self.description}",
            "Parameters:",
        ]

        if not self.parameters:
            lines.append("  (none)")

        for name, info in self.parameters.items():
            type_name = info["type"].__name__ if hasattr(info["type"], "__name__") else str(info["type"])
            lines.append(f"  - {name}: type={type_name}, " f"{'required' if info['required'] else 'optional'}, " f"default={info['default']}")

        return "\n".join(lines)
