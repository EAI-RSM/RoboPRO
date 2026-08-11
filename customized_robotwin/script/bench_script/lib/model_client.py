"""Socket RPC client shared by bench policy drivers."""

import base64
import json
import socket
import time
from typing import Any

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that preserves NumPy array dtype and shape."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {
                "__numpy_array__": True,
                "data": base64.b64encode(obj.tobytes()).decode("ascii"),
                "dtype": str(obj.dtype),
                "shape": obj.shape,
            }
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def numpy_to_json(data: Any) -> str:
    """Convert NumPy-containing data to a JSON string."""
    return json.dumps(data, cls=NumpyEncoder)


def json_to_numpy(json_str: str) -> Any:
    """Convert JSON back to Python objects, reconstructing NumPy arrays."""

    def object_hook(dct):
        if "__numpy_array__" in dct:
            data = base64.b64decode(dct["data"])
            return np.frombuffer(data, dtype=dct["dtype"]).reshape(dct["shape"])
        return dct

    return json.loads(json_str, object_hook=object_hook)


class ModelClient:
    # Attributes that deploy_policy.py reads directly (not as methods).
    # Mirrored on the client side because the server's RPC dispatch only
    # supports method calls, and `model.x is None` cannot be satisfied by a
    # callable proxy.
    _MIRRORED_DEFAULTS = {"observation_window": None}

    def __init__(self, host="localhost", port=9999, timeout=30, **mirrored):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock = None
        # Local mirrors. `mirrored` kwargs (e.g. pi0_step) are forwarded to
        # client-side attributes so deploy_policy can read them without RPC.
        for key, value in self._MIRRORED_DEFAULTS.items():
            object.__setattr__(self, key, value)
        for key, value in mirrored.items():
            object.__setattr__(self, key, value)
        self._connect()

    def _connect(self):
        attempts = 0
        max_attempts = 1000
        retry_delay = 5

        while attempts < max_attempts:
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.settimeout(self.timeout)
                self.sock.connect((self.host, self.port))
                print(f"🔗 Connected to model server at {self.host}:{self.port}")
                return
            except Exception as exc:
                attempts += 1
                if self.sock:
                    self.sock.close()
                if attempts < max_attempts:
                    print(f"⚠️ Connection attempt {attempts} failed: {exc}")
                    print(f"🔄 Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    raise ConnectionError(
                        f"Failed to connect to server after {max_attempts} attempts: {exc}"
                    )

    def _send_recv(self, data):
        """Send a request and receive a response with NumPy array support."""
        try:
            json_data = numpy_to_json(data).encode("utf-8")
            self.sock.sendall(len(json_data).to_bytes(4, "big"))
            self.sock.sendall(json_data)
            return self._recv_response()
        except Exception as exc:
            self.close()
            raise ConnectionError(f"Communication error: {exc}")

    def _recv_response(self):
        len_data = self.sock.recv(4)
        if not len_data:
            raise ConnectionError("Connection closed by server")

        size = int.from_bytes(len_data, "big")
        chunks = []
        received = 0
        while received < size:
            chunk = self.sock.recv(min(size - received, 4096))
            if not chunk:
                raise ConnectionError("Incomplete response received")
            chunks.append(chunk)
            received += len(chunk)
        return json_to_numpy(b"".join(chunks).decode("utf-8"))

    def call(self, func_name=None, obs=None):
        response = self._send_recv({"cmd": func_name, "obs": obs})
        if "res" not in response:
            err = response.get("error", "<no error field in response>")
            tb = response.get("traceback", "")
            raise RuntimeError(
                f"Server error during RPC '{func_name}':\n"
                f"{err}\n--- server traceback ---\n{tb}"
            )
        return response["res"]

    def __getattr__(self, name):
        # __getattr__ only fires for missing attributes, so mirrored fields
        # set via object.__setattr__ are untouched.
        if name.startswith("_") or name in {"host", "port", "timeout", "sock"}:
            raise AttributeError(name)

        def _proxy(*args, **kwargs):
            if kwargs:
                raise TypeError("ModelClient RPC does not support kwargs")
            if len(args) == 0:
                obs = None
            elif len(args) == 1:
                obs = args[0]
            else:
                # Multi-arg -> wrap in dict; server unpacks with `_args` key.
                obs = {"_args": list(args)}
            result = self.call(func_name=name, obs=obs)
            # Update client-side mirrors after methods that change state.
            if name in {"reset_obsrvationwindows", "reset_model"}:
                object.__setattr__(self, "observation_window", None)
            elif name == "update_observation_window":
                # Sentinel: any non-None value satisfies `is None` checks.
                object.__setattr__(self, "observation_window", True)
            return result

        return _proxy

    def close(self):
        """Close the connection."""
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
            finally:
                self.sock = None
                print("🔌 Connection closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
