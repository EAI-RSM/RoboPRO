"""Run the package's pure unit tests:  python -m robo_negative"""
import robo_negative as rn

_fns = [v for k, v in sorted(vars(rn).items()) if k.startswith("test_") and callable(v)]
for _fn in _fns:
    _fn()
    print(f"PASS {_fn.__name__}")
print(f"{len(_fns)} tests passed")
