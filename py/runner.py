"""Test harness shared by the in-browser runner (public/pyodide-worker.js)
and the offline solution check (scripts/validate_solutions.py).

Keep this file dependency-free apart from NumPy: it must run unchanged in
Pyodide and in CPython.
"""
import json
import linecache
import random
import re
import sys
import traceback

import numpy as np

SEED = 0


def _normalize(value):
    """Turn torch tensors (real or shim) into NumPy / Python values, recursively."""
    if hasattr(value, 'detach') and hasattr(value, 'numpy') and not isinstance(value, np.ndarray):
        arr = value.detach().numpy()
        arr = np.asarray(arr)
        return arr.item() if arr.ndim == 0 else arr
    if isinstance(value, tuple):
        return tuple(_normalize(v) for v in value)
    if isinstance(value, list):
        return [_normalize(v) for v in value]
    if isinstance(value, dict):
        return {k: _normalize(v) for k, v in value.items()}
    return value


def _to_str(result):
    """Stringify a test result the way expected values are written."""
    result = _normalize(result)
    if isinstance(result, (bool, np.bool_)):
        return str(bool(result))
    if isinstance(result, np.ndarray):
        return str(result.tolist())
    if isinstance(result, tuple):
        converted = []
        for item in result:
            if isinstance(item, (bool, np.bool_)):
                converted.append(bool(item))
            elif isinstance(item, np.ndarray):
                converted.append(item.tolist())
            elif isinstance(item, (np.floating, float)):
                converted.append(round(float(item), 6) + 0.0)  # + 0.0 turns -0.0 into 0.0
            elif isinstance(item, (np.integer, int)):
                converted.append(int(item))
            else:
                converted.append(item)
        return str(tuple(converted))
    if isinstance(result, (list, dict)):
        return json.dumps(result)
    if isinstance(result, (np.floating, float)):
        return str(round(float(result), 6) + 0.0)
    if isinstance(result, (np.integer, int)):
        return str(int(result))
    return str(result)


def _is_expression(inp, function_name):
    """Expression tests are evaluated as-is; argument tests are passed to the function."""
    s = inp.strip()
    looks_like_expr = (
        function_name in inp
        or '.' in inp
        or re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', s) is not None
    )
    is_lambda = s.startswith('(lambda')
    return looks_like_expr and not inp.startswith('[') and (not inp.startswith('(') or is_lambda)


def _format_error(exc):
    return f"{type(exc).__name__}: {exc}"


def _compile(code):
    # Register the source so tracebacks can show the offending line.
    linecache.cache['<your code>'] = (len(code), None, code.splitlines(True), '<your code>')
    return compile(code, '<your code>', 'exec')


def _seed():
    np.random.seed(SEED)
    random.seed(SEED)
    _seed_torch()


def _seed_torch():
    torch = sys.modules.get('torch')
    if torch is not None and hasattr(torch, 'manual_seed'):
        torch.manual_seed(SEED)


def run_tests(code, test_cases_json, function_name, language='numpy'):
    """Run test cases against user code in a fresh namespace.

    language='pytorch' makes `torch` (the real package or the shim installed
    by torch_shim.install()) available to test expressions.

    Returns a JSON list of {id, passed, actual}. If the user code itself fails
    to execute, every test is reported with that error.
    """
    test_cases = json.loads(test_cases_json)
    ns = {'__name__': '__main__'}
    _seed()
    try:
        exec(_compile(code), ns)
    except BaseException as exc:  # noqa: BLE001 - report anything, incl. SystemExit
        msg = _format_error(exc)
        print(f"[Error] {msg}")
        return json.dumps([{'id': tc['id'], 'passed': False, 'actual': msg} for tc in test_cases])

    # Test expressions may use np/json even if the user code didn't import them.
    ns.setdefault('np', np)
    ns.setdefault('json', json)
    if language == 'pytorch':
        import torch  # noqa: F401 - the shim registers itself as `torch`
        ns.setdefault('torch', torch)

    results = []
    for tc in test_cases:
        inp = tc['input']
        _seed()
        try:
            if _is_expression(inp, function_name):
                result = eval(inp, ns)
            else:
                test_input = eval(inp, ns)
                fn = ns.get(function_name)
                if fn is None:
                    raise NameError(f"name '{function_name}' is not defined")
                if isinstance(test_input, list):
                    result = fn(np.array(test_input))
                elif isinstance(test_input, tuple):
                    result = fn(*[np.array(x) if isinstance(x, list) else x for x in test_input])
                else:
                    result = fn(test_input)
            actual = _to_str(result)
            passed = re.sub(r'\s', '', actual) == re.sub(r'\s', '', tc['expected'])
        except BaseException as exc:  # noqa: BLE001
            actual = _format_error(exc)
            passed = False
        results.append({'id': tc['id'], 'passed': passed, 'actual': actual})
    return json.dumps(results)


def run_code(code):
    """Run free-form code (scratchpad) in a fresh namespace.

    Returns None on success, or a traceback trimmed to the user's own frames.
    """
    ns = {'__name__': '__main__'}
    try:
        exec(_compile(code), ns)
        return None
    except BaseException as exc:  # noqa: BLE001
        frames = [f for f in traceback.extract_tb(exc.__traceback__) if f.filename == '<your code>']
        lines = ['Traceback (most recent call last):']
        lines += [f'  line {f.lineno}, in {f.name}' + (f'\n    {f.line}' if f.line else '') for f in frames]
        lines.append(_format_error(exc))
        return '\n'.join(lines) if frames else _format_error(exc)
