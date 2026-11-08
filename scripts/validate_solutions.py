"""Check every problem's reference solution against its own test cases.

Usage: python3 scripts/validate_solutions.py problems.json [--torch shim|real]

PyTorch tracks (problem['pytorch']) are checked too: by default against the
NumPy-backed shim in public/py/torch_shim.py (what the browser runs), or with
--torch real against an installed PyTorch, to confirm the shim matches torch.

problems.json (a list of problems) is written by
src/__tests__/solutions.test.ts, which runs this script as part of
`npm test`. It uses the exact harness the browser uses (public/py/runner.py),
so a pass here means a pass in the app.
"""
import contextlib
import io
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'public', 'py'))
import runner  # noqa: E402

SLOW_SECONDS = 5


def function_name(starter_code):
    names = re.findall(r'def\s+(\w+)\s*\(', starter_code)
    return names[-1] if names else 'solution'


def tracks(p):
    yield 'numpy', p['id'], p
    if p.get('pytorch'):
        yield 'pytorch', p['id'] + ' [pytorch]', p['pytorch']


def check(problems, torch_mode='shim'):
    failures = []
    torch_ok = setup_torch(torch_mode)
    for language, label, p in (t for problem in problems for t in tracks(problem)):
        if language == 'pytorch' and not torch_ok:
            continue
        fn = function_name(p['starterCode'])
        tests = json.dumps(p['testCases'])

        start = time.time()
        results = json.loads(runner.run_tests(p['solution'], tests, fn, language))
        elapsed = time.time() - start
        for r in results:
            if not r['passed']:
                tc = next(t for t in p['testCases'] if t['id'] == r['id'])
                failures.append(
                    f"{label} test {r['id']}: expected {tc['expected']!r}, got {r['actual']!r}"
                )
        if elapsed > SLOW_SECONDS:
            failures.append(f"{label}: solution took {elapsed:.1f}s (browser timeout risk)")

        # The starter code must not already pass everything.
        starter = json.loads(runner.run_tests(p['starterCode'], tests, fn, language))
        if starter and all(r['passed'] for r in starter):
            failures.append(f"{label}: starter code passes all tests")
        if language == 'pytorch' and any(
            not runner._is_expression(t['input'], fn) for t in p['testCases']
        ):
            failures.append(f"{label}: PyTorch tests must be expression-based (call {fn}(...))")
    return failures


def setup_torch(mode):
    """Make `import torch` resolve to the shim (default) or real PyTorch."""
    if mode == 'real':
        try:
            import torch  # noqa: F401
            return True
        except ImportError:
            print('warning: --torch real but PyTorch is not installed; skipping PyTorch tracks',
                  file=sys.stderr)
            return False
    try:
        import torch_shim
    except ImportError:
        print('warning: public/py/torch_shim.py not found; skipping PyTorch tracks', file=sys.stderr)
        return False
    torch_shim.install()
    return True


def main(path, torch_mode='shim'):
    with open(path) as f:
        problems = json.load(f)
    if isinstance(problems, dict):  # tolerate {problems: [...], sections: [...]}
        problems = problems['problems']
    # Solutions may print; keep stdout clean for the JSON report.
    with contextlib.redirect_stdout(io.StringIO()):
        failures = check(problems, torch_mode)
    print(json.dumps({'checked': len(problems), 'failures': failures}, indent=2))
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    args = sys.argv[1:]
    mode = 'shim'
    if '--torch' in args:
        i = args.index('--torch')
        mode = args[i + 1]
        del args[i:i + 2]
    main(args[0], mode)
