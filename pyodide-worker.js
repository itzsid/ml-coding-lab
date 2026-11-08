/* Python runtime worker (classic worker, not bundled).
 *
 * Runs Pyodide off the main thread so long-running or infinite-looping code
 * can't freeze the page: the main thread enforces a timeout by terminating
 * this worker and starting a fresh one (src/lib/pythonRunner.ts).
 *
 * Protocol (main -> worker):
 *   { type: 'init', version }
 *   { type: 'run',  id, code }
 *   { type: 'test', id, code, testCases, functionName, language }
 * Protocol (worker -> main):
 *   { type: 'ready' } | { type: 'init-error', error }
 *   { type: 'stdout' | 'stderr', id, text }
 *   { type: 'done', id, error? }                      // for 'run'
 *   { type: 'test-results', id, results? , error? }   // for 'test'
 */
/* global importScripts, loadPyodide */
const PYODIDE_VERSION = '0.26.4';
const PYODIDE_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;

let pyodide = null;
let runner = null;
let torchInstalled = false;
let currentId = null;
let version = '';

function post(msg) {
  self.postMessage(msg);
}

async function fetchText(path) {
  const url = new URL(path, self.location.href);
  if (version) url.searchParams.set('v', version);
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to load ${path} (${res.status})`);
  return res.text();
}

async function init() {
  importScripts(`${PYODIDE_URL}pyodide.js`);
  pyodide = await loadPyodide({ indexURL: PYODIDE_URL });
  pyodide.setStdout({ batched: text => post({ type: 'stdout', id: currentId, text }) });
  pyodide.setStderr({ batched: text => post({ type: 'stderr', id: currentId, text }) });
  await pyodide.loadPackage(['numpy']);

  const runnerSrc = await fetchText('py/runner.py');
  pyodide.FS.writeFile('/home/pyodide/runner.py', runnerSrc);
  pyodide.runPython("import sys\nif '/home/pyodide' not in sys.path: sys.path.insert(0, '/home/pyodide')");
  runner = pyodide.pyimport('runner');
}

async function ensureTorch() {
  if (torchInstalled) return;
  const shimSrc = await fetchText('py/torch_shim.py');
  pyodide.FS.writeFile('/home/pyodide/torch_shim.py', shimSrc);
  pyodide.runPython('import torch_shim\ntorch_shim.install()');
  torchInstalled = true;
}

function errorMessage(err) {
  return err && err.message ? err.message : String(err);
}

self.onmessage = async event => {
  const msg = event.data;
  if (msg.type === 'init') {
    version = msg.version || '';
    try {
      await init();
      post({ type: 'ready' });
    } catch (err) {
      post({ type: 'init-error', error: errorMessage(err) });
    }
    return;
  }

  currentId = msg.id;
  if (msg.type === 'run') {
    try {
      // run_code returns None on success or a trimmed traceback string.
      const error = runner.run_code(msg.code);
      post({ type: 'done', id: msg.id, error: error === undefined ? null : error });
    } catch (err) {
      post({ type: 'done', id: msg.id, error: errorMessage(err) });
    }
  } else if (msg.type === 'test') {
    try {
      if (msg.language === 'pytorch') await ensureTorch();
      const json = runner.run_tests(
        msg.code,
        JSON.stringify(msg.testCases),
        msg.functionName,
        msg.language || 'numpy'
      );
      post({ type: 'test-results', id: msg.id, results: JSON.parse(json) });
    } catch (err) {
      post({ type: 'test-results', id: msg.id, error: errorMessage(err) });
    }
  }
  currentId = null;
};
