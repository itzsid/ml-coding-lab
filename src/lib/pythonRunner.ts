/**
 * Singleton Python runtime shared by every page.
 *
 * Pyodide lives in a Web Worker (public/pyodide-worker.js), so:
 *  - it loads once per visit, not once per page mount;
 *  - user code can't freeze the UI;
 *  - a run that exceeds its timeout (or is stopped) is killed by terminating
 *    the worker, and a fresh worker is started in the background.
 */
import type { Language, TestCase } from '../types';

export type RuntimeStatus = 'idle' | 'loading' | 'ready' | 'error';

export interface RawTestResult {
  id: string;
  passed: boolean;
  actual: string;
}

export interface RunOutcome {
  error?: string;
  /** Set when the run was killed by the timeout or by stop(). */
  aborted?: 'timeout' | 'stopped';
}

export interface TestOutcome extends RunOutcome {
  results?: RawTestResult[];
}

type OutputHandler = (text: string, stream: 'stdout' | 'stderr') => void;

interface PendingJob {
  id: number;
  onOutput: OutputHandler;
  resolve: (value: TestOutcome) => void;
  timer: ReturnType<typeof setTimeout>;
}

export const DEFAULT_TIMEOUT_MS = 30_000;

class PythonRunner {
  private worker: Worker | null = null;
  private ready: Promise<void> | null = null;
  private status: RuntimeStatus = 'idle';
  private initError: string | null = null;
  private listeners = new Set<() => void>();
  private pending: PendingJob | null = null;
  private queue: Promise<unknown> = Promise.resolve();
  private nextId = 1;

  getStatus(): RuntimeStatus {
    return this.status;
  }

  getInitError(): string | null {
    return this.initError;
  }

  subscribe(listener: () => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  private setStatus(status: RuntimeStatus, error: string | null = null) {
    this.status = status;
    this.initError = error;
    this.listeners.forEach(l => l());
  }

  /** Start the worker if needed. Safe to call repeatedly. */
  start(): Promise<void> {
    if (this.ready) return this.ready;
    if (typeof Worker === 'undefined') {
      this.setStatus('error', 'Web Workers are not supported in this browser');
      return Promise.reject(new Error(this.initError!));
    }

    const base = import.meta.env.BASE_URL || '/';
    const worker = new Worker(`${base}pyodide-worker.js`);
    this.worker = worker;
    this.setStatus('loading');

    this.ready = new Promise<void>((resolve, reject) => {
      worker.onmessage = (event: MessageEvent) => {
        const msg = event.data;
        if (msg.type === 'ready') {
          this.setStatus('ready');
          resolve();
        } else if (msg.type === 'init-error') {
          this.setStatus('error', msg.error);
          this.ready = null; // allow a retry on the next call
          reject(new Error(msg.error));
        } else {
          this.handleMessage(msg);
        }
      };
      worker.onerror = event => {
        const message = event.message || 'Python worker crashed';
        if (this.status === 'loading') {
          this.setStatus('error', message);
          this.ready = null;
          reject(new Error(message));
        }
      };
    });
    const version = typeof __BUILD_ID__ !== 'undefined' ? __BUILD_ID__ : '';
    worker.postMessage({ type: 'init', version });
    // Avoid unhandled-rejection noise; callers observe failures via status.
    this.ready.catch(() => {});
    return this.ready;
  }

  private handleMessage(msg: { type: string; id: number; text?: string; error?: string | null; results?: RawTestResult[] }) {
    const job = this.pending;
    if (!job || msg.id !== job.id) return;
    if (msg.type === 'stdout' || msg.type === 'stderr') {
      job.onOutput(msg.text ?? '', msg.type);
    } else if (msg.type === 'done') {
      this.finish(job, msg.error ? { error: msg.error } : {});
    } else if (msg.type === 'test-results') {
      this.finish(job, msg.error ? { error: msg.error } : { results: msg.results });
    }
  }

  private finish(job: PendingJob, outcome: TestOutcome) {
    clearTimeout(job.timer);
    if (this.pending === job) this.pending = null;
    job.resolve(outcome);
  }

  /** Kill the worker (aborting any running job) and boot a fresh one. */
  private restart(reason: 'timeout' | 'stopped') {
    const job = this.pending;
    this.worker?.terminate();
    this.worker = null;
    this.ready = null;
    this.setStatus('idle');
    if (job) this.finish(job, { aborted: reason });
    this.start().catch(() => {});
  }

  /** Abort the running job, if any. */
  stop() {
    if (this.pending) this.restart('stopped');
  }

  private enqueue(
    message: Record<string, unknown>,
    onOutput: OutputHandler,
    timeoutMs: number
  ): Promise<TestOutcome> {
    const run = async (): Promise<TestOutcome> => {
      try {
        await this.start();
      } catch (err) {
        return { error: err instanceof Error ? err.message : 'Python failed to load' };
      }
      return new Promise<TestOutcome>(resolve => {
        const id = this.nextId++;
        const job: PendingJob = {
          id,
          onOutput,
          resolve,
          timer: setTimeout(() => this.restart('timeout'), timeoutMs),
        };
        this.pending = job;
        this.worker!.postMessage({ ...message, id });
      });
    };
    // Serialize jobs: one worker, one job at a time.
    const result = this.queue.then(run, run);
    this.queue = result.catch(() => {});
    return result;
  }

  runCode(code: string, onOutput: OutputHandler, timeoutMs = DEFAULT_TIMEOUT_MS): Promise<RunOutcome> {
    return this.enqueue({ type: 'run', code }, onOutput, timeoutMs);
  }

  runTests(
    code: string,
    testCases: TestCase[],
    functionName: string,
    language: Language,
    onOutput: OutputHandler,
    timeoutMs = DEFAULT_TIMEOUT_MS
  ): Promise<TestOutcome> {
    const slim = testCases.map(({ id, input, expected }) => ({ id, input, expected }));
    return this.enqueue(
      { type: 'test', code, testCases: slim, functionName, language },
      onOutput,
      timeoutMs
    );
  }
}

export const pythonRunner = new PythonRunner();
