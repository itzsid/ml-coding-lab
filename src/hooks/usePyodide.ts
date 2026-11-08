import { useState, useCallback, useEffect, useSyncExternalStore } from 'react';
import { Language, TestCase, TestResult } from '../types';
import { pythonRunner, RunOutcome, TestOutcome, DEFAULT_TIMEOUT_MS } from '../lib/pythonRunner';

interface UsePyodideReturn {
  isLoading: boolean;
  isReady: boolean;
  error: string | null;
  output: string[];
  runCode: (code: string) => Promise<{ success: boolean; error?: string }>;
  /** Run tests. `silent` skips console output (used for background checks). */
  runTests: (
    code: string,
    testCases: TestCase[],
    functionName: string,
    language?: Language,
    options?: { silent?: boolean }
  ) => Promise<TestResult[]>;
  /** Abort the running code/tests (restarts the Python runtime). */
  stop: () => void;
  clearOutput: () => void;
}

function abortMessage(outcome: RunOutcome): string | null {
  if (outcome.aborted === 'timeout') {
    return `Stopped: your code ran longer than ${DEFAULT_TIMEOUT_MS / 1000}s (infinite loop?). Python is restarting…`;
  }
  if (outcome.aborted === 'stopped') return 'Stopped. Python is restarting…';
  return null;
}

/**
 * React binding for the shared Python runtime (src/lib/pythonRunner.ts).
 * Every component gets its own console output; the runtime itself is shared.
 */
export function usePyodide(): UsePyodideReturn {
  const status = useSyncExternalStore(
    cb => pythonRunner.subscribe(cb),
    () => pythonRunner.getStatus()
  );
  const [output, setOutput] = useState<string[]>([]);

  useEffect(() => {
    pythonRunner.start().catch(() => {});
  }, []);

  useEffect(() => {
    if (status === 'ready') {
      setOutput(prev => (prev.length === 0 ? ['Python environment ready with NumPy loaded.'] : prev));
    } else if (status === 'error') {
      setOutput(prev => [...prev, `[Error] ${pythonRunner.getInitError() ?? 'Failed to load Python'}`]);
    }
  }, [status]);

  const appendLines = useCallback((text: string, prefix = '') => {
    const lines = text.split('\n').filter(line => line.trim());
    if (lines.length) setOutput(prev => [...prev, ...lines.map(l => prefix + l)]);
  }, []);

  const clearOutput = useCallback(() => setOutput([]), []);

  const runCode = useCallback(async (code: string) => {
    setOutput([]);
    const outcome = await pythonRunner.runCode(code, (text, stream) =>
      appendLines(text, stream === 'stderr' ? '[Error] ' : '')
    );
    const aborted = abortMessage(outcome);
    if (aborted) {
      appendLines(aborted, '[Error] ');
      return { success: false, error: aborted };
    }
    if (outcome.error) {
      appendLines(outcome.error, '[Error] ');
      return { success: false, error: outcome.error };
    }
    return { success: true };
  }, [appendLines]);

  const runTests = useCallback(async (
    code: string,
    testCases: TestCase[],
    functionName: string,
    language: Language = 'numpy',
    options: { silent?: boolean } = {}
  ): Promise<TestResult[]> => {
    const silent = options.silent ?? false;
    if (!silent) setOutput([]);
    const outcome: TestOutcome = await pythonRunner.runTests(
      code,
      testCases,
      functionName,
      language,
      silent ? () => {} : (text, stream) => appendLines(text, stream === 'stderr' ? '[Error] ' : '')
    );

    const failAll = (message: string): TestResult[] =>
      testCases.map(tc => ({
        id: tc.id,
        passed: false,
        description: tc.description,
        expected: tc.expected,
        actual: message,
        hidden: tc.hidden,
      }));

    const aborted = abortMessage(outcome);
    if (aborted || outcome.error || !outcome.results) {
      const message = aborted ?? outcome.error ?? 'Test execution error';
      if (!silent) appendLines(message, '[Error] ');
      return failAll(message);
    }

    const byId = new Map(outcome.results.map(r => [r.id, r]));
    const results = testCases.map(tc => {
      const r = byId.get(tc.id);
      return {
        id: tc.id,
        passed: r?.passed ?? false,
        description: tc.description,
        expected: tc.expected,
        actual: r?.actual ?? 'Not run',
        hidden: tc.hidden,
      };
    });
    if (!silent) {
      for (const r of results) {
        const label = r.hidden ? `Test ${r.id} (hidden)` : `Test ${r.id}`;
        if (r.passed) appendLines(`${label}: PASSED`);
        else if (r.hidden) appendLines(`${label}: FAILED`);
        else appendLines(`${label}: FAILED - Expected ${r.expected}, got ${r.actual}`);
      }
    }
    return results;
  }, [appendLines]);

  const stop = useCallback(() => pythonRunner.stop(), []);

  return {
    isLoading: status === 'loading' || status === 'idle',
    isReady: status === 'ready',
    error: status === 'error' ? pythonRunner.getInitError() : null,
    output,
    runCode,
    runTests,
    stop,
    clearOutput,
  };
}
