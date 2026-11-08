/**
 * Runs every reference solution (NumPy and PyTorch tracks) against its own
 * test cases with the same Python harness the browser uses
 * (public/py/runner.py). Skipped when python3 + NumPy aren't available.
 */
import { describe, it, expect } from 'vitest';
import { spawnSync } from 'child_process';
import { mkdtempSync, writeFileSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';
import { allProblems } from '../data/problems';

const hasPython =
  spawnSync('python3', ['-c', 'import numpy'], { encoding: 'utf8' }).status === 0;

describe.skipIf(!hasPython)('Reference solutions', () => {
  it('pass all of their own test cases', () => {
    const dir = mkdtempSync(join(tmpdir(), 'mlcl-'));
    const file = join(dir, 'problems.json');
    writeFileSync(file, JSON.stringify(allProblems));

    const run = spawnSync('python3', ['scripts/validate_solutions.py', file], {
      encoding: 'utf8',
      timeout: 120_000,
    });
    if (run.error) throw run.error;
    const report = JSON.parse(run.stdout) as { checked: number; failures: string[] };

    expect(report.checked).toBe(allProblems.length);
    expect(report.failures).toEqual([]);
  }, 150_000);
});
