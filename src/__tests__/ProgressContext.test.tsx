import { describe, it, expect, beforeEach, vi } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { ProgressProvider, useProgress } from '../context/ProgressContext';
import { ReactNode } from 'react';

// Mock localStorage
const localStorageMock = (() => {
  let store: Record<string, string> = {};
  return {
    getItem: vi.fn((key: string) => store[key] || null),
    setItem: vi.fn((key: string, value: string) => {
      store[key] = value;
    }),
    removeItem: vi.fn((key: string) => {
      delete store[key];
    }),
    clear: vi.fn(() => {
      store = {};
    }),
  };
})();

Object.defineProperty(window, 'localStorage', {
  value: localStorageMock,
});

const wrapper = ({ children }: { children: ReactNode }) => (
  <ProgressProvider>{children}</ProgressProvider>
);

describe('ProgressContext', () => {
  beforeEach(() => {
    localStorageMock.clear();
    vi.clearAllMocks();
  });

  it('should initialize with empty progress', () => {
    const { result } = renderHook(() => useProgress(), { wrapper });
    expect(result.current.progress).toEqual({});
  });

  it('should update problem status', () => {
    const { result } = renderHook(() => useProgress(), { wrapper });

    act(() => {
      result.current.updateProblemStatus('python-basics', 'problem-1', 'completed');
    });

    const progress = result.current.getProblemProgress('python-basics', 'problem-1');
    expect(progress.status).toBe('completed');
  });

  it('should save problem code', () => {
    const { result } = renderHook(() => useProgress(), { wrapper });

    act(() => {
      result.current.saveProblemCode('python-basics', 'problem-1', 'print("hello")');
    });

    const progress = result.current.getProblemProgress('python-basics', 'problem-1');
    expect(progress.code).toBe('print("hello")');
    expect(progress.status).toBe('in_progress');
  });

  it('should calculate section progress', () => {
    const { result } = renderHook(() => useProgress(), { wrapper });

    act(() => {
      result.current.updateProblemStatus('python-basics', 'problem-1', 'completed');
    });

    const progress = result.current.getSectionProgress('python-basics', 3);
    expect(progress).toBe(33); // 1 out of 3 completed
  });

  it('should calculate overall progress', () => {
    const { result } = renderHook(() => useProgress(), { wrapper });

    act(() => {
      result.current.updateProblemStatus('section-1', 'problem-1', 'completed');
      result.current.updateProblemStatus('section-2', 'problem-1', 'completed');
    });

    const sections = [
      { id: 'section-1', problemCount: 2 },
      { id: 'section-2', problemCount: 2 },
    ];

    const progress = result.current.getOverallProgress(sections);
    expect(progress).toBe(50); // 2 out of 4 completed
  });

  it('should reset progress', () => {
    const { result } = renderHook(() => useProgress(), { wrapper });

    act(() => {
      result.current.updateProblemStatus('python-basics', 'problem-1', 'completed');
      result.current.resetProgress();
    });

    expect(result.current.progress).toEqual({});
  });
});

describe('migrateProgress', () => {
  it('re-files entries for problems that moved to another section', async () => {
    const { migrateProgress } = await import('../context/ProgressContext');
    const migrated = migrateProgress({
      'python-basics': { 'numpy-array-sum': { status: 'completed', code: 'x' } },
      'deep-learning': { perceptron: { status: 'in_progress' } },
      'reinforcement-learning': {
        'rl-dpo-loss': { status: 'completed' },
        'rl-discounted-return': { status: 'completed' },
      },
      'numpy-fundamentals': { 'numpy-indexing': { status: 'completed' } },
    });
    expect(migrated['python-basics']).toBeUndefined();
    expect(migrated['deep-learning']).toBeUndefined();
    expect(migrated['numpy-fundamentals']).toEqual({
      'numpy-indexing': { status: 'completed' },
      'numpy-array-sum': { status: 'completed', code: 'x' },
    });
    expect(migrated['neural-networks']).toEqual({ perceptron: { status: 'in_progress' } });
    expect(migrated['post-training']).toEqual({ 'rl-dpo-loss': { status: 'completed' } });
    expect(migrated['reinforcement-learning']).toEqual({ 'rl-discounted-return': { status: 'completed' } });
  });

  it('keeps entries for unknown problems where they were', async () => {
    const { migrateProgress } = await import('../context/ProgressContext');
    expect(migrateProgress({ old: { gone: { status: 'completed' } } })).toEqual({
      old: { gone: { status: 'completed' } },
    });
  });
});
