export type Difficulty = 'easy' | 'medium' | 'hard';

export type ProblemStatus = 'not_started' | 'in_progress' | 'completed';

export interface Example {
  input: string;
  output: string;
  explanation?: string;
}

export interface TestCase {
  id: string;
  description: string;
  input: string;
  expected: string;
  hidden: boolean;
}

export interface TestResult {
  id: string;
  passed: boolean;
  description: string;
  expected: string;
  actual: string;
  hidden: boolean;
}

export type Language = 'numpy' | 'pytorch';

/** A solution track for a problem in another framework (e.g. PyTorch). */
export interface LanguageVariant {
  starterCode: string;
  solution: string;
  /** Expression-based tests only (they must call the function explicitly). */
  testCases: TestCase[];
  /** Optional framework-specific hints; falls back to the problem's hints. */
  hints?: string[];
  /** Optional short note shown above the editor, e.g. which torch APIs are allowed. */
  note?: string;
}

export interface Problem {
  id: string;
  title: string;
  section: string;
  difficulty: Difficulty;
  description: string;
  examples: Example[];
  starterCode: string;
  testCases: TestCase[];
  hints: string[];
  solution: string;
  /** PyTorch track, attached from src/data/problems/pytorch/. */
  pytorch?: LanguageVariant;
}

export interface Section {
  id: string;
  title: string;
  description: string;
  introduction: string;
  icon: string;
  problems: string[];
  /** Solution track to open by default (e.g. PyTorch Essentials opens in PyTorch). */
  defaultLanguage?: Language;
}

export interface ProblemProgress {
  status: ProblemStatus;
  lastAttempt?: string;
  /** Saved NumPy-track code (kept under this name for backward compatibility). */
  code?: string;
  /** Saved code for other tracks, e.g. { pytorch: '...' }. */
  codes?: Partial<Record<Language, string>>;
  /** Track(s) the problem was solved in. */
  solvedIn?: Language[];
  /** The reference solution was revealed before/after solving. */
  solutionViewed?: boolean;
}

export interface SectionProgress {
  [problemId: string]: ProblemProgress;
}

export interface Progress {
  [sectionId: string]: SectionProgress;
}

export interface PyodideInterface {
  runPythonAsync: (code: string) => Promise<unknown>;
  loadPackage: (packages: string | string[]) => Promise<void>;
  globals: Map<string, unknown>;
}
