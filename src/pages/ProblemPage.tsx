import { useState, useEffect, useCallback, useRef } from 'react';
import { useParams, Link } from 'react-router-dom';
import Split from 'react-split';
import { getProblemById } from '../data/problems';
import { sections } from '../data/sections';
import { useProgress } from '../context/ProgressContext';
import { usePyodide } from '../hooks/usePyodide';
import CodeEditor from '../components/CodeEditor/CodeEditor';
import Console from '../components/Console/Console';
import ProblemDescription from '../components/ProblemView/ProblemDescription';
import Examples from '../components/ProblemView/Examples';
import Hints from '../components/ProblemView/Hints';
import SEO from '../components/SEO/SEO';
import { TestResult, TestCase, Language } from '../types';
import SuccessBanner from '../components/SuccessBanner/SuccessBanner';
import { useDarkMode } from '../hooks/useDarkMode';

const LANGUAGE_KEY = 'ml-lab-language';

const LANGUAGE_LABELS: Record<Language, string> = { numpy: 'NumPy', pytorch: 'PyTorch' };

export default function ProblemPage() {
  const { sectionId, problemId } = useParams<{ sectionId: string; problemId: string }>();
  const { getProblemProgress, saveProblemCode, updateProblemStatus, markSolutionViewed } = useProgress();
  const { isLoading, isReady, output, runTests, stop, clearOutput } = usePyodide();

  const problem = problemId ? getProblemById(problemId) : undefined;
  const section = sections.find(s => s.id === sectionId);

  // Solution track: NumPy (default) or PyTorch, remembered across problems.
  const [language, setLanguage] = useState<Language>(() =>
    localStorage.getItem(LANGUAGE_KEY) === 'pytorch' ? 'pytorch' : 'numpy'
  );
  const activeLanguage: Language = language === 'pytorch' && problem?.pytorch ? 'pytorch' : 'numpy';
  const track = activeLanguage === 'pytorch' && problem?.pytorch ? problem.pytorch : problem;
  const trackHints = (activeLanguage === 'pytorch' && problem?.pytorch?.hints) || problem?.hints || [];

  const [code, setCode] = useState('');
  const [testResults, setTestResults] = useState<TestResult[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [isEditorMaximized, setIsEditorMaximized] = useState(false);
  const [editableTestCases, setEditableTestCases] = useState<TestCase[]>([]);
  const [resetKey, setResetKey] = useState(0);
  const [showShortcutHint, setShowShortcutHint] = useState(
    () => !localStorage.getItem('ml-lab-shortcut-hint-dismissed')
  );
  const [showCelebration, setShowCelebration] = useState(false);
  // Shown when edited tests passed but the original tests didn't.
  const [completionNote, setCompletionNote] = useState<string | null>(null);
  const wasCompletedRef = useRef(false);
  const [descriptionTab, setDescriptionTab] = useState<'description' | 'examples' | 'hints'>('description');
  const [isMobile, setIsMobile] = useState(() => window.innerWidth < 1024);
  const [mobileTab, setMobileTab] = useState<'problem' | 'code'>('problem');
  const { isDark } = useDarkMode();

  // Load saved code or starter code — only when the problem itself changes.
  // getProblemProgress changes identity on every progress update (autosave,
  // test completion), so without this guard each autosave would reset the
  // edited test cases and jump the description tab back to "Description".
  const loadedProblemKeyRef = useRef<string | null>(null);
  const loadedProblemIdRef = useRef<string | null>(null);
  useEffect(() => {
    if (problem && track && sectionId) {
      const key = `${sectionId}/${problem.id}/${activeLanguage}`;
      if (loadedProblemKeyRef.current === key) return;
      loadedProblemKeyRef.current = key;
      const savedProgress = getProblemProgress(sectionId, problem.id);
      const savedCode = activeLanguage === 'numpy' ? savedProgress.code : savedProgress.codes[activeLanguage];
      setCode(savedCode || track.starterCode);
      setEditableTestCases(track.testCases);
      setTestResults([]);
      setCompletionNote(null);
      setResetKey(k => k + 1);
      // Only a new problem (not a language switch) resets the left panel.
      if (loadedProblemIdRef.current !== problem.id) {
        loadedProblemIdRef.current = problem.id;
        setDescriptionTab('description');
      }
    }
  }, [problem, track, activeLanguage, sectionId, getProblemProgress]);

  // Track if problem was already completed to prevent repeat celebrations
  useEffect(() => {
    if (problem && sectionId) {
      const savedProgress = getProblemProgress(sectionId, problem.id);
      wasCompletedRef.current = savedProgress.status === 'completed';
    }
  }, [problem?.id, sectionId, getProblemProgress]);

  // Auto-save code (per language). The ref guards against saving code from the
  // previous track under the new one right after a language switch.
  const codeTrackRef = useRef<string | null>(null);
  useEffect(() => {
    codeTrackRef.current = loadedProblemKeyRef.current;
  }, [code]);
  useEffect(() => {
    if (code && problem && track && sectionId && code !== track.starterCode
        && codeTrackRef.current === `${sectionId}/${problem.id}/${activeLanguage}`) {
      const timer = setTimeout(() => {
        saveProblemCode(sectionId, problem.id, code, activeLanguage);
      }, 1000);
      return () => clearTimeout(timer);
    }
  }, [code, problem, track, activeLanguage, sectionId, saveProblemCode]);

  const handleLanguageChange = useCallback((next: Language) => {
    if (next === activeLanguage || !problem || !track || !sectionId) return;
    // Flush unsaved edits for the current track before switching.
    if (code && code !== track.starterCode) saveProblemCode(sectionId, problem.id, code, activeLanguage);
    localStorage.setItem(LANGUAGE_KEY, next);
    setLanguage(next);
  }, [activeLanguage, problem, track, sectionId, code, saveProblemCode]);

  // Auto-dismiss shortcut hint after 10 seconds
  useEffect(() => {
    if (showShortcutHint) {
      const timer = setTimeout(() => {
        setShowShortcutHint(false);
        localStorage.setItem('ml-lab-shortcut-hint-dismissed', 'true');
      }, 10000);
      return () => clearTimeout(timer);
    }
  }, [showShortcutHint]);

  const dismissShortcutHint = useCallback(() => {
    setShowShortcutHint(false);
    localStorage.setItem('ml-lab-shortcut-hint-dismissed', 'true');
  }, []);

  // Detect mobile viewport
  useEffect(() => {
    const handleResize = () => setIsMobile(window.innerWidth < 1024);
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const handleCodeChange = useCallback((newCode: string) => {
    setCode(newCode);
  }, []);

  const handleReset = useCallback(() => {
    if (track) {
      const hasChanges = code !== track.starterCode;
      if (!hasChanges || window.confirm('Reset your code? Your changes will be lost.')) {
        setCode(track.starterCode);
        setTestResults([]);
        setCompletionNote(null);
        setEditableTestCases(track.testCases);
        setResetKey(k => k + 1);
        clearOutput();
      }
    }
  }, [track, code, clearOutput]);

  const extractFunctionName = (starterCode: string): string => {
    const matches = [...starterCode.matchAll(/def\s+(\w+)\s*\(/g)];
    return matches.length > 0 ? matches[matches.length - 1][1] : 'solution';
  };

  const handleRunTests = useCallback(async () => {
    if (!problem || !track || !isReady || !sectionId || isRunning) return;

    setIsRunning(true);
    setTestResults([]);
    setCompletionNote(null);

    try {
      const functionName = extractFunctionName(track.starterCode);
      const results = await runTests(code, editableTestCases, functionName, activeLanguage);
      setTestResults(results);

      if (!results.every(r => r.passed)) return;

      // Completion always counts against the original tests, so editing an
      // expected value can't mark a problem solved.
      const testsEdited = JSON.stringify(editableTestCases) !== JSON.stringify(track.testCases);
      if (testsEdited) {
        const original = await runTests(code, track.testCases, functionName, activeLanguage, { silent: true });
        if (!original.every(r => r.passed)) {
          setCompletionNote('Your edited tests pass, but the original tests don\'t yet. Reset the tests to see which ones fail.');
          return;
        }
      }

      updateProblemStatus(sectionId, problem.id, 'completed', activeLanguage);
      if (!wasCompletedRef.current) {
        setShowCelebration(true);
        wasCompletedRef.current = true;
      }
    } finally {
      setIsRunning(false);
    }
  }, [problem, track, isReady, isRunning, sectionId, code, editableTestCases, activeLanguage, runTests, updateProblemStatus]);

  const handleShowSolution = useCallback(() => {
    if (problem && sectionId) markSolutionViewed(sectionId, problem.id);
  }, [problem, sectionId, markSolutionViewed]);

  if (!problem || !section || !track) {
    return (
      <div className="text-center py-12">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-dark-100 mb-4">Problem Not Found</h1>
        <Link to="/" className="text-primary-500 hover:text-primary-400">
          Return to Home
        </Link>
      </div>
    );
  }

  // Find prev/next problems in section
  const currentIndex = section.problems.indexOf(problem.id);
  const prevProblemId = currentIndex > 0 ? section.problems[currentIndex - 1] : undefined;
  const nextProblemId = section.problems[currentIndex + 1];

  // Create description from first paragraph of problem description
  const problemDescriptionText = problem.description
    .replace(/[#*`]/g, '')
    .split('\n')
    .filter(line => line.trim())
    .slice(0, 2)
    .join(' ')
    .substring(0, 155);

  const shortcutHintEl = showShortcutHint ? (
    <div className="absolute top-full left-1/2 -translate-x-1/2 mt-2 px-3 py-2 bg-dark-800 dark:bg-dark-600 text-white text-xs rounded-lg shadow-lg whitespace-nowrap z-50">
      <div className="absolute -top-1 left-1/2 -translate-x-1/2 w-2 h-2 bg-dark-800 dark:bg-dark-600 rotate-45" />
      <span>Pro tip: </span>
      <span className="font-medium">Shift+Enter</span>
      <span> runs tests</span>
      <button
        onClick={dismissShortcutHint}
        className="ml-2 text-gray-400 hover:text-white underline"
      >
        Got it
      </button>
    </div>
  ) : null;

  const runButton = isRunning ? (
    <button
      onClick={stop}
      className="px-4 py-1.5 bg-red-500/90 text-white text-sm font-medium rounded-lg hover:bg-red-600 transition-colors flex items-center gap-2"
      title="Stop running code"
    >
      <span className="w-2.5 h-2.5 bg-white rounded-sm" aria-hidden="true" />
      Stop
    </button>
  ) : (
    <button
      onClick={handleRunTests}
      disabled={!isReady}
      className="px-4 py-1.5 bg-primary-500 text-white text-sm font-medium rounded-lg hover:bg-primary-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
    >
      {isReady ? 'Run Tests' : 'Loading Python…'}
    </button>
  );

  const languageToggle = problem.pytorch ? (
    <div
      role="radiogroup"
      aria-label="Solution language"
      className="flex items-center p-0.5 bg-gray-200 dark:bg-dark-700 border border-gray-300 dark:border-dark-500 rounded-lg"
    >
      {(['numpy', 'pytorch'] as Language[]).map(lang => (
        <button
          key={lang}
          role="radio"
          aria-checked={activeLanguage === lang}
          onClick={() => handleLanguageChange(lang)}
          disabled={isRunning}
          className={`px-2.5 py-1 text-xs font-medium rounded-md transition-colors disabled:cursor-not-allowed ${
            activeLanguage === lang
              ? 'bg-white dark:bg-dark-500 text-gray-900 dark:text-dark-100 shadow-sm'
              : 'text-gray-500 dark:text-dark-200 hover:text-gray-800 dark:hover:text-dark-100'
          }`}
        >
          {LANGUAGE_LABELS[lang]}
        </button>
      ))}
    </div>
  ) : null;

  const trackNotes = (completionNote || (activeLanguage === 'pytorch' && problem.pytorch)) ? (
    <div className="px-4 py-1.5 space-y-1 text-xs border-b border-gray-200 dark:border-dark-500 bg-gray-50 dark:bg-dark-900">
      {activeLanguage === 'pytorch' && problem.pytorch && (
        <p className="text-gray-500 dark:text-dark-200">
          <span className="font-medium text-gray-700 dark:text-dark-100">PyTorch track</span>
          {' '}runs on an in-browser, NumPy-backed subset of the torch API.
          {problem.pytorch.note ? ` ${problem.pytorch.note}` : ''}
        </p>
      )}
      {completionNote && <p className="text-amber-600 dark:text-amber-400">{completionNote}</p>}
    </div>
  ) : null;

  const actionBar = (
    <div className="flex items-center justify-between px-4 py-2 bg-gray-50 dark:bg-dark-800 border-b border-gray-200 dark:border-dark-500">
      <div className="flex items-center gap-2">
        {languageToggle}
        {runButton}
        <span className="text-xs text-gray-400 dark:text-dark-300">or</span>
        <div className="relative">
          <div className="flex items-center gap-1 text-gray-500 dark:text-dark-200">
            <kbd className="px-1.5 py-0.5 text-xs font-mono bg-gray-100 dark:bg-dark-600 border border-gray-300 dark:border-dark-500 rounded shadow-sm">Shift</kbd>
            <span className="text-xs">+</span>
            <kbd className="px-1.5 py-0.5 text-xs font-mono bg-gray-100 dark:bg-dark-600 border border-gray-300 dark:border-dark-500 rounded shadow-sm">Enter</kbd>
          </div>
          {shortcutHintEl}
        </div>
      </div>

      <div className="flex items-center gap-2">
        {testResults.length > 0 && testResults.every(r => r.passed) && !completionNote && nextProblemId && (
          <Link
            to={`/problem/${sectionId}/${nextProblemId}`}
            className="px-4 py-1.5 bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400 text-sm font-medium rounded-lg hover:bg-green-200 dark:hover:bg-green-900/50 transition-colors"
          >
            Next Problem &rarr;
          </Link>
        )}
        <button
          onClick={() => setIsEditorMaximized(!isEditorMaximized)}
          className="px-4 py-1.5 bg-gray-200 dark:bg-dark-600 text-gray-700 dark:text-dark-200 text-sm font-medium rounded-lg hover:bg-gray-300 dark:hover:bg-dark-500 transition-colors"
          title={isEditorMaximized ? 'Restore layout' : 'Maximize editor'}
        >
          {isEditorMaximized ? '\u229f Restore' : '\u229e Maximize'}
        </button>
        <button
          onClick={handleReset}
          className="px-4 py-1.5 bg-gray-200 dark:bg-dark-600 text-gray-700 dark:text-dark-200 text-sm font-medium rounded-lg hover:bg-gray-300 dark:hover:bg-dark-500 transition-colors"
        >
          Reset
        </button>
      </div>
    </div>
  );

  return (
    <div className="h-[calc(100dvh-4rem)] flex flex-col -m-6">
      <SEO
        title={`${problem.title} - ${section.title}`}
        description={`${problemDescriptionText}... Practice this ${problem.difficulty} ML coding problem with instant feedback.`}
        canonical={`/problem/${sectionId}/${problemId}`}
        keywords={`${problem.title}, ${section.title}, machine learning, coding practice, ${problem.difficulty}`}
      />
      <SuccessBanner show={showCelebration} onDismiss={() => setShowCelebration(false)} />
      {/* Problem Header */}
      <div className="flex flex-wrap items-center justify-between gap-x-3 gap-y-1 px-4 sm:px-6 py-3 bg-white dark:bg-dark-800 border-b border-gray-200 dark:border-dark-500">
        <div className="flex items-center gap-3 min-w-0">
          <Link
            to={`/section/${sectionId}`}
            className="shrink-0 text-gray-500 dark:text-dark-200 hover:text-gray-900 dark:hover:text-dark-100 transition-colors"
          >
            &larr; <span className="hidden sm:inline">{section.title}</span>
          </Link>
          <span className="hidden sm:inline text-gray-300 dark:text-dark-500">/</span>
          <span className="text-gray-900 dark:text-dark-100 font-medium truncate">{problem.title}</span>
        </div>

        <div className="flex items-center gap-3">
          {/* Problem Navigation */}
          <div className="flex items-center gap-1">
            {prevProblemId ? (
              <Link
                to={`/problem/${sectionId}/${prevProblemId}`}
                className="px-2 py-1 text-sm text-gray-600 dark:text-dark-200 hover:text-gray-900 dark:hover:text-dark-100 hover:bg-gray-100 dark:hover:bg-dark-600 rounded transition-colors"
              >
                &larr; Prev
              </Link>
            ) : (
              <span className="px-2 py-1 text-sm text-gray-300 dark:text-dark-400 cursor-default">&larr; Prev</span>
            )}
            <span className="text-xs text-gray-400 dark:text-dark-300 font-mono">
              ({currentIndex + 1}/{section.problems.length})
            </span>
            {nextProblemId ? (
              <Link
                to={`/problem/${sectionId}/${nextProblemId}`}
                className="px-2 py-1 text-sm text-gray-600 dark:text-dark-200 hover:text-gray-900 dark:hover:text-dark-100 hover:bg-gray-100 dark:hover:bg-dark-600 rounded transition-colors"
              >
                Next &rarr;
              </Link>
            ) : (
              <span className="px-2 py-1 text-sm text-gray-300 dark:text-dark-400 cursor-default">Next &rarr;</span>
            )}
          </div>

          <span className="hidden sm:inline text-gray-300 dark:text-dark-500">|</span>

          {!isReady && (
            <span className="text-gray-500 dark:text-dark-200 text-sm hidden sm:flex items-center gap-2">
              <div className="w-2 h-2 bg-yellow-500 rounded-full animate-pulse" />
              {isLoading ? 'Loading Python...' : 'Python ready'}
            </span>
          )}
          {isReady && (
            <span className="text-gray-500 dark:text-dark-200 text-sm hidden sm:flex items-center gap-2">
              <div className="w-2 h-2 bg-accent-500 rounded-full" />
              Python ready
            </span>
          )}
        </div>
      </div>

      {/* Mobile Layout */}
      {isMobile ? (
        <div className="flex flex-col flex-1 overflow-hidden bg-white dark:bg-dark-800">
          {/* Mobile Tab Switcher */}
          <div className="flex border-b border-gray-200 dark:border-dark-500 bg-gray-50 dark:bg-dark-900">
            <button
              onClick={() => setMobileTab('problem')}
              className={`flex-1 px-4 py-2.5 text-sm font-medium text-center transition-colors ${
                mobileTab === 'problem'
                  ? 'text-primary-600 dark:text-primary-300 border-b-2 border-primary-500 bg-white dark:bg-dark-800'
                  : 'text-gray-500 dark:text-dark-200 hover:text-gray-700 dark:hover:text-dark-100'
              }`}
            >
              Problem
            </button>
            <button
              onClick={() => setMobileTab('code')}
              className={`flex-1 px-4 py-2.5 text-sm font-medium text-center transition-colors ${
                mobileTab === 'code'
                  ? 'text-primary-600 dark:text-primary-300 border-b-2 border-primary-500 bg-white dark:bg-dark-800'
                  : 'text-gray-500 dark:text-dark-200 hover:text-gray-700 dark:hover:text-dark-100'
              }`}
            >
              Code
            </button>
          </div>

          {mobileTab === 'problem' ? (
            <div className="flex-1 overflow-y-auto p-4">
              <ProblemDescription problem={problem} />
              <Examples examples={problem.examples} />
              <Hints
                key={`${problem.id}-${activeLanguage}`}
                hints={trackHints}
                solution={track.solution}
                onShowSolution={handleShowSolution}
                onLoadToEditor={(s) => { setCode(s); setMobileTab('code'); }}
              />
            </div>
          ) : (
            <div className="flex-1 flex flex-col overflow-hidden">
              {/* Action Bar */}
              <div className="flex items-center justify-between px-4 py-2 bg-gray-50 dark:bg-dark-900 border-b border-gray-200 dark:border-dark-500">
                <div className="flex items-center gap-2">
                  {languageToggle}
                  {runButton}
                  <button
                    onClick={handleReset}
                    className="px-3 py-1.5 bg-gray-200 dark:bg-dark-600 text-gray-700 dark:text-dark-200 text-sm font-medium rounded-lg hover:bg-gray-300 dark:hover:bg-dark-500 transition-colors"
                  >
                    Reset
                  </button>
                </div>
                {testResults.length > 0 && testResults.every(r => r.passed) && !completionNote && nextProblemId && (
                  <Link
                    to={`/problem/${sectionId}/${nextProblemId}`}
                    className="px-3 py-1.5 bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400 text-sm font-medium rounded-lg hover:bg-green-200 dark:hover:bg-green-900/50 transition-colors"
                  >
                    Next &rarr;
                  </Link>
                )}
              </div>
              {trackNotes}
              {/* Editor */}
              <div className="flex-[3] overflow-hidden p-2">
                <CodeEditor
                  value={code}
                  onChange={handleCodeChange}
                  height="100%"
                  onRun={handleRunTests}
                  darkMode={isDark}
                />
              </div>
              {/* Console */}
              <div className="flex-[2] overflow-auto p-2 space-y-2 bg-gray-50 dark:bg-dark-900 border-t border-gray-200 dark:border-dark-500">
                <Console
                  output={output}
                  isLoading={isRunning}
                  testCases={editableTestCases}
                  onTestCasesChange={setEditableTestCases}
                  problemId={problemId}
                  resetKey={resetKey}
                  testResults={testResults}
                />
              </div>
            </div>
          )}
        </div>
      ) : isEditorMaximized ? (
        /* Maximized Layout - Editor and Console side by side */
        <div className="flex flex-col flex-1 overflow-hidden bg-white dark:bg-dark-800">
          {actionBar}
          {trackNotes}

          <Split
            className="flex flex-1 overflow-hidden"
            sizes={[60, 40]}
            minSize={200}
            gutterSize={8}
            gutterAlign="center"
          >
            <div className="h-full overflow-hidden p-4">
              <CodeEditor
                value={code}
                onChange={handleCodeChange}
                height="100%"
                onRun={handleRunTests}
                darkMode={isDark}
              />
            </div>

            <div className="overflow-auto p-4 bg-gray-50 dark:bg-dark-900">
              <Console
                output={output}
                isLoading={isRunning}
                testCases={editableTestCases}
                onTestCasesChange={setEditableTestCases}
                problemId={problemId}
                resetKey={resetKey}
                testResults={testResults}
              />
            </div>
          </Split>
        </div>
      ) : (
        /* Normal Layout - Problem description on left, Editor/Console stacked on right */
        <Split
          className="flex flex-1 overflow-hidden"
          sizes={[40, 60]}
          minSize={300}
          gutterSize={8}
          gutterAlign="center"
        >
          {/* Left Panel - Tabbed Problem Description */}
          <div className="problem-panel flex flex-col overflow-hidden bg-gray-50 dark:bg-dark-900 min-w-0">
            {/* Tab Bar */}
            <div className="flex overflow-x-auto whitespace-nowrap border-b border-gray-200 dark:border-dark-500 bg-white dark:bg-dark-800 px-4 shrink-0">
              <button
                onClick={() => setDescriptionTab('description')}
                className={`whitespace-nowrap shrink-0 px-4 py-2.5 text-sm font-medium border-b-2 transition-colors ${
                  descriptionTab === 'description'
                    ? 'border-primary-500 text-primary-600 dark:text-primary-300'
                    : 'border-transparent text-gray-500 dark:text-dark-200 hover:text-gray-700 dark:hover:text-dark-100 hover:border-gray-300 dark:hover:border-dark-400'
                }`}
              >
                Description
              </button>
              <button
                onClick={() => setDescriptionTab('examples')}
                className={`whitespace-nowrap shrink-0 px-4 py-2.5 text-sm font-medium border-b-2 transition-colors ${
                  descriptionTab === 'examples'
                    ? 'border-primary-500 text-primary-600 dark:text-primary-300'
                    : 'border-transparent text-gray-500 dark:text-dark-200 hover:text-gray-700 dark:hover:text-dark-100 hover:border-gray-300 dark:hover:border-dark-400'
                }`}
              >
                Examples
              </button>
              <button
                onClick={() => setDescriptionTab('hints')}
                className={`whitespace-nowrap shrink-0 px-4 py-2.5 text-sm font-medium border-b-2 transition-colors ${
                  descriptionTab === 'hints'
                    ? 'border-primary-500 text-primary-600 dark:text-primary-300'
                    : 'border-transparent text-gray-500 dark:text-dark-200 hover:text-gray-700 dark:hover:text-dark-100 hover:border-gray-300 dark:hover:border-dark-400'
                }`}
              >
                Hints & Solution
              </button>
            </div>
            {/* Tab Content */}
            <div className="flex-1 overflow-y-auto overflow-x-hidden p-6">
              {descriptionTab === 'description' && (
                <ProblemDescription problem={problem} />
              )}
              {descriptionTab === 'examples' && (
                <div>
                  <Examples examples={problem.examples} />
                </div>
              )}
              {descriptionTab === 'hints' && (
                <div>
                  <Hints
                    key={`${problem.id}-${activeLanguage}`}
                    hints={trackHints}
                    solution={track.solution}
                    onShowSolution={handleShowSolution}
                    onLoadToEditor={setCode}
                  />
                </div>
              )}
            </div>
          </div>

          {/* Right Panel - Editor and Console */}
          <div className="flex flex-col overflow-hidden bg-white dark:bg-dark-800">
            {actionBar}
            {trackNotes}

            <Split
              className="flex-1 overflow-hidden"
              direction="vertical"
              sizes={[60, 40]}
              minSize={100}
              gutterSize={8}
            >
              <div className="h-full overflow-hidden p-4">
                <CodeEditor
                  value={code}
                  onChange={handleCodeChange}
                  height="100%"
                  onRun={handleRunTests}
                  darkMode={isDark}
                />
              </div>

              <div className="overflow-auto p-4 bg-gray-50 dark:bg-dark-900">
                <Console
                  output={output}
                  isLoading={isRunning}
                  testCases={editableTestCases}
                  onTestCasesChange={setEditableTestCases}
                  problemId={problemId}
                  resetKey={resetKey}
                  testResults={testResults}
                />
              </div>
            </Split>
          </div>
        </Split>
      )}
    </div>
  );
}
