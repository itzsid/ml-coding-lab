import { useState, useEffect, useCallback } from 'react';
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
import TestResults from '../components/TestRunner/TestResults';
import SEO from '../components/SEO/SEO';
import { TestResult, TestCase } from '../types';

export default function ProblemPage() {
  const { sectionId, problemId } = useParams<{ sectionId: string; problemId: string }>();
  const { getProblemProgress, saveProblemCode, updateProblemStatus } = useProgress();
  const { isLoading, isReady, output, runTests, clearOutput } = usePyodide();

  const problem = problemId ? getProblemById(problemId) : undefined;
  const section = sections.find(s => s.id === sectionId);

  const [code, setCode] = useState('');
  const [testResults, setTestResults] = useState<TestResult[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [isEditorMaximized, setIsEditorMaximized] = useState(false);
  const [editableTestCases, setEditableTestCases] = useState<TestCase[]>([]);
  const [resetKey, setResetKey] = useState(0);
  const [isMobile, setIsMobile] = useState(() => window.innerWidth < 1024);
  const [mobileTab, setMobileTab] = useState<'problem' | 'code'>('problem');

  // Load saved code or starter code
  useEffect(() => {
    if (problem && sectionId) {
      const savedProgress = getProblemProgress(sectionId, problem.id);
      setCode(savedProgress.code || problem.starterCode);
      setEditableTestCases(problem.testCases);
    }
  }, [problem, sectionId, getProblemProgress]);

  // Auto-save code
  useEffect(() => {
    if (code && problem && sectionId && code !== problem.starterCode) {
      const timer = setTimeout(() => {
        saveProblemCode(sectionId, problem.id, code);
      }, 1000);
      return () => clearTimeout(timer);
    }
  }, [code, problem, sectionId, saveProblemCode]);

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
    if (problem) {
      setCode(problem.starterCode);
      setTestResults([]);
      setEditableTestCases(problem.testCases);
      setResetKey(k => k + 1);
      clearOutput();
    }
  }, [problem, clearOutput]);

  const extractFunctionName = (starterCode: string): string => {
    // Find all function definitions and return the last one
    // This handles cases where helper functions (like sigmoid) are defined before the main function
    const matches = [...starterCode.matchAll(/def\s+(\w+)\s*\(/g)];
    return matches.length > 0 ? matches[matches.length - 1][1] : 'solution';
  };

  const handleRunTests = useCallback(async () => {
    if (!problem || !isReady || !sectionId) return;

    setIsRunning(true);
    setTestResults([]);

    try {
      const functionName = extractFunctionName(problem.starterCode);
      const results = await runTests(code, editableTestCases, functionName);
      setTestResults(results);

      // Check if all tests passed
      const allPassed = results.every(r => r.passed);
      if (allPassed) {
        updateProblemStatus(sectionId, problem.id, 'completed');
      }
    } finally {
      setIsRunning(false);
    }
  }, [problem, isReady, sectionId, code, editableTestCases, runTests, updateProblemStatus]);

  if (!problem || !section) {
    return (
      <div className="text-center py-12">
        <h1 className="text-2xl font-bold text-gray-900 mb-4">Problem Not Found</h1>
        <Link to="/" className="text-primary-600 hover:text-primary-500">
          Return to Home
        </Link>
      </div>
    );
  }

  // Find next problem in section
  const currentIndex = section.problems.indexOf(problem.id);
  const nextProblemId = section.problems[currentIndex + 1];

  // Create description from first paragraph of problem description
  const problemDescriptionText = problem.description
    .replace(/[#*`]/g, '')
    .split('\n')
    .filter(line => line.trim())
    .slice(0, 2)
    .join(' ')
    .substring(0, 155);

  return (
    <div className="h-[calc(100vh-8rem)] flex flex-col -m-6">
      <SEO
        title={`${problem.title} - ${section.title}`}
        description={`${problemDescriptionText}... Practice this ${problem.difficulty} ML coding problem with instant feedback.`}
        canonical={`/problem/${sectionId}/${problemId}`}
        keywords={`${problem.title}, ${section.title}, machine learning, coding practice, ${problem.difficulty}`}
      />
      {/* Problem Header */}
      <div className="flex items-center justify-between px-6 py-3 bg-white border-b border-gray-200">
        <div className="flex items-center gap-3">
          <Link
            to={`/section/${sectionId}`}
            className="text-gray-500 hover:text-gray-900 transition-colors"
          >
            &larr; {section.title}
          </Link>
          <span className="text-gray-300">/</span>
          <span className="text-gray-900 font-medium">{problem.title}</span>
        </div>

        <div className="flex items-center gap-3">
          {!isReady && (
            <span className="text-gray-500 text-sm flex items-center gap-2">
              <div className="w-2 h-2 bg-yellow-500 rounded-full animate-pulse" />
              {isLoading ? 'Loading Python...' : 'Python ready'}
            </span>
          )}
          {isReady && (
            <span className="text-gray-500 text-sm flex items-center gap-2">
              <div className="w-2 h-2 bg-green-500 rounded-full" />
              Python ready
            </span>
          )}
        </div>
      </div>

      {/* Mobile Layout */}
      {isMobile ? (
        <div className="flex flex-col flex-1 overflow-hidden bg-white">
          {/* Mobile Tab Switcher */}
          <div className="flex border-b border-gray-200 bg-gray-50">
            <button
              onClick={() => setMobileTab('problem')}
              className={`flex-1 px-4 py-2.5 text-sm font-medium text-center transition-colors ${
                mobileTab === 'problem'
                  ? 'text-primary-600 border-b-2 border-primary-500 bg-white'
                  : 'text-gray-500 hover:text-gray-700'
              }`}
            >
              Problem
            </button>
            <button
              onClick={() => setMobileTab('code')}
              className={`flex-1 px-4 py-2.5 text-sm font-medium text-center transition-colors ${
                mobileTab === 'code'
                  ? 'text-primary-600 border-b-2 border-primary-500 bg-white'
                  : 'text-gray-500 hover:text-gray-700'
              }`}
            >
              Code
            </button>
          </div>

          {mobileTab === 'problem' ? (
            <div className="flex-1 overflow-y-auto p-4">
              <ProblemDescription problem={problem} />
              <Examples examples={problem.examples} />
              <Hints hints={problem.hints} solution={problem.solution} />
            </div>
          ) : (
            <div className="flex-1 flex flex-col overflow-hidden">
              {/* Action Bar */}
              <div className="flex items-center justify-between px-4 py-2 bg-gray-50 border-b border-gray-200">
                <div className="flex items-center gap-2">
                  <button
                    onClick={handleRunTests}
                    disabled={!isReady || isRunning}
                    className="px-4 py-1.5 bg-primary-500 text-white text-sm font-medium rounded-md hover:bg-primary-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  >
                    {isRunning ? 'Running...' : 'Run Tests'}
                  </button>
                  <button
                    onClick={handleReset}
                    className="px-3 py-1.5 bg-gray-200 text-gray-700 text-sm font-medium rounded-md hover:bg-gray-300 transition-colors"
                  >
                    Reset
                  </button>
                </div>
                {testResults.length > 0 && testResults.every(r => r.passed) && nextProblemId && (
                  <Link
                    to={`/problem/${sectionId}/${nextProblemId}`}
                    className="px-3 py-1.5 bg-green-100 text-green-700 text-sm font-medium rounded-md hover:bg-green-200 transition-colors"
                  >
                    Next &rarr;
                  </Link>
                )}
              </div>
              {/* Editor */}
              <div className="flex-[3] overflow-hidden p-2">
                <CodeEditor
                  value={code}
                  onChange={handleCodeChange}
                  height="100%"
                  onRun={handleRunTests}
                />
              </div>
              {/* Console */}
              <div className="flex-[2] overflow-auto p-2 space-y-2 bg-gray-50 border-t border-gray-200">
                <TestResults results={testResults} isRunning={isRunning} />
                <Console
                  output={output}
                  isLoading={isRunning}
                  testCases={editableTestCases}
                  onTestCasesChange={setEditableTestCases}
                  problemId={problemId}
                  resetKey={resetKey}
                />
              </div>
            </div>
          )}
        </div>
      ) : isEditorMaximized ? (
        /* Maximized Layout - Editor and Console side by side */
        <div className="flex flex-col flex-1 overflow-hidden bg-white">
          {/* Action Bar */}
          <div className="flex items-center justify-between px-4 py-2 bg-gray-50 border-b border-gray-200">
            <div className="flex items-center gap-2">
              <button
                onClick={handleRunTests}
                disabled={!isReady || isRunning}
                className="px-4 py-1.5 bg-primary-500 text-white text-sm font-medium rounded-md hover:bg-primary-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                {isRunning ? 'Running...' : 'Run Tests'}
              </button>
              <span className="text-xs text-gray-400">or</span>
              <div className="flex items-center gap-1 text-gray-500">
                <kbd className="px-1.5 py-0.5 text-xs font-mono bg-gray-100 border border-gray-300 rounded shadow-sm">Shift</kbd>
                <span className="text-xs">+</span>
                <kbd className="px-1.5 py-0.5 text-xs font-mono bg-gray-100 border border-gray-300 rounded shadow-sm">Enter</kbd>
              </div>
            </div>

            <div className="flex items-center gap-2">
              {testResults.length > 0 && testResults.every(r => r.passed) && nextProblemId && (
                <Link
                  to={`/problem/${sectionId}/${nextProblemId}`}
                  className="px-4 py-1.5 bg-green-100 text-green-700 text-sm font-medium rounded-md hover:bg-green-200 transition-colors"
                >
                  Next Problem &rarr;
                </Link>
              )}
              <button
                onClick={() => setIsEditorMaximized(!isEditorMaximized)}
                className="px-4 py-1.5 bg-gray-200 text-gray-700 text-sm font-medium rounded-md hover:bg-gray-300 transition-colors"
                title={isEditorMaximized ? 'Restore layout' : 'Maximize editor'}
              >
                {isEditorMaximized ? '\u229f Restore' : '\u229e Maximize'}
              </button>
              <button
                onClick={handleReset}
                className="px-4 py-1.5 bg-gray-200 text-gray-700 text-sm font-medium rounded-md hover:bg-gray-300 transition-colors"
              >
                Reset
              </button>
            </div>
          </div>

          {/* Maximized: Editor and Console side by side */}
          <Split
            className="flex flex-1 overflow-hidden"
            sizes={[60, 40]}
            minSize={200}
            gutterSize={8}
            gutterAlign="center"
          >
            {/* Code Editor */}
            <div className="h-full overflow-hidden p-4">
              <CodeEditor
                value={code}
                onChange={handleCodeChange}
                height="100%"
                onRun={handleRunTests}
              />
            </div>

            {/* Console and Test Results */}
            <div className="overflow-auto p-4 space-y-4 bg-gray-50">
              <TestResults results={testResults} isRunning={isRunning} />
              <Console
                output={output}
                isLoading={isRunning}
                testCases={editableTestCases}
                onTestCasesChange={setEditableTestCases}
                problemId={problemId}
                resetKey={resetKey}
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
          {/* Left Panel - Problem Description */}
          <div className="problem-panel overflow-y-auto overflow-x-hidden p-6 bg-gray-50 min-w-0">
            <ProblemDescription problem={problem} />
            <Examples examples={problem.examples} />
            <Hints hints={problem.hints} solution={problem.solution} />
          </div>

          {/* Right Panel - Editor and Console */}
          <div className="flex flex-col overflow-hidden bg-white">
            {/* Action Bar */}
            <div className="flex items-center justify-between px-4 py-2 bg-gray-50 border-b border-gray-200">
              <div className="flex items-center gap-2">
                <button
                  onClick={handleRunTests}
                  disabled={!isReady || isRunning}
                  className="px-4 py-1.5 bg-primary-500 text-white text-sm font-medium rounded-md hover:bg-primary-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  {isRunning ? 'Running...' : 'Run Tests'}
                </button>
                <span className="text-xs text-gray-400">or</span>
                <div className="flex items-center gap-1 text-gray-500">
                  <kbd className="px-1.5 py-0.5 text-xs font-mono bg-gray-100 border border-gray-300 rounded shadow-sm">Shift</kbd>
                  <span className="text-xs">+</span>
                  <kbd className="px-1.5 py-0.5 text-xs font-mono bg-gray-100 border border-gray-300 rounded shadow-sm">Enter</kbd>
                </div>
              </div>

              <div className="flex items-center gap-2">
                {testResults.length > 0 && testResults.every(r => r.passed) && nextProblemId && (
                  <Link
                    to={`/problem/${sectionId}/${nextProblemId}`}
                    className="px-4 py-1.5 bg-green-100 text-green-700 text-sm font-medium rounded-md hover:bg-green-200 transition-colors"
                  >
                    Next Problem &rarr;
                  </Link>
                )}
                <button
                  onClick={() => setIsEditorMaximized(!isEditorMaximized)}
                  className="px-4 py-1.5 bg-gray-200 text-gray-700 text-sm font-medium rounded-md hover:bg-gray-300 transition-colors"
                  title={isEditorMaximized ? 'Restore layout' : 'Maximize editor'}
                >
                  {isEditorMaximized ? '\u229f Restore' : '\u229e Maximize'}
                </button>
                <button
                  onClick={handleReset}
                  className="px-4 py-1.5 bg-gray-200 text-gray-700 text-sm font-medium rounded-md hover:bg-gray-300 transition-colors"
                >
                  Reset
                </button>
              </div>
            </div>

            {/* Editor and Results */}
            <Split
              className="flex-1 overflow-hidden"
              direction="vertical"
              sizes={[60, 40]}
              minSize={100}
              gutterSize={8}
            >
              {/* Code Editor */}
              <div className="h-full overflow-hidden p-4">
                <CodeEditor
                  value={code}
                  onChange={handleCodeChange}
                  height="100%"
                  onRun={handleRunTests}
                />
              </div>

              {/* Console and Test Results */}
              <div className="overflow-auto p-4 space-y-4 bg-gray-50">
                <TestResults results={testResults} isRunning={isRunning} />
                <Console
                  output={output}
                  isLoading={isRunning}
                  testCases={editableTestCases}
                  onTestCasesChange={setEditableTestCases}
                  problemId={problemId}
                  resetKey={resetKey}
                />
              </div>
            </Split>
          </div>
        </Split>
      )}
    </div>
  );
}
