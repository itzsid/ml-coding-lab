import { useRef, useEffect, useState } from 'react';
import { TestCase } from '../../types';

interface ConsoleProps {
  output: string[];
  isLoading?: boolean;
  testCases?: TestCase[];
  onTestCasesChange?: (testCases: TestCase[]) => void;
  problemId?: string; // Used to detect problem changes
  resetKey?: number; // Incremented when reset button is clicked
}

function formatTestCasesForEditing(testCases: TestCase[]): string {
  const visibleTests = testCases.filter(tc => !tc.hidden);
  if (visibleTests.length === 0) return '';
  return visibleTests.map((tc, idx) => (
    `# Test ${idx + 1}: ${tc.description}\n` +
    `input: ${tc.input}\n` +
    `expected: ${tc.expected}`
  )).join('\n\n');
}

function parseTestCasesFromText(text: string, originalTestCases: TestCase[]): TestCase[] {
  const hiddenTests = originalTestCases.filter(tc => tc.hidden);

  // If text is empty or whitespace only, return just hidden tests
  if (!text.trim()) {
    return hiddenTests;
  }

  const blocks = text.split(/\n\n+/).filter(b => b.trim());

  const parsedTests: TestCase[] = blocks.map((block, idx) => {
    const lines = block.split('\n');
    let description = `Test ${idx + 1}`;
    let input = '';
    let expected = '';

    for (const line of lines) {
      if (line.startsWith('# Test')) {
        const match = line.match(/# Test \d+: (.+)/);
        if (match) description = match[1];
      } else if (line.startsWith('input:')) {
        input = line.replace('input:', '').trim();
      } else if (line.startsWith('expected:')) {
        expected = line.replace('expected:', '').trim();
      }
    }

    return {
      id: String(idx + 1),
      description,
      input,
      expected,
      hidden: false
    };
  }).filter(tc => tc.input && tc.expected);

  return [...parsedTests, ...hiddenTests];
}

export default function Console({
  output,
  isLoading = false,
  testCases,
  onTestCasesChange,
  problemId,
  resetKey = 0
}: ConsoleProps) {
  const bottomRef = useRef<HTMLDivElement>(null);
  const [activeTab, setActiveTab] = useState<'tests' | 'output'>('tests');
  const [testCaseText, setTestCaseText] = useState('');
  const lastProblemIdRef = useRef<string | undefined>(undefined);
  const lastResetKeyRef = useRef<number>(0);

  const visibleTestCount = testCases?.filter(tc => !tc.hidden).length ?? 0;
  const hiddenCount = testCases?.filter(tc => tc.hidden).length ?? 0;

  // Reset test case text when problem changes, reset button is clicked, or initial load
  useEffect(() => {
    if (testCases && testCases.length > 0) {
      const problemChanged = problemId !== lastProblemIdRef.current;
      const resetClicked = resetKey !== lastResetKeyRef.current;
      const visibleTests = testCases.filter(tc => !tc.hidden);
      const needsInitialLoad = testCaseText === '' && visibleTests.length > 0;

      if (problemChanged || resetClicked || needsInitialLoad) {
        setTestCaseText(formatTestCasesForEditing(testCases));
        lastProblemIdRef.current = problemId;
        lastResetKeyRef.current = resetKey;
        // Reset to tests tab when problem changes
        if (problemChanged) {
          setActiveTab('tests');
        }
      }
    }
  }, [testCases, problemId, resetKey, testCaseText]);

  // Switch to output tab when there's new output
  useEffect(() => {
    if (output.length > 0) {
      setActiveTab('output');
    }
  }, [output]);

  useEffect(() => {
    if (activeTab === 'output') {
      bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [output, activeTab]);

  const handleTestCaseTextChange = (newText: string) => {
    setTestCaseText(newText);
    if (onTestCasesChange && testCases) {
      const parsed = parseTestCasesFromText(newText, testCases);
      onTestCasesChange(parsed);
    }
  };

  const showTestsTab = testCases && visibleTestCount > 0;

  return (
    <div className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded-lg h-full flex flex-col shadow-sm">
      <div className="flex items-center justify-between px-2 py-1 border-b border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800">
        <div className="flex">
          {showTestsTab && (
            <button
              onClick={() => setActiveTab('tests')}
              className={`px-3 py-1.5 text-sm font-medium rounded-t transition-colors ${
                activeTab === 'tests'
                  ? 'text-primary-600 bg-white dark:bg-gray-900 border-t border-l border-r border-gray-200 dark:border-gray-700 -mb-px'
                  : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200'
              }`}
            >
              Tests ({visibleTestCount})
            </button>
          )}
          <button
            onClick={() => setActiveTab('output')}
            className={`px-3 py-1.5 text-sm font-medium rounded-t transition-colors ${
              activeTab === 'output'
                ? 'text-primary-600 bg-white dark:bg-gray-900 border-t border-l border-r border-gray-200 dark:border-gray-700 -mb-px'
                : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200'
            }`}
          >
            Output
          </button>
        </div>
        {isLoading && (
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 bg-primary-500 rounded-full animate-pulse" />
            <span className="text-xs text-gray-500 dark:text-gray-400">Running...</span>
          </div>
        )}
      </div>

      {activeTab === 'tests' && showTestsTab ? (
        <div className="flex-1 flex flex-col overflow-hidden">
          <textarea
            value={testCaseText}
            onChange={(e) => handleTestCaseTextChange(e.target.value)}
            className="flex-1 p-4 font-mono text-sm bg-gray-900 text-gray-300 resize-none focus:outline-none"
            placeholder="# Test 1: Description&#10;input: your_function(args)&#10;expected: expected_result"
            spellCheck={false}
          />
          {hiddenCount > 0 && (
            <div className="px-4 py-2 bg-gray-800 text-gray-400 text-xs border-t border-gray-700">
              + {hiddenCount} hidden test{hiddenCount > 1 ? 's' : ''} (not editable)
            </div>
          )}
        </div>
      ) : (
        <div className="flex-1 overflow-auto p-4 font-mono text-sm bg-gray-900 rounded-b-lg">
          {output.length === 0 ? (
            <div className="text-gray-500 italic">
              Output will appear here when you run your code...
            </div>
          ) : (
            output.map((line, index) => (
              <div
                key={index}
                className={`py-0.5 ${
                  line.startsWith('[Error]')
                    ? 'text-red-400'
                    : line.includes('PASSED')
                    ? 'text-green-400'
                    : line.includes('FAILED')
                    ? 'text-red-400'
                    : 'text-gray-300'
                }`}
              >
                {line.startsWith('Test') ? (
                  <span>
                    {line.includes('PASSED') ? '✓ ' : line.includes('FAILED') ? '✗ ' : ''}
                    {line}
                  </span>
                ) : (
                  <span>&gt; {line}</span>
                )}
              </div>
            ))
          )}
          <div ref={bottomRef} />
        </div>
      )}
    </div>
  );
}
