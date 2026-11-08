import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { Progress, ProblemStatus, Language } from '../types';

const STORAGE_KEY = 'ml-interview-progress';

interface ProgressContextType {
  progress: Progress;
  updateProblemStatus: (sectionId: string, problemId: string, status: ProblemStatus, language?: Language) => void;
  saveProblemCode: (sectionId: string, problemId: string, code: string, language?: Language) => void;
  markSolutionViewed: (sectionId: string, problemId: string) => void;
  getProblemProgress: (sectionId: string, problemId: string) => {
    status: ProblemStatus;
    code?: string;
    codes: Partial<Record<Language, string>>;
    solvedIn: Language[];
    solutionViewed: boolean;
  };
  getSectionProgress: (sectionId: string, totalProblems: number) => number;
  getOverallProgress: (sections: { id: string; problemCount: number }[]) => number;
  resetProgress: () => void;
}

const ProgressContext = createContext<ProgressContextType | null>(null);

export function ProgressProvider({ children }: { children: React.ReactNode }) {
  const [progress, setProgress] = useState<Progress>(() => {
    const saved = localStorage.getItem(STORAGE_KEY);
    return saved ? JSON.parse(saved) : {};
  });

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(progress));
  }, [progress]);

  const updateProblemStatus = useCallback((
    sectionId: string,
    problemId: string,
    status: ProblemStatus,
    language?: Language
  ) => {
    setProgress(prev => {
      const current = prev[sectionId]?.[problemId];
      const solvedIn = current?.solvedIn ?? [];
      return {
        ...prev,
        [sectionId]: {
          ...prev[sectionId],
          [problemId]: {
            ...current,
            status,
            ...(status === 'completed' && language && !solvedIn.includes(language)
              ? { solvedIn: [...solvedIn, language] }
              : {}),
            lastAttempt: new Date().toISOString(),
          },
        },
      };
    });
  }, []);

  const saveProblemCode = useCallback((
    sectionId: string,
    problemId: string,
    code: string,
    language: Language = 'numpy'
  ) => {
    setProgress(prev => {
      const current = prev[sectionId]?.[problemId];
      return {
        ...prev,
        [sectionId]: {
          ...prev[sectionId],
          [problemId]: {
            ...current,
            status: current?.status || 'in_progress',
            ...(language === 'numpy'
              ? { code }
              : { codes: { ...current?.codes, [language]: code } }),
            lastAttempt: new Date().toISOString(),
          },
        },
      };
    });
  }, []);

  const markSolutionViewed = useCallback((sectionId: string, problemId: string) => {
    setProgress(prev => {
      const current = prev[sectionId]?.[problemId];
      if (current?.solutionViewed) return prev;
      return {
        ...prev,
        [sectionId]: {
          ...prev[sectionId],
          [problemId]: {
            ...current,
            status: current?.status || 'in_progress',
            solutionViewed: true,
          },
        },
      };
    });
  }, []);

  const getProblemProgress = useCallback((sectionId: string, problemId: string) => {
    const problemProgress = progress[sectionId]?.[problemId];
    return {
      status: problemProgress?.status || 'not_started',
      code: problemProgress?.code,
      codes: problemProgress?.codes ?? {},
      solvedIn: problemProgress?.solvedIn ?? [],
      solutionViewed: problemProgress?.solutionViewed ?? false,
    };
  }, [progress]);

  const getSectionProgress = useCallback((sectionId: string, totalProblems: number) => {
    const sectionProgress = progress[sectionId];
    if (!sectionProgress || totalProblems === 0) return 0;

    const completedCount = Object.values(sectionProgress).filter(
      p => p.status === 'completed'
    ).length;

    return Math.round((completedCount / totalProblems) * 100);
  }, [progress]);

  const getOverallProgress = useCallback((sections: { id: string; problemCount: number }[]) => {
    const totalProblems = sections.reduce((sum, s) => sum + s.problemCount, 0);
    if (totalProblems === 0) return 0;

    let completedCount = 0;
    for (const section of sections) {
      const sectionProgress = progress[section.id];
      if (sectionProgress) {
        completedCount += Object.values(sectionProgress).filter(
          p => p.status === 'completed'
        ).length;
      }
    }

    return Math.round((completedCount / totalProblems) * 100);
  }, [progress]);

  const resetProgress = useCallback(() => {
    setProgress({});
    localStorage.removeItem(STORAGE_KEY);
  }, []);

  return (
    <ProgressContext.Provider
      value={{
        progress,
        updateProblemStatus,
        saveProblemCode,
        markSolutionViewed,
        getProblemProgress,
        getSectionProgress,
        getOverallProgress,
        resetProgress,
      }}
    >
      {children}
    </ProgressContext.Provider>
  );
}

export function useProgress() {
  const context = useContext(ProgressContext);
  if (!context) {
    throw new Error('useProgress must be used within a ProgressProvider');
  }
  return context;
}
