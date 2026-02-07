import { useEffect } from 'react';

interface SuccessBannerProps {
  show: boolean;
  onDismiss: () => void;
}

export default function SuccessBanner({ show, onDismiss }: SuccessBannerProps) {
  useEffect(() => {
    if (show) {
      const timer = setTimeout(onDismiss, 3000);
      return () => clearTimeout(timer);
    }
  }, [show, onDismiss]);

  if (!show) return null;

  return (
    <div className="fixed top-16 left-0 right-0 z-40 animate-slide-down">
      <div className="bg-green-500 text-white px-4 py-3 shadow-lg">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-lg">&#x2705;</span>
            <span className="font-medium">All tests passed! Great work!</span>
          </div>
          <button
            onClick={onDismiss}
            className="text-white/80 hover:text-white transition-colors text-sm"
          >
            Dismiss
          </button>
        </div>
      </div>
    </div>
  );
}
