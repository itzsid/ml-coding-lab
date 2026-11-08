import { ReactNode, useState } from 'react';
import { useLocation } from 'react-router-dom';
import Header from './Header';
import Sidebar from './Sidebar';
import Footer from './Footer';

interface LayoutProps {
  children: ReactNode;
}

export default function Layout({ children }: LayoutProps) {
  const [isMobileSidebarOpen, setIsMobileSidebarOpen] = useState(false);
  // Full-height workspaces (problem, scratchpad) get the whole width: no
  // curriculum sidebar and no footer. The sidebar stays reachable via the menu.
  const { pathname } = useLocation();
  const isWorkspace = pathname.startsWith('/problem/') || pathname === '/scratchpad';

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-dark-900 flex flex-col">
      {/* Background atmosphere (dark mode only, hidden via CSS) */}
      <div className="bg-mesh" />
      <div className="bg-noise" />

      <div className="relative z-[1] flex flex-col min-h-screen">
        <Header
          onMenuToggle={() => setIsMobileSidebarOpen(!isMobileSidebarOpen)}
          showMenuOnDesktop={isWorkspace}
        />
        <div className="flex flex-1">
          {/* Desktop sidebar */}
          {!isWorkspace && (
            <div className="hidden lg:block">
              <Sidebar />
            </div>
          )}

          {/* Mobile sidebar overlay */}
          {isMobileSidebarOpen && (
            <div className={`fixed inset-0 z-40 ${isWorkspace ? '' : 'lg:hidden'}`}>
              <div
                className="fixed inset-0 bg-black/50 backdrop-blur-sm"
                onClick={() => setIsMobileSidebarOpen(false)}
              />
              <div className="fixed inset-y-0 left-0 z-50 w-64 bg-white dark:bg-dark-800 shadow-xl overflow-y-auto border-r border-gray-200 dark:border-dark-500">
                <div className="pt-2">
                  <Sidebar onNavigate={() => setIsMobileSidebarOpen(false)} />
                </div>
              </div>
            </div>
          )}

          <main className="flex-1 min-w-0 p-6">
            {children}
          </main>
        </div>
        {!isWorkspace && <Footer />}
      </div>
    </div>
  );
}
