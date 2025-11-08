import { useState } from 'react';
import { AppBar } from './AppBar';
import { Sidebar } from './Sidebar';
import { cn } from '@/lib/utils';

export function Layout({ children, currentPath }) {
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const handleMenuToggle = () => {
    setSidebarOpen(!sidebarOpen);
  };

  return (
    <div className="min-h-screen bg-[var(--color-bg)]">
      {/* Skip to content link for accessibility */}
      <a 
        href="#main-content" 
        className="skip-link"
      >
        Pular para o conteúdo principal
      </a>

      {/* App Bar */}
      <AppBar onMenuToggle={handleMenuToggle} />

      <div className="flex">
        {/* Sidebar */}
        <Sidebar 
          isOpen={sidebarOpen}
          onToggle={handleMenuToggle}
          currentPath={currentPath}
        />

        {/* Main Content */}
        <main 
          id="main-content"
          className={cn(
            'flex-1 min-h-[calc(100vh-4rem)] p-6',
            'focus:outline-none'
          )}
          tabIndex={-1}
        >
          {children}
        </main>
      </div>
    </div>
  );
}

