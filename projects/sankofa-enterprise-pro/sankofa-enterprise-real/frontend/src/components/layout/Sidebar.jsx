import { useState } from 'react';
import { 
  LayoutDashboard, 
  CreditCard, 
  Shield, 
  BarChart3, 
  Settings, 
  FileText,
  ChevronLeft,
  ChevronRight,
  Activity,
  AlertTriangle,
  Sliders,
  Database,
  UserCheck,
  UserX,
  Eye
} from 'lucide-react';
import { Button } from '@/components/ui/Button.jsx';
import { cn } from '@/lib/utils';

const navigationItems = [
  {
    title: 'Dashboard',
    icon: LayoutDashboard,
    href: '/',
    description: 'Visão geral e KPIs'
  },
  {
    title: 'Transações',
    icon: CreditCard,
    href: '/transactions',
    description: 'Lista e busca de transações'
  },
  {
    title: 'Calibragem',
    icon: Sliders,
    href: '/calibration',
    description: 'Ajuste manual dos algoritmos'
  },
  {
    title: 'Investigação',
    icon: Shield,
    href: '/investigation',
    description: 'Análise detalhada de fraudes'
  },
  {
    title: 'Revisão Manual',
    icon: Eye,
    href: '/manual-review',
    description: 'Human-in-the-Loop review',
    badge: 'NEW'
  },
  {
    title: 'Monitoramento',
    icon: Activity,
    href: '/monitoring',
    description: 'Saúde dos modelos de IA'
  },
  {
    title: 'Relatórios',
    icon: BarChart3,
    href: '/reports',
    description: 'Análises e métricas'
  },
  {
    title: 'Métricas',
    icon: Activity,
    href: '/metrics',
    description: 'Contadores e métricas em tempo real',
    badge: 'LIVE'
  },
  {
    title: 'Alertas',
    icon: AlertTriangle,
    href: '/alerts',
    description: 'Alertas e notificações'
  },
  {
    title: 'Datasets',
    icon: Database,
    href: '/datasets',
    description: 'Catálogo de datasets',
    badge: '200+'
  },
  {
    title: 'Regras Duras',
    icon: Shield,
    href: '/hard-rules',
    description: 'Regras de bloqueio imediato',
    badge: '12'
  },
  {
    title: 'Lista VIP',
    icon: UserCheck,
    href: '/vip-list',
    description: 'Lista branca - aprovação direta'
  },
  {
    title: 'Lista HOT',
    icon: UserX,
    href: '/hot-list',
    description: 'Lista negra - bloqueio direto'
  },
  {
    title: 'Auditoria',
    icon: FileText,
    href: '/audit',
    description: 'Trilhas de auditoria'
  },
  {
    title: 'Configurações',
    icon: Settings,
    href: '/settings',
    description: 'Configurações do sistema'
  }
];

export function Sidebar({ isOpen, onToggle, currentPath = '/', className }) {
  const [collapsed, setCollapsed] = useState(false);

  const handleToggleCollapse = () => {
    setCollapsed(!collapsed);
  };

  return (
    <>
      {/* Mobile Overlay */}
      {isOpen && (
        <div 
          className="fixed inset-0 z-40 bg-black/50 md:hidden"
          onClick={onToggle}
          aria-hidden="true"
        />
      )}

      {/* Sidebar */}
      <aside className={cn(
        'fixed left-0 top-16 z-50 h-[calc(100vh-4rem)] w-64 transform border-r border-[var(--color-border)] bg-[var(--color-surface)] transition-transform duration-200 ease-in-out',
        'md:relative md:top-0 md:h-[calc(100vh-4rem)] md:translate-x-0',
        isOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0',
        collapsed && 'md:w-16',
        className
      )}>
        <div className="flex h-full flex-col">
          {/* Collapse Toggle (Desktop) */}
          <div className="hidden md:flex justify-end p-2 border-b border-[var(--color-border)]">
            <Button
              variant="ghost"
              size="sm"
              onClick={handleToggleCollapse}
              aria-label={collapsed ? 'Expandir sidebar' : 'Recolher sidebar'}
            >
              {collapsed ? (
                <ChevronRight className="h-4 w-4" />
              ) : (
                <ChevronLeft className="h-4 w-4" />
              )}
            </Button>
          </div>

          {/* Navigation */}
          <nav className="flex-1 overflow-y-auto p-4" aria-label="Navegação principal">
            <ul className="space-y-2">
              {navigationItems.map((item) => {
                const Icon = item.icon;
                const isActive = currentPath === item.href;

                return (
                  <li key={item.href}>
                    <a
                      href={item.href}
                      className={cn(
                        'flex items-center space-x-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors',
                        'hover:bg-[var(--neutral-100)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--color-focus)]',
                        isActive 
                          ? 'bg-[var(--brand-100)] text-[var(--brand-700)]' 
                          : 'text-[var(--color-text-primary)]',
                        collapsed && 'justify-center'
                      )}
                      aria-current={isActive ? 'page' : undefined}
                      title={collapsed ? item.title : undefined}
                    >
                      <Icon className={cn(
                        'h-5 w-5 flex-shrink-0',
                        isActive ? 'text-[var(--brand-600)]' : 'text-[var(--color-text-secondary)]'
                      )} />
                      {!collapsed && (
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center justify-between">
                            <span className="truncate">{item.title}</span>
                            {item.badge && (
                              <span className="ml-2 px-2 py-0.5 text-xs bg-red-500 text-white rounded-full">
                                {item.badge}
                              </span>
                            )}
                          </div>
                          <div className="text-xs text-[var(--color-text-secondary)] truncate">
                            {item.description}
                          </div>
                        </div>
                      )}
                    </a>
                  </li>
                );
              })}
            </ul>
          </nav>

          {/* Footer */}
          {!collapsed && (
            <div className="border-t border-[var(--color-border)] p-4">
              <div className="text-xs text-[var(--color-text-secondary)]">
                <div className="font-medium">Sankofa v11.0</div>
                <div>Sistema de Análise de Risco</div>
              </div>
            </div>
          )}
        </div>
      </aside>
    </>
  );
}

