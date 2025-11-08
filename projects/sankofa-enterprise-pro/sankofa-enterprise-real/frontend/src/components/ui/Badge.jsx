import { forwardRef } from 'react';
import { cn } from '@/lib/utils';

const Badge = forwardRef(({ 
  className, 
  variant = 'default',
  size = 'md',
  ...props 
}, ref) => {
  const baseClasses = 'inline-flex items-center font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2';
  
  const sizeClasses = {
    sm: 'px-2 py-0.5 text-xs rounded-[calc(var(--radius-sm)-2px)]',
    md: 'px-2.5 py-0.5 text-xs rounded-[var(--radius-sm)]',
    lg: 'px-3 py-1 text-sm rounded-[var(--radius-sm)]'
  };
  
  const variantClasses = {
    default: 'bg-[var(--neutral-100)] text-[var(--color-text-primary)] border border-[var(--neutral-200)]',
    success: 'bg-green-100 text-green-800 border border-green-200',
    warning: 'bg-amber-100 text-amber-800 border border-amber-200',
    error: 'bg-red-100 text-red-800 border border-red-200',
    info: 'bg-blue-100 text-blue-800 border border-blue-200',
    brand: 'bg-[var(--brand-100)] text-[var(--brand-800)] border border-[var(--brand-200)]',
    // Status específicos para fraude
    approved: 'bg-green-100 text-green-800 border border-green-200',
    rejected: 'bg-red-100 text-red-800 border border-red-200',
    pending: 'bg-amber-100 text-amber-800 border border-amber-200',
    reviewing: 'bg-blue-100 text-blue-800 border border-blue-200'
  };

  return (
    <div
      className={cn(
        baseClasses,
        sizeClasses[size],
        variantClasses[variant],
        className
      )}
      ref={ref}
      {...props}
    />
  );
});

Badge.displayName = 'Badge';

// Componente específico para status de transações
const TransactionStatusBadge = ({ status, ...props }) => {
  const statusMap = {
    'APROVADA': { variant: 'approved', text: 'Aprovada' },
    'REJEITADA': { variant: 'rejected', text: 'Rejeitada' },
    'PENDENTE': { variant: 'pending', text: 'Pendente' },
    'EM_REVISAO': { variant: 'reviewing', text: 'Em Revisão' },
    'SUSPEITA': { variant: 'warning', text: 'Suspeita' },
    'BLOQUEADA': { variant: 'error', text: 'Bloqueada' }
  };

  const config = statusMap[status] || { variant: 'default', text: status };

  return (
    <Badge variant={config.variant} {...props}>
      {config.text}
    </Badge>
  );
};

// Componente para score de risco
const RiskScoreBadge = ({ score, ...props }) => {
  let variant = 'success';
  let text = 'Baixo Risco';

  if (score >= 0.7) {
    variant = 'error';
    text = 'Alto Risco';
  } else if (score >= 0.4) {
    variant = 'warning';
    text = 'Médio Risco';
  }

  return (
    <Badge variant={variant} {...props}>
      {text} ({(score * 100).toFixed(1)}%)
    </Badge>
  );
};

export { Badge, TransactionStatusBadge, RiskScoreBadge };

