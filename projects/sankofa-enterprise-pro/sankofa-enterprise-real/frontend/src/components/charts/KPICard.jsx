import { TrendingUp, TrendingDown, Minus } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card.jsx';
import { cn } from '@/lib/utils';

export function KPICard({ 
  title, 
  value, 
  previousValue,
  format = 'number',
  icon: Icon,
  trend,
  className,
  ...props 
}) {
  // Calculate trend if not provided
  const calculatedTrend = trend || (previousValue ? 
    (value > previousValue ? 'up' : value < previousValue ? 'down' : 'neutral') 
    : 'neutral'
  );

  // Format value based on type
  const formatValue = (val) => {
    if (typeof val !== 'number') return val;
    
    switch (format) {
      case 'currency':
        return new Intl.NumberFormat('pt-BR', {
          style: 'currency',
          currency: 'BRL'
        }).format(val);
      case 'percentage':
        return `${(val * 100).toFixed(1)}%`;
      case 'number':
        return new Intl.NumberFormat('pt-BR').format(val);
      case 'decimal':
        return val.toFixed(2);
      default:
        return val;
    }
  };

  // Calculate percentage change
  const percentageChange = previousValue && previousValue !== 0 
    ? ((value - previousValue) / previousValue) * 100 
    : 0;

  const trendConfig = {
    up: {
      icon: TrendingUp,
      color: 'text-green-600',
      bgColor: 'bg-green-50',
      label: 'Aumento'
    },
    down: {
      icon: TrendingDown,
      color: 'text-red-600',
      bgColor: 'bg-red-50',
      label: 'Diminuição'
    },
    neutral: {
      icon: Minus,
      color: 'text-[var(--color-text-secondary)]',
      bgColor: 'bg-[var(--neutral-100)]',
      label: 'Estável'
    }
  };

  const trendInfo = trendConfig[calculatedTrend];
  const TrendIcon = trendInfo.icon;

  return (
    <Card className={cn('relative overflow-hidden', className)} {...props}>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-small font-medium text-[var(--color-text-secondary)]">
          {title}
        </CardTitle>
        {Icon && (
          <div className="h-8 w-8 rounded-lg bg-[var(--brand-100)] flex items-center justify-center">
            <Icon className="h-4 w-4 text-[var(--brand-600)]" />
          </div>
        )}
      </CardHeader>
      
      <CardContent>
        <div className="flex items-baseline justify-between">
          <div className="text-h1 font-bold text-[var(--color-text-primary)]">
            {formatValue(value)}
          </div>
          
          {previousValue !== undefined && (
            <div className={cn(
              'flex items-center space-x-1 rounded-full px-2 py-1 text-xs font-medium',
              trendInfo.bgColor
            )}>
              <TrendIcon className={cn('h-3 w-3', trendInfo.color)} />
              <span className={trendInfo.color}>
                {Math.abs(percentageChange).toFixed(1)}%
              </span>
            </div>
          )}
        </div>
        
        {previousValue !== undefined && (
          <p className="text-xs text-[var(--color-text-secondary)] mt-1">
            {trendInfo.label} em relação ao período anterior
          </p>
        )}
      </CardContent>
    </Card>
  );
}

