import { forwardRef } from 'react';
import { cn } from '@/lib/utils';

const Switch = forwardRef(({ 
  className,
  checked = false,
  onCheckedChange,
  disabled = false,
  size = 'md',
  ...props 
}, ref) => {
  const sizeClasses = {
    sm: 'h-4 w-7',
    md: 'h-5 w-9',
    lg: 'h-6 w-11'
  };

  const thumbSizeClasses = {
    sm: 'h-3 w-3',
    md: 'h-4 w-4', 
    lg: 'h-5 w-5'
  };

  const translateClasses = {
    sm: checked ? 'translate-x-3' : 'translate-x-0.5',
    md: checked ? 'translate-x-4' : 'translate-x-0.5',
    lg: checked ? 'translate-x-5' : 'translate-x-0.5'
  };

  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      ref={ref}
      className={cn(
        'peer inline-flex shrink-0 cursor-pointer items-center rounded-full border-2 border-transparent transition-colors',
        'focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-[var(--color-focus)]',
        'disabled:cursor-not-allowed disabled:opacity-50',
        checked 
          ? 'bg-[var(--color-brand)]' 
          : 'bg-[var(--neutral-200)]',
        sizeClasses[size],
        className
      )}
      onClick={() => !disabled && onCheckedChange?.(!checked)}
      disabled={disabled}
      {...props}
    >
      <span
        className={cn(
          'pointer-events-none block rounded-full bg-white shadow-lg ring-0 transition-transform',
          thumbSizeClasses[size],
          translateClasses[size]
        )}
      />
    </button>
  );
});

Switch.displayName = 'Switch';

// Componente de controle com label
const SwitchControl = ({ 
  label, 
  checked, 
  onCheckedChange, 
  description,
  disabled = false,
  size = 'md',
  className,
  ...props 
}) => {
  return (
    <div className={cn('flex items-center justify-between space-x-4', className)} {...props}>
      <div className="flex-1">
        <label className="text-sm font-medium text-[var(--color-text-primary)]">
          {label}
        </label>
        {description && (
          <p className="text-xs text-[var(--color-text-secondary)] mt-1">
            {description}
          </p>
        )}
      </div>
      <Switch
        checked={checked}
        onCheckedChange={onCheckedChange}
        disabled={disabled}
        size={size}
      />
    </div>
  );
};

export { Switch, SwitchControl };

