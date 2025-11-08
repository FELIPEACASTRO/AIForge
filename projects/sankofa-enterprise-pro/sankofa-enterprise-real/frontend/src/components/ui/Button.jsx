import { forwardRef } from 'react';
import { cn } from '@/lib/utils';
import { Loader2 } from 'lucide-react';

const Button = forwardRef(({ 
  className, 
  variant = 'primary', 
  size = 'md', 
  loading = false,
  children, 
  disabled,
  ...props 
}, ref) => {
  const baseClasses = 'inline-flex items-center justify-center font-medium transition-colors focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-[var(--color-focus)] disabled:pointer-events-none disabled:opacity-50';
  
  const sizeClasses = {
    sm: 'h-8 px-3 text-sm rounded-[var(--radius-sm)]',
    md: 'h-10 px-4 text-base rounded-[var(--radius-sm)]',
    lg: 'h-12 px-5 text-lg rounded-[var(--radius-md)]'
  };
  
  const variantClasses = {
    primary: 'bg-[var(--color-cta)] text-white hover:bg-[var(--brand-700)] active:bg-[var(--brand-800)]',
    secondary: 'bg-[var(--neutral-100)] text-[var(--color-text-primary)] hover:bg-[var(--neutral-200)] active:bg-[var(--neutral-300)] border border-[var(--neutral-200)]',
    tertiary: 'text-[var(--color-brand)] hover:bg-[var(--brand-50)] active:bg-[var(--brand-100)]',
    ghost: 'text-[var(--color-text-primary)] hover:bg-[var(--neutral-100)] active:bg-[var(--neutral-200)]',
    danger: 'bg-[var(--error-500)] text-white hover:bg-red-600 active:bg-red-700'
  };

  return (
    <button
      className={cn(
        baseClasses,
        sizeClasses[size],
        variantClasses[variant],
        className
      )}
      ref={ref}
      disabled={disabled || loading}
      aria-busy={loading}
      {...props}
    >
      {loading && (
        <Loader2 className="mr-2 h-4 w-4 animate-spin" aria-hidden="true" />
      )}
      {children}
    </button>
  );
});

Button.displayName = 'Button';

export { Button };

