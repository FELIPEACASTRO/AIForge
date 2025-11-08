import { forwardRef } from 'react';
import { cn } from '@/lib/utils';

const Input = forwardRef(({ 
  className, 
  type = 'text',
  error,
  ...props 
}, ref) => {
  return (
    <input
      type={type}
      className={cn(
        'flex h-10 w-full rounded-[var(--radius-sm)] border border-[var(--color-border)] bg-[var(--color-surface)] px-3 py-2 text-sm',
        'placeholder:text-[var(--color-text-secondary)]',
        'focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-[var(--color-focus)]',
        'disabled:cursor-not-allowed disabled:opacity-50',
        error && 'border-[var(--error-500)] focus-visible:ring-red-500/20',
        className
      )}
      ref={ref}
      aria-invalid={error ? 'true' : 'false'}
      {...props}
    />
  );
});
Input.displayName = 'Input';

const Label = forwardRef(({ className, ...props }, ref) => (
  <label
    ref={ref}
    className={cn(
      'text-small font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70',
      className
    )}
    {...props}
  />
));
Label.displayName = 'Label';

const FormField = ({ label, error, children, required, ...props }) => {
  return (
    <div className="space-y-2" {...props}>
      {label && (
        <Label>
          {label}
          {required && <span className="text-[var(--error-500)] ml-1" aria-label="obrigatório">*</span>}
        </Label>
      )}
      {children}
      {error && (
        <p className="text-small text-[var(--error-500)]" role="alert">
          {error}
        </p>
      )}
    </div>
  );
};

export { Input, Label, FormField };

