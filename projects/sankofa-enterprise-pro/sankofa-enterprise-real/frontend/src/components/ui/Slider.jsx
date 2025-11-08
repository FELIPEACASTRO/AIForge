import { forwardRef } from 'react';
import { cn } from '@/lib/utils';

const Slider = forwardRef(({ 
  className,
  value = [0],
  onValueChange,
  min = 0,
  max = 100,
  step = 1,
  disabled = false,
  ...props 
}, ref) => {
  const handleChange = (e) => {
    const newValue = [parseFloat(e.target.value)];
    onValueChange?.(newValue);
  };

  return (
    <div className={cn('relative flex w-full touch-none select-none items-center', className)} {...props}>
      <input
        ref={ref}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value[0]}
        onChange={handleChange}
        disabled={disabled}
        className={cn(
          'relative h-2 w-full cursor-pointer appearance-none rounded-full bg-[var(--neutral-200)] outline-none',
          'disabled:cursor-not-allowed disabled:opacity-50',
          '[&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:h-5 [&::-webkit-slider-thumb]:w-5',
          '[&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-[var(--color-brand)]',
          '[&::-webkit-slider-thumb]:cursor-pointer [&::-webkit-slider-thumb]:border-2 [&::-webkit-slider-thumb]:border-white',
          '[&::-webkit-slider-thumb]:shadow-md [&::-webkit-slider-thumb]:transition-all',
          '[&::-webkit-slider-thumb]:hover:bg-[var(--brand-600)]',
          '[&::-moz-range-thumb]:appearance-none [&::-moz-range-thumb]:h-5 [&::-moz-range-thumb]:w-5',
          '[&::-moz-range-thumb]:rounded-full [&::-moz-range-thumb]:bg-[var(--color-brand)]',
          '[&::-moz-range-thumb]:cursor-pointer [&::-moz-range-thumb]:border-2 [&::-moz-range-thumb]:border-white',
          '[&::-moz-range-thumb]:shadow-md [&::-moz-range-thumb]:transition-all',
          'focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-[var(--color-focus)]'
        )}
        style={{
          background: `linear-gradient(to right, var(--color-brand) 0%, var(--color-brand) ${((value[0] - min) / (max - min)) * 100}%, var(--neutral-200) ${((value[0] - min) / (max - min)) * 100}%, var(--neutral-200) 100%)`
        }}
      />
    </div>
  );
});

Slider.displayName = 'Slider';

// Componente de controle com label e valor
const SliderControl = ({ 
  label, 
  value, 
  onValueChange, 
  min = 0, 
  max = 1, 
  step = 0.01,
  format = 'decimal',
  description,
  unit = '',
  className,
  ...props 
}) => {
  const formatValue = (val) => {
    switch (format) {
      case 'percentage':
        return `${(val * 100).toFixed(1)}%`;
      case 'decimal':
        return val.toFixed(3);
      case 'integer':
        return Math.round(val).toString();
      case 'currency':
        return new Intl.NumberFormat('pt-BR', {
          style: 'currency',
          currency: 'BRL'
        }).format(val);
      default:
        return val.toString();
    }
  };

  return (
    <div className={cn('space-y-3', className)} {...props}>
      <div className="flex items-center justify-between">
        <div>
          <label className="text-sm font-medium text-[var(--color-text-primary)]">
            {label}
          </label>
          {description && (
            <p className="text-xs text-[var(--color-text-secondary)] mt-1">
              {description}
            </p>
          )}
        </div>
        <div className="text-right">
          <span className="text-sm font-mono font-semibold text-[var(--color-brand)]">
            {formatValue(value[0])}{unit}
          </span>
          <div className="text-xs text-[var(--color-text-secondary)]">
            {min} - {max}
          </div>
        </div>
      </div>
      <Slider
        value={value}
        onValueChange={onValueChange}
        min={min}
        max={max}
        step={step}
      />
    </div>
  );
};

export { Slider, SliderControl };

