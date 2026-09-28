import clsx from "clsx";
import type { ReactNode } from "react";
import type { Badge } from "@/lib/types";
import { EvidenceBadge } from "./EvidenceBadge";

/** A titled surface. Every panel that carries numbers must declare a badge. */
export function Panel({
  title,
  eyebrow,
  badge,
  badgeDetail,
  aside,
  children,
  className,
  bodyClassName,
}: {
  title?: string;
  eyebrow?: string;
  badge?: Badge;
  badgeDetail?: string;
  aside?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClassName?: string;
}) {
  return (
    <section
      className={clsx(
        "rounded-xl border border-line bg-bg-raised overflow-hidden",
        className,
      )}
    >
      {(title || badge || aside) && (
        <header className="flex flex-wrap items-center gap-x-3 gap-y-2 border-b border-line-soft px-4 py-3">
          <div className="min-w-0 flex-1">
            {eyebrow && <p className="eyebrow mb-0.5">{eyebrow}</p>}
            {title && (
              <h2 className="truncate text-[13.5px] font-semibold tracking-tight text-ink">
                {title}
              </h2>
            )}
          </div>
          {badge && <EvidenceBadge badge={badge} detail={badgeDetail} />}
          {aside}
        </header>
      )}
      <div className={clsx("p-4", bodyClassName)}>{children}</div>
    </section>
  );
}

/** A labelled figure, for the readout strips. */
export function Stat({
  label,
  value,
  unit,
  tone = "default",
  hint,
}: {
  label: string;
  value: string | number;
  unit?: string;
  tone?: "default" | "good" | "warn" | "bad";
  hint?: string;
}) {
  const toneClass = {
    default: "text-ink",
    good: "text-ok",
    warn: "text-warn",
    bad: "text-bad",
  }[tone];
  return (
    <div title={hint}>
      <p className="eyebrow mb-1">{label}</p>
      <p className={clsx("tnum text-[19px] font-semibold leading-none", toneClass)}>
        {value}
        {unit && (
          <span className="ml-1 text-[11px] font-normal text-ink-faint">{unit}</span>
        )}
      </p>
    </div>
  );
}
