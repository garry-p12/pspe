"use client";

import { useEffect, useId, useRef, useState } from "react";

/**
 * The one place an explanation is allowed to live.
 *
 * This portal had grown three habits at once: `<details>` folds, HTML
 * `title` tooltips, and long muted paragraphs sitting between a control and
 * the button the reader came to press. Measured across the five views that
 * was 8,251 characters of panel, 4,315 of them explanatory prose -- 74% of
 * the street panel, where a 464-character note about a calibration
 * multiplier stood between the input and "Open the street view".
 *
 * The rule this component enforces: a panel carries the verdict, the
 * numbers, and anything that changes what the reader would DO. Everything
 * else -- how the model works, where the data came from, why a decision was
 * made -- goes behind one small marked control, in the same place every
 * time, so it is findable without being in the way.
 *
 * It is a button and a panel, not a tooltip: tooltips cannot be opened from
 * a keyboard, cannot be read by touch, and cannot hold a paragraph.
 */
export function Explain({
  title,
  children,
  label,
  align = "right",
}: {
  /** The heading inside the popover, and the thing the button is "about". */
  title: string;
  children: React.ReactNode;
  /** Overrides the accessible name. Defaults to `About {title}`. */
  label?: string;
  /** Which edge the panel hangs from; `left` for triggers near the right. */
  align?: "left" | "right";
}) {
  const [open, setOpen] = useState(false);
  const host = useRef<HTMLSpanElement>(null);
  const id = useId();

  useEffect(() => {
    if (!open) return;
    const away = (e: MouseEvent) => {
      if (!host.current?.contains(e.target as Node)) setOpen(false);
    };
    const key = (e: KeyboardEvent) => { if (e.key === "Escape") setOpen(false); };
    // `capture` so a click that also lands on a map layer still closes this.
    document.addEventListener("mousedown", away, true);
    document.addEventListener("keydown", key);
    return () => {
      document.removeEventListener("mousedown", away, true);
      document.removeEventListener("keydown", key);
    };
  }, [open]);

  return (
    <span ref={host} className="relative inline-flex align-middle">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        aria-controls={open ? id : undefined}
        aria-label={label ?? `About ${title}`}
        className="grid h-[15px] w-[15px] place-items-center rounded-full border
                   border-line text-[10px] font-semibold leading-none
                   text-ink-faint transition-colors hover:border-ink-faint
                   hover:text-ink focus:outline-none focus-visible:ring-2
                   focus-visible:ring-accent/50"
      >
        i
      </button>
      {open && (
        <div
          id={id}
          role="dialog"
          aria-label={title}
          className={`absolute top-[20px] z-40 w-[290px] rounded-[10px] border
                      border-line bg-bg-raised p-3 text-left shadow-[0_8px_28px_rgba(16,20,28,0.16)]
                      ${align === "right" ? "left-0" : "right-0"}`}
        >
          <div className="flex items-baseline justify-between gap-2">
            <p className="eyebrow">{title}</p>
            <button
              type="button"
              onClick={() => setOpen(false)}
              className="text-[11px] text-ink-faint hover:text-ink"
            >
              close
            </button>
          </div>
          <div className="mt-1.5 flex flex-col gap-2 text-[12px] leading-relaxed text-ink-mute">
            {children}
          </div>
        </div>
      )}
    </span>
  );
}
