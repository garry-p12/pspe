import clsx from "clsx";
import type { Badge } from "@/lib/types";

/**
 * The portal's central honesty device.
 *
 * Every panel that shows a number states where that number's authority comes
 * from. A demo is exactly where a carefully-scoped result turns into an
 * overclaim, so the distinction between "a satellite recorded this", "our model
 * was scored against an independent reference" and "this rests on an action
 * model nobody has validated" is made structural rather than left to prose.
 */
const STYLE: Record<Badge, { dot: string; text: string; ring: string }> = {
  OBSERVED: {
    dot: "bg-[var(--badge-observed)]",
    text: "text-[var(--badge-observed)]",
    ring: "ring-[var(--badge-observed)]/25",
  },
  "VALIDATED MODEL": {
    dot: "bg-[var(--badge-validated)]",
    text: "text-[var(--badge-validated)]",
    ring: "ring-[var(--badge-validated)]/25",
  },
  PROJECTION: {
    dot: "bg-[var(--badge-projection)]",
    text: "text-[var(--badge-projection)]",
    ring: "ring-[var(--badge-projection)]/25",
  },
};

export const BADGE_MEANING: Record<Badge, string> = {
  OBSERVED: "A satellite or survey recorded this. No model stands between the reading and the display.",
  "VALIDATED MODEL":
    "Our model, scored against an independent reference and reported with that score.",
  PROJECTION:
    "Rests on a stated action model. No observational record contains the counterfactual, so this is not validated.",
};

export function EvidenceBadge({
  badge,
  detail,
  className,
}: {
  badge: Badge;
  detail?: string;
  className?: string;
}) {
  const s = STYLE[badge];
  return (
    <span
      className={clsx(
        "inline-flex items-center gap-1.5 rounded-full px-2 py-[3px]",
        "bg-[var(--bg-inset)] ring-1", s.ring, className,
      )}
      title={detail ? `${BADGE_MEANING[badge]} — ${detail}` : BADGE_MEANING[badge]}
    >
      <span className={clsx("h-1.5 w-1.5 rounded-full", s.dot)} aria-hidden />
      <span
        className={clsx("text-[9.5px] font-semibold tracking-[0.12em]", s.text)}
      >
        {badge}
      </span>
    </span>
  );
}

/** Legend for the three states, used on the overview and limits pages. */
export function BadgeLegend() {
  return (
    <dl className="grid gap-3 sm:grid-cols-3">
      {(Object.keys(BADGE_MEANING) as Badge[]).map((b) => (
        <div
          key={b}
          className="rounded-lg border border-line bg-bg-raised p-3"
        >
          <dt className="mb-1.5">
            <EvidenceBadge badge={b} />
          </dt>
          <dd className="text-[12px] leading-relaxed text-ink-mute">
            {BADGE_MEANING[b]}
          </dd>
        </div>
      ))}
    </dl>
  );
}
