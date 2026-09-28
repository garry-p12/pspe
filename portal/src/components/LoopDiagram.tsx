import Link from "next/link";
import type { Badge } from "@/lib/types";
import { EvidenceBadge } from "./EvidenceBadge";

const MODULES: {
  key: string;
  name: string;
  line: string;
  badge: Badge;
  evidence: string;
  href?: string;
}[] = [
  {
    key: "perceive",
    name: "Perceive",
    line: "Recover the hazard's state where the sensor could not see.",
    badge: "VALIDATED MODEL",
    evidence: "Scored against held-out satellite observations.",
  },
  {
    key: "simulate",
    name: "Simulate",
    line: "Carry that state forward under the governing physics.",
    badge: "VALIDATED MODEL",
    evidence: "CSI 0.979 against an independent shallow-water reference.",
    href: "/methodology/flood",
  },
  {
    key: "plan",
    name: "Plan",
    line: "Choose interventions under a budget and a hard limit.",
    badge: "PROJECTION",
    evidence: "No record contains the counterfactual. Rests on an action model.",
    href: "/methodology/margin",
  },
  {
    key: "explain",
    name: "Explain",
    line: "State why the plan is what it is, with a checkable certificate.",
    badge: "PROJECTION",
    evidence: "Certificate holds; the faithfulness claim did not survive testing.",
    href: "/methodology/limits",
  },
];

export function LoopDiagram() {
  return (
    <ol className="grid gap-3 lg:grid-cols-4">
      {MODULES.map((m, i) => (
        <li key={m.key} className="relative">
          <Link
            href={m.href ?? "#"}
            className="group flex h-full flex-col rounded-xl border border-line bg-bg-raised p-4 transition-colors hover:border-accent/40"
          >
            <div className="mb-2 flex items-center gap-2">
              <span className="tnum text-[11px] text-ink-faint">
                {String(i + 1).padStart(2, "0")}
              </span>
              <h3 className="text-[15px] font-semibold tracking-tight text-ink group-hover:text-accent">
                {m.name}
              </h3>
            </div>
            <p className="mb-3 text-[13px] leading-relaxed text-ink-mute">
              {m.line}
            </p>
            <div className="mt-auto">
              <EvidenceBadge badge={m.badge} />
              <p className="mt-1.5 text-[12px] leading-relaxed text-ink-faint">
                {m.evidence}
              </p>
            </div>
          </Link>
          {i < MODULES.length - 1 && (
            <span
              aria-hidden
              className="absolute -right-2 top-1/2 hidden h-px w-4 bg-line lg:block"
            />
          )}
        </li>
      ))}
    </ol>
  );
}
