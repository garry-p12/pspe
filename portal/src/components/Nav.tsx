"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import clsx from "clsx";
import { useRegion } from "./RegionContext";

/** Operational chrome. The research material lives behind one quiet link. */
export function Nav() {
  const path = usePathname();
  const { region, live } = useRegion();
  const inMethod = path.startsWith("/methodology");
  return (
    <header className="z-30 border-b border-line bg-bg-raised">
      <div className="flex flex-wrap items-center gap-x-5 gap-y-2 px-5 py-2.5">
        <Link href="/" className="flex items-center gap-2.5">
          <span className="flex h-6 w-6 items-center justify-center rounded bg-accent/15 text-[12px] font-bold text-accent">
            FP
          </span>
          <span className="text-[15px] font-semibold tracking-tight text-ink">
            Floodplain Planner
          </span>
        </Link>
        {region && (
          <span className="hidden border-l border-line pl-5 text-[13px] text-ink-mute sm:inline">
            {region}
          </span>
        )}
        <div className="ml-auto flex items-center gap-4">
          {/* What on this page is live, and how often. Only shown when
              something actually is: the Richmond study has no gauge feed and
              a LIVE badge over a replayed archive would be a lie. */}
          {live && (
            <span className="hidden items-center gap-1.5 text-[12px] text-ink-faint md:flex">
              <span className="h-[7px] w-[7px] rounded-full bg-ink" aria-hidden />
              <span className="font-semibold text-ink">LIVE</span>
              <span>· {live}</span>
            </span>
          )}
          <Link
            href={inMethod ? "/" : "/methodology"}
            className={clsx(
              "text-[13px] transition-colors",
              inMethod ? "text-accent" : "text-ink-faint hover:text-ink-mute",
            )}
          >
            {inMethod ? "← Back to planner" : "How this works"}
          </Link>
        </div>
      </div>
    </header>
  );
}
