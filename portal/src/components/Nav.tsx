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
  const inFire = path.startsWith("/wildfire");
  const inGym = path.startsWith("/gym");
  /* The chrome belongs to the planner. `region` and `live` are written once,
     by the planner's gauge effect, and the provider sits above the router --
     so a click through to the Richmond archive carried "Austin, Texas" and a
     LIVE badge onto a page about a 2022 flood in New South Wales, with no
     feed behind either. The research pages name their own subject in their
     own h1; the header says nothing on their behalf. */
  const subPage = inMethod && path !== "/methodology";
  return (
    <header className="z-30 border-b border-line bg-bg-raised">
      <div className="flex flex-wrap items-center gap-x-5 gap-y-2 px-5 py-2.5">
        <Link href="/" className="flex items-center gap-2.5">
          <span className="flex h-6 w-6 items-center justify-center rounded bg-accent/15 text-[12px] font-bold text-accent">
            FP
          </span>
          <span className="text-[15px] font-semibold tracking-tight text-ink">
            {inGym ? "Simulation Gym" : inFire ? "Wildfire Planner" : "Floodplain Planner"}
          </span>
        </Link>
        {region && !subPage && !inFire && !inGym && (
          <span className="hidden border-l border-line pl-5 text-[13px] text-ink-mute sm:inline">
            {region}
          </span>
        )}
        <div className="ml-auto flex items-center gap-4">
          {/* What on this page is live, and how often. Only shown when
              something actually is: the Richmond study has no gauge feed and
              a LIVE badge over a replayed archive would be a lie. */}
          {live && !inMethod && !inFire && !inGym && (
            <span className="hidden items-center gap-1.5 text-[12px] text-ink-faint md:flex">
              <span className="h-[7px] w-[7px] rounded-full bg-ink" aria-hidden />
              <span className="font-semibold text-ink">LIVE</span>
              <span>· {live}</span>
            </span>
          )}
          {/* Once inside the research pages the only link read "Back to
              planner", so there was no way back to the overview that sent
              you here except the browser's own button. */}
          {subPage && (
            <Link href="/methodology"
                  className="text-[13px] text-ink-faint transition-colors hover:text-ink-mute">
              ← How this works
            </Link>
          )}
          {/* The two hazards are peers, so the link goes both ways rather
              than treating flood as the product and fire as an annexe. */}
          <Link
            href={inFire ? "/" : "/wildfire"}
            className={clsx(
              "text-[13px] transition-colors",
              inFire ? "text-accent" : "text-ink-faint hover:text-ink-mute",
            )}
          >
            {inFire ? "← Flood" : "Wildfire"}
          </Link>
          {/* The gym is the research surface for both hazards, so it sits
              beside "How this works" rather than inside either planner. */}
          <Link
            href={inGym ? "/" : "/gym"}
            className={clsx(
              "text-[13px] transition-colors",
              inGym ? "text-accent" : "text-ink-faint hover:text-ink-mute",
            )}
          >
            {inGym ? "← Planner" : "Environments"}
          </Link>
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
