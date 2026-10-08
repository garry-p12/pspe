"use client";

import { createContext, useContext, useMemo, useState } from "react";

/** The place the header names.
 *
 * The header used to be a constant in the layout, which was wrong the moment
 * the planner analysed somewhere else: the chrome said Richmond Valley while
 * the map showed Iowa. The planner owns which place is on screen, so it sets
 * this, and the header reads it.
 */
const Ctx = createContext<{
  region: string | null;
  setRegion: (r: string | null) => void;
  /** What on this page updates by itself, and how often. Null when nothing
   *  does, which is the honest state for a replayed archive. */
  live: string | null;
  setLive: (s: string | null) => void;
}>({ region: null, setRegion: () => {}, live: null, setLive: () => {} });

export function useRegion() {
  return useContext(Ctx);
}

export function RegionProvider({
  initial,
  children,
}: {
  initial: string;
  children: React.ReactNode;
}) {
  const [region, setRegion] = useState<string | null>(null);
  const [live, setLive] = useState<string | null>(null);
  const value = useMemo(
    () => ({ region: region ?? initial, setRegion, live, setLive }),
    [region, initial, live],
  );
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}
