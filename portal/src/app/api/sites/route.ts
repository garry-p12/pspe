import { NextResponse } from "next/server";

/**
 * Candidate levee alignments for a box.
 *
 * Proxies the service's `/sites`, which solves the baseline flood and returns
 * the runs of ground that touch it and sit within a couple of metres of being
 * overtopped — the band where a wall of a metre or two changes the answer.
 * Raising ground four metres under water buys nothing for the money, and
 * raising ground already above the flood buys nothing at all.
 *
 * Slow, because it is a solve.
 */
const SERVICE = process.env.FLOOD_SERVICE ?? "http://127.0.0.1:8000";

export const maxDuration = 600;

export async function POST(req: Request) {
  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "bad_request" }, { status: 400 });
  }
  try {
    const r = await fetch(`${SERVICE}/sites`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(580_000),
    });
    if (!r.ok) {
      return NextResponse.json(
        { error: "sites_error", detail: (await r.text()).slice(0, 400) },
        { status: 502 },
      );
    }
    return NextResponse.json(await r.json());
  } catch (e) {
    return NextResponse.json(
      {
        error: "service_unavailable",
        detail: "The flood service is not reachable.",
        cause: String((e as Error).message).slice(0, 200),
      },
      { status: 503 },
    );
  }
}
