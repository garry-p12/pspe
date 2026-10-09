import { NextResponse } from "next/server";

/**
 * A trip through a flood: origin, destination, and the reasons.
 *
 * Proxies the service's `/route`. Slower than the other endpoints and
 * unavoidably so: the domain is derived from the two ends, so terrain has to be
 * fetched for a box that contains both, the flood solved across it, and the
 * road network pulled from OpenStreetMap before anything can be routed. Two
 * minutes is normal for a 20 km trip.
 */
const SERVICE = process.env.FLOOD_SERVICE ?? "http://127.0.0.1:8000";

export const maxDuration = 600;

/** Did WE stop waiting, or did the service really not answer?
 *
 *  Node's fetch gives up on response headers after five minutes
 *  (undici's `headersTimeout`) and `AbortSignal.timeout` fires on our own
 *  deadline. Both arrive here as exceptions, and both used to be reported as
 *  "not reachable" — which is the opposite of true when the service is
 *  mid-solve on the question that was asked. */
function timedOut(e: unknown): boolean {
  const s = `${(e as Error)?.name ?? ""} ${(e as Error)?.message ?? ""} `
    + `${(e as { cause?: { code?: string } })?.cause?.code ?? ""}`;
  return /TimeoutError|HeadersTimeout|BodyTimeout|UND_ERR_(HEADERS|BODY)_TIMEOUT|aborted/i.test(s);
}

export async function POST(req: Request) {
  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "bad_request" }, { status: 400 });
  }
  try {
    const r = await fetch(`${SERVICE}/route`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(580_000),
    });
    if (!r.ok) {
      const t = await r.text();
      return NextResponse.json(
        { error: "routing_error", detail: t.slice(0, 400) },
        { status: 502 },
      );
    }
    return NextResponse.json(await r.json());
  } catch (e) {
    return NextResponse.json(
      timedOut(e)
        ? {
            error: "timed_out",
            detail: "The solve is still running after five minutes, which "
                  + "is as long as this page will wait for it. It is not "
                  + "lost — the service finishes and caches it, so asking "
                  + "again is usually quick.",
            cause: String((e as Error).message).slice(0, 200),
          }
        : {
            error: "service_unavailable",
            detail:
              "The flood service is not reachable. Start it with: " +
              "python3 -m uvicorn service.app:app --port 8000",
            cause: String((e as Error).message).slice(0, 200),
          },
      { status: 503 },
    );
  }
}
