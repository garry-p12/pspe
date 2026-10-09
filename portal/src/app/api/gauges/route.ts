import { NextResponse } from "next/server";

/**
 * Live river gauges over a box.
 *
 * Proxies the service's `/gauges`, which reads USGS instantaneous values on a
 * 15-minute cadence and converts feet and cubic feet per second into metres
 * and cubic metres per second — a portal that puts a gauge reading beside a
 * solved depth has to convert, or it labels two different quantities the same
 * way.
 *
 * United States only, and the response says so rather than coming back empty:
 * the Richmond study has no equivalent open feed.
 */
const SERVICE = process.env.FLOOD_SERVICE ?? "http://127.0.0.1:8000";

export const maxDuration = 120;

/** Did WE stop waiting, or did the service really not answer?
 *
 *  Node's fetch gives up on response headers after five minutes
 *  (undici's `headersTimeout`) and `AbortSignal.timeout` fires on our own
 *  deadline. Both arrive here as exceptions, and both used to be reported as
 *  "the flood service is not reachable" — which is the opposite of true when
 *  the service is mid-solve on the question that was asked. */
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
    const r = await fetch(`${SERVICE}/gauges`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(110_000),
    });
    if (!r.ok) {
      return NextResponse.json(
        { error: "gauge_error", detail: (await r.text()).slice(0, 300) },
        { status: 502 },
      );
    }
    return NextResponse.json(await r.json());
  } catch (e) {
    return NextResponse.json(
      // "Not reachable" was a lie whenever the service was answering and the
      // CLIENT gave up: Node's fetch stops waiting for response headers at
      // five minutes, and a cold solve on a contended machine passes that.
      // The panel then told the reader to check a service that was busy
      // working on their question.
      timedOut(e)
        ? {
            error: "timed_out",
            detail: "The solve is still running after five minutes, which is "
                  + "as long as this page will wait for it. It is not lost — "
                  + "the service finishes and caches it, so asking again is "
                  + "usually quick.",
            cause: String((e as Error).message).slice(0, 200),
          }
        : {
            error: "service_unavailable",
            detail: "The flood service is not reachable.",
            cause: String((e as Error).message).slice(0, 200),
          },
      { status: 503 },
    );
  }
}
