import { NextResponse } from "next/server";

/**
 * How much notice the upstream gauge gives this reach.
 *
 * Proxies the service's `/warning`, which measures the lag between the
 * upstream gauge's peak and the outlet's on real floods. Slow the first time
 * — it reads the annual-peak record and then two days of instantaneous values
 * per event from USGS — and cached by box thereafter.
 */
const SERVICE = process.env.FLOOD_SERVICE ?? "http://127.0.0.1:8000";

export const maxDuration = 300;

/** Did WE stop waiting, or did the service really not answer? */
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
    const r = await fetch(`${SERVICE}/warning`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(280_000),
    });
    if (!r.ok) {
      const t = await r.text();
      return NextResponse.json(
        { error: "warning_error", detail: t.slice(0, 400) },
        { status: 502 },
      );
    }
    return NextResponse.json(await r.json());
  } catch (e) {
    return NextResponse.json(
      timedOut(e)
        ? {
            error: "timed_out",
            detail: "The gauge record is still being read after five minutes, which is "
                  + "as long as this page will wait. It is not lost — the "
                  + "service finishes and caches it, so asking again is "
                  + "usually quick.",
            cause: String((e as Error).message).slice(0, 200),
          }
        : {
            error: "service_unavailable",
            detail:
              "The planning service is not reachable. Start it with: " +
              "python3 -m uvicorn service.app:app --port 8000",
            cause: String((e as Error).message).slice(0, 200),
          },
      { status: 503 },
    );
  }
}
