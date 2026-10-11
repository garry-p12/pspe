import { NextResponse } from "next/server";

/**
 * Run the structures a plan placed, and compare the ways of running them.
 *
 * Proxies the service's `/operate`. Where `/plan` answers what to build, this
 * answers how to run it: a schedule per controller, scored over the same
 * protected ground `/plan` scores a design over, through
 * `pspe.envs.flood_env.FloodOperateEnv` — the environment a policy is trained
 * in rather than a second implementation of it.
 *
 * Slow, because it is several solves: one event per controller compared.
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
    const r = await fetch(`${SERVICE}/operate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(280_000),
    });
    if (!r.ok) {
      const t = await r.text();
      return NextResponse.json(
        { error: "operate_error", detail: t.slice(0, 400) },
        { status: 502 },
      );
    }
    return NextResponse.json(await r.json());
  } catch (e) {
    return NextResponse.json(
      timedOut(e)
        ? {
            error: "timed_out",
            detail: "The event is still running after five minutes, which is "
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
