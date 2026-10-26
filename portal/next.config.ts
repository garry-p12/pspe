import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* `next dev` blocks cross-origin requests to /_next/hmr by default, and the
   * loopback address counts as cross-origin against `localhost`. The block is
   * silent in the page: HMR never connects, React never hydrates, and the
   * portal renders its server HTML forever -- which is why screenshots of it
   * came out reading "Loading...". Naming the loopback host fixes the dev
   * server; it has no effect on `next build`/`next start`. */
  allowedDevOrigins: ["127.0.0.1"],
};

export default nextConfig;
