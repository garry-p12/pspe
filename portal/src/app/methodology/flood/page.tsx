import type { Metadata } from "next";
import { FloodTwin } from "@/components/FloodTwin";

export const metadata: Metadata = {
  title: "Validation case — 2022 Northern Rivers | PSPE",
  description:
    "The archived Richmond River case, kept as the one place the solver is scored against a reference solution rather than a regulatory map: CSI 0.979, and levee allocation where half the candidate sites make flooding worse. The planner itself ships Onion Creek, Austin.",
};

export default function FloodPage() {
  return <FloodTwin />;
}
