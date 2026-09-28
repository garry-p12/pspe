import type { Metadata } from "next";
import { FloodTwin } from "@/components/FloodTwin";

export const metadata: Metadata = {
  title: "Flood twin — 2022 Northern Rivers | PSPE",
  description:
    "Recorded inundation of the lower Richmond River, our solver scored against the reference at CSI 0.979, and levee allocation where half the candidate sites make flooding worse.",
};

export default function FloodPage() {
  return <FloodTwin />;
}
