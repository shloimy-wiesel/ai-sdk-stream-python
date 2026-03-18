import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  allowedDevOrigins: ["10.111.117.4"],
  reactCompiler: true,
  // Installed vendor components (ui/, ai-elements/) have minor type issues;
  // we type-check only our app code via tsconfig.check.json
  typescript: {
    ignoreBuildErrors: true,
  },
};

export default nextConfig;
