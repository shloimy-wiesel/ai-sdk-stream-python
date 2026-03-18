import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  allowedDevOrigins: ["10.111.117.4"],
  reactCompiler: true,
  // Installed vendor components (ui/, ai-elements/) have minor type issues;
  // we type-check only our app code via tsconfig.check.json
  typescript: {
    ignoreBuildErrors: true,
  },
  // Rewrite /api/* to the Python FastAPI server.
  // Dev:  uvicorn runs on port 8000 alongside Next.js
  // Prod: Vercel serves api/index.py as a Python serverless function
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination:
          process.env.NODE_ENV === "development"
            ? "http://127.0.0.1:8000/api/:path*"
            : "/api/",
      },
    ];
  },
};

export default nextConfig;
