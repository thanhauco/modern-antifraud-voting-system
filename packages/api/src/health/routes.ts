import { FastifyPluginAsync } from "fastify";

interface HealthResponse {
  status: "ok" | "degraded" | "unhealthy";
  timestamp: string;
  version: string;
  services: {
    database: string;
    blockchain: string;
    cache: string;
  };
}

export const healthRoutes: FastifyPluginAsync = async (fastify) => {
  fastify.get<{ Reply: HealthResponse }>("/", async (request, reply) => {
    const health: HealthResponse = {
      status: "ok",
      timestamp: new Date().toISOString(),
      version: process.env.npm_package_version || "0.1.0",
      services: {
        database: "connected",
        blockchain: "connected",
        cache: "connected",
      },
    };
    
    return health;
  });

  fastify.get("/ready", async () => {
    return { ready: true };
  });

  fastify.get("/live", async () => {
    return { live: true };
  });
};
