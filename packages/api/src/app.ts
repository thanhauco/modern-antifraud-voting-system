import Fastify, { FastifyInstance } from "fastify";
import cors from "@fastify/cors";
import helmet from "@fastify/helmet";
import rateLimit from "@fastify/rate-limit";
import { authRoutes } from "./auth/routes";
import { votingRoutes } from "./voting/routes";
import { antifraudRoutes } from "./antifraud/routes";
import { auditRoutes } from "./audit/routes";
import { healthRoutes } from "./health/routes";

export interface AppConfig {
  port: number;
  host: string;
  logLevel: string;
  trustProxy: boolean;
}

const defaultConfig: AppConfig = {
  port: parseInt(process.env.PORT || "3000"),
  host: process.env.HOST || "0.0.0.0",
  logLevel: process.env.LOG_LEVEL || "info",
  trustProxy: process.env.TRUST_PROXY === "true",
};

export async function buildApp(config: AppConfig = defaultConfig): Promise<FastifyInstance> {
  const app = Fastify({
    logger: {
      level: config.logLevel,
      transport: {
        target: "pino-pretty",
        options: { colorize: true },
      },
    },
    trustProxy: config.trustProxy,
  });

  // Security middleware
  await app.register(helmet, {
    contentSecurityPolicy: {
      directives: {
        defaultSrc: ["'self'"],
        scriptSrc: ["'self'"],
        styleSrc: ["'self'", "'unsafe-inline'"],
      },
    },
  });

  await app.register(cors, {
    origin: process.env.CORS_ORIGIN || "http://localhost:3001",
    credentials: true,
  });

  await app.register(rateLimit, {
    max: 100,
    timeWindow: "1 minute",
    errorResponseBuilder: () => ({
      statusCode: 429,
      error: "Too Many Requests",
      message: "Rate limit exceeded. Please try again later.",
    }),
  });

  // Register routes
  await app.register(healthRoutes, { prefix: "/health" });
  await app.register(authRoutes, { prefix: "/api/v1/auth" });
  await app.register(votingRoutes, { prefix: "/api/v1/vote" });
  await app.register(antifraudRoutes, { prefix: "/api/v1/antifraud" });
  await app.register(auditRoutes, { prefix: "/api/v1/audit" });

  // Global error handler
  app.setErrorHandler((error, request, reply) => {
    app.log.error(error);
    
    const statusCode = error.statusCode || 500;
    reply.status(statusCode).send({
      statusCode,
      error: error.name || "Internal Server Error",
      message: statusCode === 500 ? "An unexpected error occurred" : error.message,
      requestId: request.id,
    });
  });

  return app;
}

// Start server
async function start() {
  const app = await buildApp();
  
  try {
    await app.listen({ port: defaultConfig.port, host: defaultConfig.host });
    app.log.info(`Modern Anti-Fraud Voting API running on ${defaultConfig.host}:${defaultConfig.port}`);
  } catch (err) {
    app.log.error(err);
    process.exit(1);
  }
}

start();
