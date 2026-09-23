import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In local development the browser talks to the API through this proxy.
// In Docker the nginx image performs the same reverse proxy.
export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 5173,
    proxy: {
      "/api": {
        target: process.env.VITE_API_PROXY_TARGET ?? "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
});
