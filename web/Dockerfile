# syntax=docker/dockerfile:1

# ---- build the static React bundle --------------------------------------
FROM node:20-alpine AS build

WORKDIR /app

COPY package.json package-lock.json* ./
# use a container-local cache to avoid uid-permission issues
RUN npm install --cache /tmp/npm-cache

COPY index.html tsconfig.json vite.config.ts ./
COPY src ./src
RUN npm run build

# ---- static server + reverse proxy to the API ---------------------------
FROM nginx:1.27-alpine AS runtime

# remove the default site; add our template (envsubst on container start)
RUN rm /etc/nginx/conf.d/default.conf
COPY nginx/default.conf.template /etc/nginx/templates/default.conf.template
COPY --from=build /app/dist /usr/share/nginx/html

# busybox wget ships with alpine
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=5 \
    CMD wget -q -O - http://127.0.0.1/healthz | grep -q '"ok"'

EXPOSE 80
