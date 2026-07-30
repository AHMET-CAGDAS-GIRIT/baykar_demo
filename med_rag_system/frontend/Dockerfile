# ==========================================
# 1. AŞAMA: Build (Zero-CVE Chainguard Node)
# ==========================================
FROM cgr.dev/chainguard/node:latest-dev AS build

WORKDIR /app

# Paket dosyalarını kopyala ve yükle
COPY package*.json ./
RUN npm install -D tailwindcss postcss autoprefixer
RUN npm ci

COPY . .
RUN npm run build

# ==========================================
# 2. AŞAMA: Nginx Prod Servisi
# ==========================================
FROM nginx:alpine

# Sistem paketlerini en son güvenlik yamalarıyla güncelle
RUN apk update && apk upgrade --no-cache

# Build edilen dosyaları aktar
COPY --from=build /app/dist /usr/share/nginx/html

# Güvenlik izinleri & Non-root çalışabilme ayarları
RUN chown -R nginx:nginx /usr/share/nginx/html && \
    chmod -R 755 /usr/share/nginx/html && \
    touch /var/run/nginx.pid && \
    chown -R nginx:nginx /var/run/nginx.pid /var/cache/nginx /var/log/nginx

USER nginx

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]