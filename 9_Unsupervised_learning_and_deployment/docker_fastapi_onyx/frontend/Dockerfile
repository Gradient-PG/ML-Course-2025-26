FROM node:18-alpine
WORKDIR /app

ARG VITE_BACKEND_URL=http://localhost:8080
ENV VITE_BACKEND_URL=$VITE_BACKEND_URL

COPY . .
RUN npm install

RUN npm run build

EXPOSE 4173
CMD ["npm", "run", "preview", "--", "--host"]