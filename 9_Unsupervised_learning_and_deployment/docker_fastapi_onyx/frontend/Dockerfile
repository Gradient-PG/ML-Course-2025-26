FROM node:18-alpine
WORKDIR /app

# Copy files and install dependencies
COPY . .
RUN npm install

# Build the app to the /dist folder
RUN npm run build

# Serve the app using Vite's preview server
# --host is required to expose the app outside the container
EXPOSE 4173
CMD ["npm", "run", "preview", "--", "--host"]