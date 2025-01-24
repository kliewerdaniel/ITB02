# Frontend Dockerfile
FROM node:18-alpine

WORKDIR /app

# Install dependencies
COPY frontend/package*.json ./
RUN npm install

# Copy frontend source
COPY frontend/ ./frontend/
# Build the application
RUN npm run build

# Expose port
EXPOSE 3000

# Start command
CMD ["npm", "start"]