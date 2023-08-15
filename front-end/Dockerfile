# Use an official Node runtime as the base image
FROM node:18.17.0


VOLUME [ "/app/front-end" ]
# Set the working directory in the container to /front-end
WORKDIR /app

# Copy the current directory contents into the container at /front-end
COPY . /app/front-end

# Install any needed packages specified in package.json
WORKDIR /app/front-end

COPY package.json app.json ./

RUN npm install --legacy-peer-deps

# Run the app when the container launches
WORKDIR /app/front-end

EXPOSE 19000
EXPOSE 19001
EXPOSE 19002

CMD [ "npx, start" ]