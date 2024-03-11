docker build -t descritores .
docker stop descritores
docker rm descritores
docker run -d -p 8999:8999 --restart unless-stopped --name descritores descritores