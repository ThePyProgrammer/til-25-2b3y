docker build -t "2b3y-surprise:$1" .
docker run -p 5005:5005 "2b3y-surprise:$1"