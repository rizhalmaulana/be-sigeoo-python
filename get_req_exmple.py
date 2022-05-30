import requests as rq

api_link = "https://jsonplaceholder.typicode.com/posts/2"
output = rq.get(api_link)

print("JSON: ")
print(output.json())