import requests as rq

data_object = {'userId': 1, 'id': 123, 'title': 'Example Title'}
api_link = "https://jsonplaceholder.typicode.com/posts/"

output = rq.post(api_link, data_object)
print("Content: ")
print(output.content)