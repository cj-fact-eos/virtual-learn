import pycurl
from io import BytesIO
import json

def query_google_api(json_data, api_key):
    headers = ['Content-Type: application/json']
    
    post_data = json.dumps(json_data)

    buffer = BytesIO()
    c = pycurl.Curl()
    c.setopt(c.URL, 
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key="
                +api_key)
    c.setopt(c.HTTPHEADER, headers)
    c.setopt(c.POSTFIELDS, post_data)
    c.setopt(c.WRITEDATA, buffer)
    c.perform()
    c.close()

    response = buffer.getvalue()    
    # print(response.decode("utf-8"))
    return response

def extract_text_from_response(response):
    # Assuming the response is stored in a variable named 'response'
    data = json.loads(response)  # Convert JSON string to a Python dictionary

    text_part= ''

    # Access the text content within the first candidate (index 0)
    if data["candidates"][0]:
        content = data.get("candidates", [])[0].get("content", "NO_CONTENT")
        if content != "NO_CONTENT":
            text_part = data["candidates"][0]["content"]["parts"][0]["text"]
        else:
            print("Fail to retrieve")

    return text_part