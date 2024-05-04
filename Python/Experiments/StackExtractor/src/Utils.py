from urllib.parse import urlparse

def generate_google_json_data(generated_string):
    text_json = []
    for string in generated_string:
        text_json.append({"text": string})
    pass

    post_data = {
                    "contents": [  
                        {
                            "role": "user", 
                            "parts": text_json
                        }
                    ]
                }
    return post_data

def word_calculator(text):
    words = text.split(' ')
    return len(words)

def save_to_file(fileName, content):
    # Open a file for writing (will overwrite existing content)
    with open(fileName, "w") as file:
    # Write your content to the file
        file.write(content)
        
def extract_the_last_path(url):
    # Parse the URL
    parsed_url = urlparse(url)

    try:
        # Extract the last path element (test2)
        last_option = parsed_url.path.split("/")[-1]
        return last_option
    except:  # Catch specific or general exception types
        # Code to handle the exception
        return ''
