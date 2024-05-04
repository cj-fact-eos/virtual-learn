
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