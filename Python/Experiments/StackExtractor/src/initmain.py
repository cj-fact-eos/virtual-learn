from dotenv import load_dotenv
import os
from QueryGoogleApi import query_google_api
from QueryGoogleApi import extract_text_from_response
import QuestionScrapper
import time
import Utils
import QuestionProcessor



load_dotenv()
api_key = os.getenv('GAPI_KEY')

# url = 'https://www.fullstack.cafe/interview-questions/strings'
print('Url allowed are  https://www.fullstack.cafe/interview-questions/<TheTopicAvailableInFullStack>')
print('e.g. https://www.fullstack.cafe/interview-questions/strings')
url = input("Enter url and wait for some time ")

questionnaires = QuestionScrapper.question_level_extractor(url)
list_of_question_set = QuestionProcessor.generate_list_of_question_set(questionnaires)

total_sets = len(list_of_question_set) 
time_taken = round(((total_sets * 25) + 25)/60, 2);
print(f"Total time taken will be {(time_taken)} mins.")

output = ''
index = 0

while index < len(list_of_question_set):
    if index != 0:
        print('we are processing.. wait time of 25 sec begin..')
        time.sleep(25)
    pass
    
    output = output + extract_text_from_response(
                        query_google_api(
                            Utils.generate_google_json_data(list_of_question_set[index]),
                            api_key))
    print( Utils.word_calculator(output))
    index += 1

Utils.save_to_file('outputData.txt', output)


