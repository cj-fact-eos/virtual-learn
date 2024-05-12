from dotenv import load_dotenv
import os
from HtmlProcessor import create_html
from ProcessTxtToHtml import ProcessTxtToQuestionAnswerListV2, convert_to_formatted_divs
from QueryGoogleApi import query_google_api
from QueryGoogleApi import extract_text_from_response
import QuestionScrapper
import time
import Utils
import QuestionProcessor
import datetime

load_dotenv()
api_key = os.getenv('GAPI_KEY')

pdf_txt_fileName = ''

# url = 'https://www.fullstack.cafe/interview-questions/strings'
print('Url allowed are  https://www.fullstack.cafe/interview-questions/<TheTopicAvailableInFullStack>')
print('e.g. https://www.fullstack.cafe/interview-questions/strings')

url = input("Enter or paste the url and wait for some time:  ")
if url:
    pdf_txt_fileName = Utils.extract_the_last_path(url)
    title = pdf_txt_fileName
    current_time = datetime.datetime.now()
    time_string = current_time.strftime("%H%M%S")
    if  pdf_txt_fileName == '':
        pdf_txt_fileName = 'output_file'+"_"+time_string
        pass
    else:
        pdf_txt_fileName = pdf_txt_fileName+"_"+time_string
        pass

    questionnaires = QuestionScrapper.question_level_extractor(url)

    list_of_question_set = QuestionProcessor.generate_list_of_question_set(questionnaires)

    total_sets = len(list_of_question_set) 
    time_taken = round(((total_sets * 25) + 25)/60, 2);
    print(f"Total question set to be processed {total_sets} with {len(questionnaires)} question(s).")
    print(f"Total time taken will be {(time_taken)} mins.")

    if total_sets > 0:
        output = ''
        index = 0

        while index < len(list_of_question_set):
            if index != 0:
                print('we are processing.. wait time of 25 sec begin..')
                time.sleep(25)
            pass
            extract_output = extract_text_from_response(
                                query_google_api(
                                    Utils.generate_google_json_data(list_of_question_set[index]),
                                    api_key))
            Utils.save_to_file(f"outputDir/{index}_{ datetime.datetime.now().strftime("%H%M%S")}",
                            extract_output)
            output = output + "\n\n" + extract_output
            #print( Utils.word_calculator(output))
            index += 1

        fileName_path = f"outputDir/{pdf_txt_fileName}.txt"
        Utils.save_to_file(fileName_path, output)

        question_answers = ProcessTxtToQuestionAnswerListV2(fileName_path)
        formatted_div = convert_to_formatted_divs(question_answers)
        create_html(title, formatted_div, pdf_txt_fileName)
    pass
pass

# Process file 


