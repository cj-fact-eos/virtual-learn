from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service

from QuestionQuery import QuestionQuery 

def question_level_extractor(url):
    
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')  # Optional for headless browser (comment out if you prefer a visible window)

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    driver.get(url)

    wait = WebDriverWait(driver, 20)  # Set a 10-second wait timeout
    wait.until(EC.presence_of_element_located((By.CLASS_NAME, "badge-add-to-pdf")))  # Replace with your wait condition

    html_content = driver.page_source

    soup = BeautifulSoup(html_content, 'html.parser')
    
    questionnaires = []
    questionNo = 0;
    target_spans = soup.select("[class~=badge-add-to-pdf]")
    for target_span in target_spans:
        if target_span:
            previous_sibling_div = target_span.find_parent("div")        
            if previous_sibling_div:
                level = ''
                question = ''
                
                get_level_div = previous_sibling_div.select_one("span.clickable.badge")
                if get_level_div:
                    level = get_level_div.text.strip()
                    
                get_question_div = previous_sibling_div.findPreviousSibling("div")
                if get_question_div:
                    get_h2_data = get_question_div.select_one('h2')
                    span_data = get_h2_data.select('span[class]')
                    for value in span_data:
                        value.decompose()
                        pass
                    question = get_h2_data.text.strip()
                    
                questionNo= questionNo + 1
                questionnaires.append(QuestionQuery(questionNo, question,level))
            else:
                print("div not found.")
        else:
            print("div not found.")
        pass

    driver.quit()
    return questionnaires

question_list = question_level_extractor('https://www.fullstack.cafe/interview-questions/spring')
for question in question_list:
    print(f"Q{question.questionNo}: {question.question}({question.level}")
    pass