from HtmlProcessor import create_html
from QuestionQuery import QuestionFormat
import re

from Utils import replace_only_leading_space

def ProcessTxtToQuestionAnswerListV2(filePath):
    lines = ''
    question_answers = list() 
    pattern_question = r"Q\d+:"

    # Open the file in read mode ("r")
    with open(filePath, "r") as f:
        # Read the entire file contents
        lines  = f.read()
    pass
    pattern = r"\*\*[A-Z](\d+):"  # Capture digits into a group
    digit_pattern = r"^\d+$" 
    splitted_data = re.split(pattern, lines)
    
    problematic_pattern = r"Q(\d+):"  # Capture digits into a group

    # Remove any empty elements at the beginning or end (optional)
    queried_Data = [item for item in splitted_data if item.strip() and \
                    (re.search(digit_pattern, item.strip()) == None)]
    
    list_of_prob_qa = list()
    # problematic_data = re.split(problematic_pattern, queried_Data)
    # problematic_queried_Data = [item for item in problematic_data if item.strip() and \
    #                 (re.search(digit_pattern, item.strip()) == None)]
    for question_index, data in enumerate(queried_Data):
        question = ''
        answer = ''
        level = ''
        codeSample = ''

        line_Breaks = re.split('\n', data)
        for index, actual_data in enumerate(line_Breaks):            
            if index == 0:
                question = actual_data
                pass
            else:
                if question_index != 0 and re.search(problematic_pattern, actual_data):
                    #ignoring since it will be processed differently
                    list_of_prob_qa.append(data)
                    break                
                answer = answer + "\n"+ actual_data
                pass
            pass
        
        question_answers.append(QuestionFormat(question_index + 1, question, answer, level, codeSample))
        
    pass

    for question_index, data in enumerate(list_of_prob_qa):
        question = ''
        answer = ''
        level = ''
        codeSample = ''
        line_Breaks = re.split('\n', data)

        for index, actual_data in enumerate(line_Breaks):
            if (re.search(problematic_pattern,actual_data) or 
                re.search(pattern, actual_data)):                
                    total_qa = len(question_answers)
                    question_answers.append(QuestionFormat(total_qa + 1, \
                            question, answer, level, codeSample))
                    question = ''
                    answer = ''
                    level = ''
                    codeSample = ''
                    question = actual_data
                    pass
            elif (actual_data.strip() == ''):
                pass
            else:
                if question.strip() != '':
                    answer = answer + "\n"+ actual_data
                    pass
    pass
    return question_answers
pass

def convert_to_formatted_divs(question_answers):
    formatted_div = ''
    div_collection = []
    for q_ans in question_answers:
        formatted_div = '<div>'
        if(q_ans.Question):
            if re.search('<script>', q_ans.Question.strip()):
                question = q_ans.Question.replace('<script>', '&lt;script&gt;')
                formatted_div = formatted_div + f"<h2>{q_ans.QuestionNo}: {question}</h2>"
                continue 
            formatted_div = formatted_div + f"<h2>{q_ans.QuestionNo}: {q_ans.Question}</h2>"
            pass
        if(q_ans.Answer):
            answer = format_answer(q_ans.Answer)
            formatted_div = formatted_div + f"<p class='answer'>{answer}</p>"
            pass
        if(q_ans.CodeSample):
            #formatted_div = formatted_div + "<pre class='code-sample'>"+ f"{q_ans.CodeSample}</pre></code>" 
            pass
        formatted_div = formatted_div + '</div>'
        div_collection.append(formatted_div)
        formatted_div = ''
    pass
    return div_collection

def extract_level_from_question(question):
    list_of_level = ['entry', 'junior', 'mid', 'senior', 'expert']
    question_lower = question.lower() 
    for level in list_of_level:
        if level in question_lower:
            return level  # Return the level if found
    pass
    return ''  # Return None if no level is found

def replace_newline_with_br(text):
    text = text.replace('\n\n', '\n')
    return text.replace('\n', '<br />')

def format_answer(answer):    
    answer_lines = re.split('\n', answer)
    formatted_answer = ''
    li_answer = r"^\* "
    add_ul = 0
    add_pre = 0
    total_answer_lines = len(answer_lines)
    for index, answer_line in enumerate(answer_lines):
        if re.search('<script>', answer_line.strip()):
            answer_line = answer_line.replace('<script>', '&lt;script&gt;')
            pass
        if answer_line == '':
            if add_ul != 0:
                if total_answer_lines > index :
                    checker_index = index + 1                    
                    forwardCheck = index if checker_index == total_answer_lines else checker_index
                    if (re.search(li_answer, answer_lines[forwardCheck].strip()) == None):
                        formatted_answer = formatted_answer + '</ul>'
                        add_ul = 0
                pass
            else:
                if formatted_answer != '':
                    formatted_answer = formatted_answer + '<br />'
            pass
        elif answer_line.startswith("* "):
            if add_ul == 0: 
                formatted_answer = formatted_answer + "<ul><li>" + answer_line + '</li>'
                add_ul = add_ul + 1
                pass
            else:
                formatted_answer = formatted_answer + "<li>" + answer_line + '</li>'
                pass
            pass
        elif answer_line.startswith(' '):
            formatted_answer = formatted_answer + replace_only_leading_space(answer_line) + '<br />' 
        elif re.search('```', answer_line.strip()):
            if(add_pre == 0):
                formatted_answer = formatted_answer + answer_line.replace('```', "<pre class='code-sample'>")
                add_pre = add_pre + 1
            else:
                formatted_answer = formatted_answer + answer_line.replace('```', '</pre>')
                add_pre = 0
                pass
        else:
            formatted_answer = formatted_answer + answer_line + '<br />'
            pass
        pass
    
    formatted_answer = formatted_answer.replace("**", "")
    formatted_answer = formatted_answer.replace("*", "")
    #formatted_answer = formatted_answer.replace("<br /><br />", "<br />")
    return formatted_answer


fileName_path = f"outputDir/angular_133627.txt"
question_List = ProcessTxtToQuestionAnswerListV2(fileName_path)
formatted_div = convert_to_formatted_divs(question_List)
create_html('angular', formatted_div, 'angular_133627')