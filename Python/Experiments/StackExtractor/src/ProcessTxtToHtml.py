from QuestionQuery import QuestionFormat
import re

# def ProcessTxtToQuestionAnswerListV2(filePath):
#     lines = ''
#     # Open the file in read mode ("r")
#     with open(filePath, "r") as f:
#         # Read the entire file contents
#         lines  = f.read()
#     pass
#     pattern_question = r"\*\*Q(\d+):"
#     data = lines.split(pattern_question)
#     print(data)
# pass


def ProcessTxtToQuestionAnswerList(filePath):
    lines = ''
    question_answers = list() 
    question = '' 
    answer = ''
    level = ''
    codeSample = ''
    pattern_question = r"Q\d+:"


    # Open the file in read mode ("r")
    with open(filePath, "r", encoding="utf-8") as f:
        # Read the entire file contents
        lines  = f.readlines()
        pass

        question_answer = ''
        temp_question = 0

        for index, line in enumerate(lines):
            if(line.strip().startswith('**Q') or re.search(pattern_question, line.strip())):
                if temp_question == 1:
                    temp_question = 0
                    question_answers.append(QuestionFormat(question, answer, level, codeSample))
                    question = ''
                    answer = ''
                    level = ''
                    codeSample = ''
                    if(line.strip().startswith('**Q') or re.search(pattern_question, line.strip())): # yeah i know it this idiotic but sorry
                        question =  line.strip() + '\n'
                        level = extract_level_from_question(question)
                        temp_question = temp_question + 1
                        pass
                    pass
                else:
                    question =  line.strip() + '\n'
                    level = extract_level_from_question(question)
                    temp_question = temp_question + 1
                    pass
                pass
            elif (line.strip().startswith('**Answer') or 
                    line.strip().startswith('Answer') or
                    line.strip().startswith('-Answer')):
                answer =  line.strip() + '\n'
                pass
            elif (line.strip().startswith('**Sample Code:')):
                codeSample = line.strip() + '\n'
                pass
            else:
                answer_sample_code = check_answer_or_sample_code(lines, index)
                if (answer_sample_code == 'answer'):
                    answer = answer +  line.strip() + '\n'
                    pass
                elif (answer_sample_code == 'sample'):
                    codeSample = codeSample + line.strip() + '\n'
                    pass
                else:
                    #ignore 
                    pass
                pass
        return question_answers

def convert_to_formatted_divs(question_answers):
    formatted_div = ''
    div_collection = []
    for q_ans in question_answers:
        formatted_div = '<div>'
        if(q_ans.Question):
            formatted_div = f"<h2>{q_ans.Question}</h2>"
            pass
        if(q_ans.Answer):
            formatted_div = formatted_div + f"<p class='answer''>{q_ans.Answer}</p>"
            pass
        if(q_ans.CodeSample):
            formatted_div = formatted_div + "<pre class='code-sample'>"+ f"{q_ans.CodeSample}</pre></code>" 
            pass
        formatted_div = formatted_div + '</div>'
        div_collection.append(formatted_div)
        formatted_div = ''
    pass
    return div_collection

def check_answer_or_sample_code(lines, index):
    while(index != 0):
        current_line = lines[index].strip()
        if (current_line.strip().startswith('**Answer') or 
                current_line.strip().startswith('Answer') or
                current_line.strip().startswith('-Answer')):
            return 'answer'
        elif (current_line.strip().startswith('**Sample Code:')):
            return 'sample'
        else:
            index = index - 1
        pass
    pass
    return ''

def extract_level_from_question(question):
    list_of_level = ['entry', 'junior', 'mid', 'senior', 'expert']
    question_lower = question.lower() 
    for level in list_of_level:
        if level in question_lower:
            return level  # Return the level if found
    pass
    return ''  # Return None if no level is found


def extract_question_no(question):
    
    pass

# fileName_path = f"outputDir/aspnet-mvc_151132.txt"
# question_List = ProcessTxtToQuestionAnswerList(fileName_path)
# formatted_div = convert_to_formatted_divs(question_List)
# for value in formatted_div:
#     print(value)
# pass    

