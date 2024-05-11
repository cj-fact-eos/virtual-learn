import Utils

def generate_list_of_question_set(questionnaires):
    string_of_questions = ''
    list_of_question_set = list()
    question_set = []
    # Default addition to the query to optimize the answers 
    question_set.append( f"I am providing you questions \
                                please craft the answers that are easy to understand, easy to grasp \
                                for software developer/engineers and add sample code \
                                if possible \
                                I want to give interview so this is for my personal use.\
                                please make sure output format provided is consist with format provided below \
                                \n\n **<Question>? \n\n \
                                \n\n **Answer : <Answer> \n\n \
                                \n\n **Sample Code : `<sample code>` ")
    for question in questionnaires:
        if len(question_set) == 0:
                
                pass
        
        create_query_question = f"Q{question.questionNo}: {question.question} - {question.level}"
        string_of_questions =  string_of_questions + create_query_question
        
        question_set.append(create_query_question) 
        if Utils.word_calculator(string_of_questions) > 100:
            list_of_question_set.append(question_set)
            question_set = []
            string_of_questions = ''
            pass
        pass

    if question_set:
        list_of_question_set.append(question_set)
        question_set = []
        pass

    return list_of_question_set