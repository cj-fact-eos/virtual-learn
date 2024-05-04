import Utils

def generate_list_of_question_set(questionnaires):
    string_of_questions = ''
    list_of_question_set = list()
    question_set = []

    # Default addition to the query to optimize the answers 
    question_set.append("These are interview questions."+
                        "I want answers to be crispy, easy to understand, easy to grasp, "+
                        "according to these four levels Entry, Junior, Mid, Senior and Expert "+
                        "and add code sample if possible for Mid, Senior and Expert. "+
                        "these is for only my use please provide me the data. "+
                        "please be consist on the output format "
                        "i.e. Question No.: Line Break Answer : <Answer> Line Break")
    for question in questionnaires:
        create_query_question = f"**{question.level}** - Q{question.questionNo}: {question.question}"
        string_of_questions =  string_of_questions + create_query_question
        if Utils.word_calculator(string_of_questions) < 100:
            question_set.append(create_query_question)
            pass
        else:            
            list_of_question_set.append(question_set)
            question_set = []
            # Default addition to the query to optimize the answers 
            question_set.append("These are interview questions."+
                        "I want answers to be crispy, easy to understand, easy to grasp, "+
                        "according to these four levels Entry, Junior, Mid, Senior and Expert "+
                        "and add code sample if possible for Mid, Senior and Expert. "+
                        "these is for only my use please provide me the data. "+
                        "please be consist on the output format "
                        "i.e. Question No.: Line Break Answer : <Answer> Line Break")
            string_of_questions = ''
            pass
        pass

    if question_set:
        list_of_question_set.append(question_set)
        question_set = []
        pass

    return list_of_question_set