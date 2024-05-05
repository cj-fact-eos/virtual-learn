import Utils

def generate_list_of_question_set(questionnaires):
    string_of_questions = ''
    list_of_question_set = list()
    question_set = []


    for question in questionnaires:
        if len(question_set) == 0:
            # Default addition to the query to optimize the answers 
            question_set.append("These are interview questions."+
                                "I want answers to be crispy, easy to understand, easy to grasp, "+
                                "form your answer according to these four levels Entry, Junior, Mid, Senior and Expert "+
                                " software developer/engineers and add code sample if possible for Mid, Senior and Expert. "+
                                "these is for only my use please provide me the data. "+
                                "please be consist on the output format "
                                "i.e. **<Question>? (Level) Line Break "+
                                " **Answer : <Answer> Line Break "+
                                " **Sample Code : `<sample code>`")
            pass
        
        create_query_question = f"**{question.level}** - Q{question.questionNo}: {question.question}"
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