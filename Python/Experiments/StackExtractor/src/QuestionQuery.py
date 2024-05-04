class QuestionQuery:
    questionNo = 0  # Public by default, but convention suggests internal use
    question = ''
    level = ''

    def __init__(self, questionNo, question, level):
        self.questionNo = questionNo
        self.question = question
        self.level = level

    def __str__(self):
        return f"**{self.level}** - Q{self.questionNo}: {self.question}"

