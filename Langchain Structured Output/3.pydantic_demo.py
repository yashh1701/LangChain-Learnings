from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):

    name: str = 'nitish'
    age: Optional[int] = None # thsis an optional field, which means it can be None or not provided.
    email: EmailStr    # this is a field that will validate the email address provided. It will raise an error if the email address is not valid.
    cgpa: float = Field(gt=0, lt=10, default=5, description='A decimal value representing the cgpa of the student')


new_student = {'age':'32', 'email':'abc@gmail.com'}

student = Student(**new_student)

student_dict = dict(student)

print(student_dict['age'])

student_json = student.model_dump_json()