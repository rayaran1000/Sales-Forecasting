import sys
from src.logger import logging

#Function that will give us a error message, and the structure is defined for the message here wrt custom exception.
def error_message_details(error,error_detail:sys):
     _,_,exc_tb = error_detail.exc_info()
     file_name=exc_tb.tb_frame.f_code.co_filename
     error_message="Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
     file_name,exc_tb.tb_lineno,str(error))

     return error_message
     

#Here we are creating our own custom exception
#Whenever I raise Custom Exception , it is inheriting the parent exception, and the error message we get from above if initialized here to a custom variable error_message.
class CustomException(Exception):
     def __init__(self,error_message,error_detail:sys):
          super().__init__(error_message) #Inheriting the exception class above (over riding the init method)
          self.error_message = error_message_details(error_message,error_detail=error_detail)

     def __str__(self):
          return self.error_message #Here we will be printing the error message we are getting.