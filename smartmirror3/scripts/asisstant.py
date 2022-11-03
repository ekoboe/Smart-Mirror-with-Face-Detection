import pyttsx3
import speech_recognition as sr
import datetime
import wikipedia
from flask import redirect

# this method is for taking the commands
# and recognizing the command from the
# speech_Recognition module
def takeCommand():

	r = sr.Recognizer()

	# Microphone module
	# for listening the command
	with sr.Microphone() as source:
		print('Listening')
		
		# seconds of non-speaking audio before
		# a phrase is considered complete
		r.pause_threshold = 0.7
		audio = r.listen(source)
		
		# try and catch method so that if sound is recognized
		# it is good else we will have exception
		# handling
		try:
			print("Recognizing")
			
			# for Listening the command in indonesia
			# english we can also use 'hi-Idn'
			# for indonesia recognizing
			Query = r.recognize_google(audio, language='en-idn')
			print("the command is printed=", Query)
			
		except Exception as e:
			print(e)
			print("Say that again")
			return "None"
		
		return Query

def speak(audio):
	
	engine = pyttsx3.init()
	# getter method(gets the current value
	# of engine property)
	voices = engine.getProperty('voices')
	
	# setter method .[0]=male voice and
	# [1]=female voice in set Property.
	engine.setProperty('voice', voices[0].id)
	
	# Method for the speaking of the the assistant
	engine.say(audio)
	
	# Blocks while processing all the currently
	# queued commands
	engine.runAndWait()

def tellDay():
	
	# This function is for telling the
	# day of the week
	day = datetime.datetime.today().weekday() + 1
	
	#this line tells us about the number
	# that will help us in telling the day
	Day_dict = {1: 'Monday', 2: 'Tuesday',
				3: 'Wednesday', 4: 'Thursday',
				5: 'Friday', 6: 'Saturday',
				7: 'Sunday'}
	
	if day in Day_dict.keys():
		day_of_the_week = Day_dict[day]
		print(day_of_the_week)
		speak("The day is " + day_of_the_week)


def tellTime():
	
	# This method will give the time
	time = str(datetime.datetime.now())
	
	# the time will be displayed like
	# this "2020-06-05 17:50:14.582630"
	#nd then after slicing we can get time
	print(time)
	hour = time[11:13]
	min = time[14:16]
	speak("The time is " + hour + "Hours and" + min + "Minutes")	

def Hello():
	
	# This function is for when the assistant
	# is called it will say hello and then
	# take query
	speak("hello, I am your assistant. / Tell me how may I help you")


def Take_query():

	# calling the Hello function for
	# making it more interactive
	Hello()
	
	# This loop is infinite as it will take
	# our queries continuously until and unless
	# we do not say bye to exit or terminate
	# the program
	while(True):
		
		# taking the query and making it into
		# lower case so that most of the times
		# query matches and we get the perfect
		# output
		query = takeCommand().lower()
		if "open youtube" in query:
			speak("Opening youtube ")
			
			# in the open method we just to give the link
			# of the website and it automatically open
			# it in your default browser
			redirect("www.youtube.com")
			continue
		
		elif "open google" in query:
			speak("Opening Google ")
			redirect("www.google.com")
			continue
			
		elif "which day it is" in query:
			tellDay()
			continue
		
		elif "tell me the time" in query:
			tellTime()
			continue

		elif "play" in query:
			speak("Opening in youtube")
			indx = query.split().index('youtube')
			qr = query.split()[indx + 1:]
			redirect("http://www.youtube.com/results?search_query =" + '+'.join(qr))
			continue

		elif "search" in query:
			speak("Searching in google")
			indx = query.split().index('google')
			qr = query.split()[indx + 1:]
			redirect("https://www.google.com/search?q =" + '+'.join(qr))
			continue

		# this will exit and terminate the program
		elif "bye" in query:
			speak("Bye. Check Out GFG for more exicting things")
			exit()
		
		elif "from wikipedia" in query:
			
			# if any one wants to have a information
			# from wikipedia
			speak("Checking the wikipedia ")
			query = query.replace("wikipedia", "")
			
			# it will give the summary of 4 lines from
			# wikipedia we can increase and decrease
			# it also.
			result = wikipedia.summary(query, sentences=4)
			speak("According to wikipedia")
			speak(speak)
		
		elif "tell me your name" in query:
			speak("I am Jarvis. Your deskstop Assistant")
	
      

if __name__ == '__main__':
	
	# main method for executing
	# the functions
	Take_query()