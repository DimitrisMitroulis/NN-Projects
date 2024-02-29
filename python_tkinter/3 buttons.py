# button,checkbutton,radiobutton
import tkinter as tk
from tkinter import ttk

def button_func():
    print('Test')
    

#create window
window = tk.Tk()
window.title('Window and widget')
window.geometry('800x500')


button_string = tk.StringVar(value = 'A button with string var')
button = ttk.Button(window,text = "A simple Button", command = button_func,textvariable = button_string)
button.pack()

check = ttk.Checkbutton(window,text = "A simple Button")
check.pack()
check.config(takefocus='disabled')


window.mainloop()